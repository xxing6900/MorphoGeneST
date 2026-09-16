import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum

# ==========================================================
# 辅助层: 自适应通道归一化 (解决分辨率不匹配的核心) 
# 即便输入是 112x112 或 224x224 都能自动兼容
# ==========================================================
class LayerNormChannels(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x):
        # x shape: (N, C, H, W)
        # 将通道移到最后 (N, H, W, C), 然后在 Channel 上归一化, 再换回来
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

# ==========================================================
# 1. 大核卷积图像提取器 (SOTA ConvNeXt Style)
# ==========================================================
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        # 1. 7x7 Depthwise Conv (大感受野，捕捉宏观病理组织结构)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # 2. Inverted Bottleneck (1x1 -> GELU -> 1x1)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class PathologyExtractor(nn.Module):
    def __init__(self, in_chans=3, embed_dim=256):
        super().__init__()
        # Stem 层: 缩小分辨率并增加通道数
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, kernel_size=4, stride=4),
            LayerNormChannels(embed_dim // 4) # 使用尺寸自适应 Norm!
        )
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(embed_dim // 4) for _ in range(2)])
        
        # Downsample 层
        self.downsample = nn.Sequential(
            LayerNormChannels(embed_dim // 4), # 使用尺寸自适应 Norm!
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(embed_dim) for _ in range(2)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.global_pool(x).flatten(1)
        return x

# ==========================================================
# 2. 空间感知图 Transformer (Spatial-Aware Graph Transformer)
# 无需构建邻接矩阵，直接对距离求权
# ==========================================================
class SpatialGraphAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # 距离映射 MLP
        self.distance_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(), # 这里换成 GELU 更好
            nn.Linear(16, num_heads)
        )
        
        # 🌟 关键修复：零初始化距离偏置！
        # 让模型一开始没有距离障碍，能够自由获取邻居特征，然后再慢慢精细化
        nn.init.constant_(self.distance_mlp[-1].weight, 0)
        nn.init.constant_(self.distance_mlp[-1].bias, 0)

    def forward(self, x, coords):
        B, N, C = x.shape
        # 生成查询、键、值
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 核心 1：计算节点间的语义/形态相关性
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, heads, N, N)

        # 核心 2：计算绝对物理距离偏置 (Distance Bias)
        dist_mat = torch.cdist(coords.float(), coords.float(), p=2) # (B, N, N)
        dist_bias = self.distance_mlp(dist_mat.unsqueeze(-1)) # (B, N, N, heads)
        dist_bias = dist_bias.permute(0, 3, 1, 2) # (B, heads, N, N)

        # 融合距离偏置（距离越远，bias 会自动抑制不同区域组织的过度平滑)
        attn = attn - dist_bias 
        
        attn_weights = attn.softmax(dim=-1)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class SAGTBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialGraphAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
    def forward(self, x, coords):
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.mlp(self.norm2(x))
        return x

# ==========================================================
# 3. 混合专家基因解码器 (Mixture-of-Experts Decoder)
# ==========================================================
class GeneExpert(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class MoE_Decoder(nn.Module):
    def __init__(self, in_dim, n_genes, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        # Router网络：决定特征需要激活哪几个基因专家
        self.router = nn.Sequential(
            nn.Linear(in_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([GeneExpert(in_dim, n_genes) for _ in range(num_experts)])
        
    def forward(self, x):
        # x shape: (N, in_dim)
        route_weights = self.router(x) # (N, num_experts)
        
        expert_outputs = []
        for i in range(self.num_experts):
            out_i = self.experts[i](x) # (N, n_genes)
            expert_outputs.append(out_i.unsqueeze(-1))
            
        expert_outputs = torch.cat(expert_outputs, dim=-1) # (N, n_genes, num_experts)
        
        # 加权融合不同专家的经验
        route_weights = route_weights.unsqueeze(1) # (N, 1, num_experts)
        final_out = (expert_outputs * route_weights).sum(dim=-1) # (N, n_genes)
        return final_out