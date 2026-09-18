# main.py
import os
import glob
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as transforms
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

# =========================================================================
# 导入自定义组件 (确保同一目录下存在以下文件)
# =========================================================================
from modules import PathologyExtractor, SAGTBlock, MoE_Decoder
from zinb import MeanAct, DispAct, ZINB_loss
from dataset import ViT_HER2ST 
from predict import get_R, cluster, test # 加入了 test 函数用于推理评估

warnings.filterwarnings('ignore')

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def seed_everything(seed=12000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 主模型结构
# ==========================================
class MorphoGeneST(pl.LightningModule):
    def __init__(self, n_genes=785, embed_dim=256, learning_rate=1e-4, 
                 bake=1, lamb=0.5, zinb_coef=0.25):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.bake = bake           
        self.lamb = lamb           
        self.zinb_coef = zinb_coef 
        
        self.image_encoder = PathologyExtractor(in_chans=3, embed_dim=embed_dim)
        self.graph_layers = nn.ModuleList([SAGTBlock(embed_dim) for _ in range(3)])
        
        self.head_mse = nn.Sequential(
            MoE_Decoder(embed_dim, n_genes, num_experts=4),
            nn.ReLU()
        )
        self.head_mean = nn.Sequential(MoE_Decoder(embed_dim, n_genes, num_experts=4), MeanAct())
        self.head_disp = nn.Sequential(MoE_Decoder(embed_dim, n_genes, num_experts=4), DispAct())
        self.head_pi = nn.Sequential(MoE_Decoder(embed_dim, n_genes, num_experts=4), nn.Sigmoid())

        self.distill_coef = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

        self.tf = transforms.Compose([ 
            transforms.RandomGrayscale(0.1), 
            transforms.RandomRotation(90), 
            transforms.RandomHorizontalFlip(0.2) 
        ])

    def forward(self, patches, centers, aug=False):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        
        chunk_size = 64
        spot_feats_list = []
        for i in range(0, B * N, chunk_size):
            chunk = patches[i : i + chunk_size]
            feat = self.image_encoder(chunk)
            spot_feats_list.append(feat)
        spot_feats = torch.cat(spot_feats_list, dim=0) 
        
        spot_feats = spot_feats.unsqueeze(0)     
        centers = centers.unsqueeze(0)             
        
        h = spot_feats
        for layer in self.graph_layers:
            h = layer(h, centers)
            
        h_final = h.squeeze(0) 
        
        pred_mse = self.head_mse(h_final)
        m = self.head_mean(h_final)
        d = self.head_disp(h_final)
        p = self.head_pi(h_final)
        
        if aug: 
            return pred_mse, (m, d, p), self.distill_coef(h_final)
            
        return pred_mse, (m, d, p)

    def training_step(self, batch, batch_idx):
        patch, position, exp_norm, adj, oris, sfs, center, *_ = batch 
        exp = exp_norm.squeeze(0)
        oris = oris.squeeze(0)
        sfs = sfs.squeeze(0)
        center = center.squeeze(0) 
        
        pred, (m, d, p) = self(patch, center)
        
        mse_loss = F.mse_loss(pred, exp)
        zinb_loss = ZINB_loss(oris, m, d, p, sfs)
        
        bake_loss = 0
        if self.bake > 0:
            bake_x = []
            for _ in range(self.bake):
                aug_patch = self.tf(patch.squeeze(0)).unsqueeze(0)
                aug_pred, _, weight = self(aug_patch, center, aug=True)
                bake_x.append((aug_pred.unsqueeze(0), weight.unsqueeze(0)))
                
            preds, coefs = zip(*bake_x)
            coef = F.softmax(torch.cat(coefs, 0), dim=0)
            distilled_pred = (torch.cat(preds, 0) * coef).sum(0)
            bake_loss = F.mse_loss(distilled_pred, pred)
            
        loss = mse_loss + 0.1 * zinb_loss + 0.1 * bake_loss 
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('mse_loss', mse_loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def run_experiment(fold=5, epochs=200, gpu_id=0):
    print(f"\n🚀 开始部署模型 | 目标切分 Fold: {fold} 🚀")
    seed_everything(12000)
    
    os.makedirs("./model_checkpointsljx30", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    train_dataset = ViT_HER2ST(train=True, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

    model = MorphoGeneST(n_genes=785, embed_dim=256)

    logger = TensorBoardLogger("logs", name=f"Fold_{fold}_ST_TME")
    checkpoint_callback = ModelCheckpoint(
        dirpath="./model_checkpointsljx30",
        filename=f"Fold{fold}-Best-{{epoch:03d}}",
        save_top_k=1, 
        monitor="train_loss", 
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        precision="16-mixed"
    )

    trainer.fit(model, train_loader)
    print(f"🎯 Fold {fold} 训练完成。\n")

# =========================================================================
# 🌟 新增功能模块: 基于P-value挑选Top4基因，定位最佳切片并生成空间图像
# =========================================================================
def local_test(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    preds_list, gts_list, cts_list = [], [], []
    
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(test_loader, desc="Testing"):
            patch, position, exp_norm, adj, oris, sfs, center, *_ = batch
            patch = patch.to(device)
            # 关键修复：把 center 的 Batch 维度去掉，保证输入 shape 为 [N, 2]
            center = center.to(device).squeeze(0) 
            
            # 使用 MorphoGeneST 正确的 forward 结构
            pred_mse, _ = model(patch, center, aug=False)
            
            preds_list.append(pred_mse.cpu().squeeze().numpy())
            gts_list.append(exp_norm.cpu().squeeze().numpy())
            cts_list.append(center.cpu().numpy())
            
    # 因为 batch size = 1，直接取第一个结果组装即可
    preds = preds_list[0]
    gt = gts_list[0]
    ct = cts_list[0]
    
    adata = ad.AnnData(preds)
    adata.obsm['spatial'] = ct
    adata_gt = ad.AnnData(gt)
    adata_gt.obsm['spatial'] = ct
    return adata, adata_gt


# =========================================================================
# 🌟 新增功能模块: 基于P-value挑选Top4基因，定位最佳切片并生成空间图像
# =========================================================================
def evaluate_and_visualize_top_genes(folds, gpu_id=0):
    print("\n[功能触发] 开始计算多切片综合 P-value 并可视化 Top 4 基因...")
    seed_everything(12000)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # 获取高变基因列表
    try:
        gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
    except FileNotFoundError:
        print("未找到基因列表 ./data/her_hvg_cut_1000.npy，请检查路径。")
        return
    
    all_neg_log10_p = [] # 用于存储每个 fold 的 -log10(P) 数组, list shape: [num_folds, 785]
    fold_adatas = {}     # 用于存储每个 fold 预测出的 AnnData，供作图使用
    tested_folds = []    # 记录成功加载进来的 fold

    # 1. 遍历收集各个切片 (folds) 的预测结果
    for fold in folds:
        test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)
        
        # 寻找训练保存的权重
        import glob
        ckpt_paths = glob.glob(f"./model_checkpointsljx25/Fold{fold}-Best-*.ckpt")
        if not ckpt_paths:
            print(f"⚠️ 未找到 Fold {fold} 的权重文件，跳过该切片。")
            continue
        best_ckpt = ckpt_paths[-1] # 如果有多个，取最后一个
        
        # 加载权重
        model = MorphoGeneST.load_from_checkpoint(best_ckpt, n_genes=785, embed_dim=256)
        
        # 调用修复好的 local_test 取代官方的 test
        adata_pred, adata_gt = local_test(model, test_loader, device=device)
        adata_pred.var_names = gene_list
        adata_gt.var_names = gene_list
        
        fold_adatas[fold] = adata_pred

        # 2. 计算每个基因的皮尔逊相关性及 P-value
        _, p_vals = get_R(adata_pred, adata_gt, dim=1) 
        
        # 计算 -log10(p)，加微小值防止 log(0) 或除0错误
        neg_log10_p = -np.log10(p_vals + 1e-300)
        all_neg_log10_p.append(neg_log10_p)
        tested_folds.append(fold)

    if not all_neg_log10_p:
        print("没有成功评估任何 Fold，请检查模型 Checkpoint 是否存在。")
        return

    # 3. 计算所有切片平均 -log10(P)，找出 P 值极度显著的前 4 个基因
    all_neg_log10_p = np.array(all_neg_log10_p)  # shape: (评估的fold数量, 785)
    mean_neg_log10_p = np.nanmean(all_neg_log10_p, axis=0)
    
    # 逆序排序取前四个的索引
    top4_indices = np.argsort(mean_neg_log10_p)[::-1][:4]
    
    os.makedirs('./figures', exist_ok=True)
    print(f"\n✅ 找到平均 -log10(P) 最高的前四个基因: {[gene_list[idx] for idx in top4_indices]}")
    
    # 4. 对这四个基因，寻找表现最佳的组织切片，并将其输出制图
    for rank, gene_idx in enumerate(top4_indices):
        gene_name = gene_list[gene_idx]
        
        # 寻找该基因在哪一个 fold 中取得了最大的 -log10(P) 值 (即最小的 P-value)
        best_fold_local_idx = np.argmax(all_neg_log10_p[:, gene_idx])
        best_fold = tested_folds[best_fold_local_idx]
        best_val = all_neg_log10_p[best_fold_local_idx, gene_idx]
        
        print(f"Top {rank+1} 基因 [{gene_name}] | 最佳切片/模型 Fold: {best_fold} | Max -log10(P): {best_val:.3f}")

        best_adata = fold_adatas[best_fold]
        
        import matplotlib.pyplot as plt
        plt.gcf().set_facecolor('white')
        
        # 将矩阵数值 Scale 有助于画图时去除异常极大极小值造成的颜色单调
        sc.pp.scale(best_adata)
        sc.pl.spatial(best_adata, img=None, color=gene_name, spot_size=150, color_map='magma', show=False)
        
        # 保存图像
        save_path = f'./figures/Top{rank+1}_Predicted_{gene_name}_Fold{best_fold}.pdf'
        plt.savefig(save_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"   => 图形已保存至: {save_path}")


if __name__ == '__main__':
    # 1. 假设您执行这几个折叠的实验训练
    eval_folds = [5, 11, 17, 23, 26, 30]
    
    # 注：如果您已经预训练过了，只需注释掉下方训练代码即可只执行测试画图
    # for f in eval_folds:
    #     run_experiment(fold=f, epochs=250, gpu_id=0)

    # 2. 执行综合指标提取、查找 Top4 Gene 并对最完美切片生成组织空间云图
    evaluate_and_visualize_top_genes(folds=eval_folds, gpu_id=0)