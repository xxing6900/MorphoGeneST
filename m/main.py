import os
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

# =========================================================================
# 导入自定义组件 (确保同一目录下存在以下文件)
# =========================================================================
from modules import PathologyExtractor, SAGTBlock, MoE_Decoder
from zinb import MeanAct, DispAct, ZINB_loss
from dataset import ViT_HER2ST 
# 🌟 关键修改：直接从 predict 模块中导入官方对齐的评测函数
from predict import get_R, cluster 

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
# 主模型结构保持不变
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
            nn.ReLU() # 加 ReLU 防负数干扰
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
        
        # Micro-Batching (微批次特征提取): 打断庞大的显存峰值
        chunk_size = 64
        spot_feats_list = []
        for i in range(0, B * N, chunk_size):
            chunk = patches[i : i + chunk_size]
            feat = self.image_encoder(chunk)
            spot_feats_list.append(feat)
        spot_feats = torch.cat(spot_feats_list, dim=0) # [B*N, embed_dim]
        
        spot_feats = spot_feats.unsqueeze(0)     
        centers = centers.unsqueeze(0)             
        
        h = spot_feats
        for layer in self.graph_layers:
            h = layer(h, centers)
            
        h_final = h.squeeze(0) # [N, embed_dim]
        
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
            
        # 强化主 MSE，强行拉高 PCC。zinb_coef 临时降为 0.1，lamb 降为 0.1
        loss = mse_loss + 0.1 * zinb_loss + 0.1 * bake_loss 
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('mse_loss', mse_loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def run_experiment(fold=5, epochs=200, gpu_id=0):
    print(f"\n🚀 开始部署 ST-TransFormer-MoE 联合创新模型 | 目标切分 Fold: {fold} 🚀")
    seed_everything(12000)
    
    os.makedirs("./model_checkpointsljx25", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    print("[1/4] 正在加载组织病理学与 ST 数据集...")
    train_dataset = ViT_HER2ST(train=True, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    
    # 🌟 关键修改：完全复刻 Hist2ST 的原生标签提取方式（用于后面剔除 undetermined 与 KMeans比较）
    label = test_dataset.label[test_dataset.names[0]]
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    print("[2/4] 初始化网络结构与 Pytorch Lightning Trainer...")
    model = MorphoGeneST(
        n_genes=785, 
        embed_dim=256, 
        learning_rate=1e-4, 
        bake=1, 
        lamb=0.5, 
        zinb_coef=0.25
    )

    logger = TensorBoardLogger("logs", name=f"Fold_{fold}_ST_TME")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./model_checkpointsljx25",
        filename=f"Fold{fold}-Best-{{epoch:03d}}-{{train_loss:.4f}}",
        save_top_k=3, 
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

    print(f"[3/4] 启动模型训练 (共计 {epochs} 轮)...")
    trainer.fit(model, train_loader)

    print("\n[4/4] 训练完成，进入全图推理与下游任务指标(ARI & PCC)评测...")
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    preds_list, gts_list, centers_list = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            patch, position, exp_norm, adj, oris, sfs, center, *_ = batch
            patch = patch.to(device)
            center = center.to(device).squeeze(0)
            exp_norm = exp_norm.squeeze(0).cpu() 
            
            pred_mse, _ = model(patch, center, aug=False)
            
            preds_list.append(pred_mse.cpu().squeeze())
            gts_list.append(exp_norm)
            centers_list.append(center.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    gts = torch.cat(gts_list, dim=0).numpy()
    cts = torch.cat(centers_list, dim=0).numpy()
    
    # 🌟 关键修改：按照 Hist2ST/predict.py 中的逻辑构建最终评测结构
    adata_pred = ad.AnnData(preds)
    adata_pred.obsm['spatial'] = cts
    adata_gt = ad.AnnData(gts)
    adata_gt.obsm['spatial'] = cts
    
    # 🌟 关键修改：直接调用与官方等价的评测函数
    # 1. 计算 Mean Gene-wise PCC
    R_array, _ = get_R(adata_pred, adata_gt)
    mean_gene_pcc = np.nanmean(R_array)
    
    print("\n==================================")
    print(f"🎯 Fold {fold} - Mean Gene-wise PCC:  {mean_gene_pcc:.4f}")
    
    # 2. 计算 ARI (已内置滤除 'undetermined' 和 KMeans 逻辑)
    if label is not None and not isinstance(label, torch.Tensor):
        try:
            clus, ari_val = cluster(adata_pred, label)
            print(f"🎯 Fold {fold} - ARI (严格对齐 Hist2ST标准): {ari_val:.4f}")
        except Exception as e:
            print(f"⚠️ 聚类/ARI 计算遭遇异常，跳过计算: {e}")
            
    print("==================================\n")


if __name__ == '__main__':
    # 你可以调整下面的 fold 运行不同的验证任务
    run_experiment(fold=5, epochs=250, gpu_id=0)
    run_experiment(fold=11, epochs=250, gpu_id=0)
    run_experiment(fold=17, epochs=250, gpu_id=0)
    run_experiment(fold=23, epochs=250, gpu_id=0)
    run_experiment(fold=26, epochs=250, gpu_id=0)
    run_experiment(fold=30, epochs=250, gpu_id=0)