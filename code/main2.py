import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import anndata as ad

# 导入自定义模块
from main import MorphoGeneST, seed_everything
from dataset import ViT_SKIN
from predict import get_R 

def run_cscc_experiment(fold=0, epochs=200, gpu_id=0):
    """
    针对 cSCC (Skin Cancer) 数据集的训练函数
    fold 范围: 0-11 (对应 P2, P5, P9, P10 的各 3 个 rep)
    """
    print(f"\n🚀 开始训练 cSCC 数据集 | Fold: {fold} (GPU: {gpu_id}) 🚀")
    seed_everything(12000)
    
    # 路径设置
    checkpoint_dir = "./model_checkpoints_cscc"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 1. 加载 cSCC 数据集 (ViT_SKIN)
    # 注意：cSCC 的基因列表通常是 1000 个，由 skin_hvg_cut_1000.npy 定义
    print("[1/4] 正在加载 cSCC 组织病理学与 ST 数据...")
    train_dataset = ViT_SKIN(
        train=True, 
        fold=fold, 
        flatten=False, 
        ori=True, 
        adj=True, 
        r=4 # 对应 patch 大小 224//4 = 56
    )
    test_dataset = ViT_SKIN(
        train=False, 
        fold=fold, 
        flatten=False, 
        ori=True, 
        adj=True, 
        r=4
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

    # 2. 初始化模型
    # n_genes 需设置为 1000 (对应 cSCC 的基因数)
    print(f"[2/4] 初始化网络结构 (n_genes=1000)...")
    model = MorphoGeneST(
        n_genes=171, 
        embed_dim=256, 
        learning_rate=1e-4, 
        bake=1, 
        lamb=0.1, 
        zinb_coef=0.1
    )

    # 3. 设置训练器
    logger = TensorBoardLogger("logs_cscc", name=f"Fold_{fold}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"CSCC-Fold{fold}-{{epoch:02d}}-{{train_loss:.4f}}",
        save_top_k=1, 
        monitor="train_loss", 
        mode="min"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[gpu_id],
        max_epochs=epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        precision="16-mixed"
    )

    # 4. 执行训练
    print(f"[3/4] 启动训练...")
    trainer.fit(model, train_loader)

    # 5. 简单评测 (PCC)
    print("\n[4/4] 训练完成，正在进行推理评测...")
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    
    preds_list, gts_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            # ViT_SKIN 返回顺序: [patches, positions, exps, adj, oris, sfs, centers]
            patch, _, exp_norm, _, _, _, center = batch
            patch = patch.to(device)
            center = center.to(device).squeeze(0)
            
            pred_mse, _ = model(patch, center, aug=False)
            preds_list.append(pred_mse.cpu().squeeze())
            gts_list.append(exp_norm.squeeze(0))

    preds = torch.cat(preds_list, dim=0).numpy()
    gts = torch.cat(gts_list, dim=0).numpy()
    
    adata_pred = ad.AnnData(preds)
    adata_gt = ad.AnnData(gts)
    
    R_array, _ = get_R(adata_pred, adata_gt)
    print("==================================")
    print(f"🎯 cSCC Fold {fold} Mean PCC: {np.nanmean(R_array):.4f}")
    print("==================================\n")

if __name__ == '__main__':
    # cSCC 数据集共有 12 个 fold (0-11)
    # 你可以依次运行，或者选择特定 fold
    run_cscc_experiment(fold=0, epochs=15, gpu_id=0)

