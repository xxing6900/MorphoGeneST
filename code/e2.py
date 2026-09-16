import os
import glob
import torch
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import structural_similarity

# 导入您的模块 (此处换为 cSCC 的 ViT_SKIN)
from dataset import ViT_SKIN
from main import MorphoGeneST
from predict import get_R

import warnings
warnings.filterwarnings("ignore")

def cluster_leiden(adata, label):
    """
    使用 Leiden 算法聚类并计算 ARI
    """
    idx = label != 'undetermined'
    tmp = adata[idx].copy()
    l = label[idx]
    
    sc.pp.pca(tmp)
    sc.pp.neighbors(tmp, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(tmp, resolution=0.8, key_added="leiden")
    
    p = tmp.obs['leiden'].astype(str).values
    ari_val = adjusted_rand_score(l, p)
    
    return ari_val

def compute_ssim_spatial(preds, gts, positions):
    """
    将一维 Spot 特征还原为 2D 空间表达图，逐基因计算 SSIM
    preds/gts: [N, Gene]
    positions: [N, 2] 代表物理/网格坐标
    """
    # 将坐标平移到从 0 开始以构建密集矩阵
    coords_x = (positions[:, 0] - positions[:, 0].min()).astype(int)
    coords_y = (positions[:, 1] - positions[:, 1].min()).astype(int)
    
    max_x = coords_x.max() + 1
    max_y = coords_y.max() + 1
    
    num_genes = preds.shape[1]
    ssim_scores = []
    
    for g in range(num_genes):
        img_gt = np.zeros((max_x, max_y))
        img_pred = np.zeros((max_x, max_y))
        
        img_gt[coords_x, coords_y] = gts[:, g]
        img_pred[coords_x, coords_y] = preds[:, g]
        
        data_range = img_gt.max() - img_gt.min()
        if data_range == 0:
            data_range = 1e-8
            
        score = structural_similarity(img_gt, img_pred, data_range=data_range)
        ssim_scores.append(score)
        
    return np.nanmean(ssim_scores)

def evaluate_cscc_all_folds(ckpt_dir: str, gpu_id: int = 0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    results = {
        "mean_pcc": [],
        "overall_pcc": [],
        "rmse": [],
        "ssim": [],
        "ari": []  
    }

    # cSCC 数据集固定为 12 折 (P2, P5, P9, P10 * 3个rep)
    print("🚀 开始 cSCC 数据集 12折(样本) 全量评测...")

    for fold in range(12):
        # 1. 查找权重文件 (兼容各种命名)
        exact_path = os.path.join(ckpt_dir, f"Fold{fold}.ckpt")
        ckpt_path = None
        
        if os.path.exists(exact_path):
            ckpt_path = exact_path
        else:
            # 您之前的命名可能是 CSCC-FoldX-... 这里做模糊匹配
            possible_files = glob.glob(os.path.join(ckpt_dir, f"*Fold{fold}*.ckpt"))
            if possible_files:
                ckpt_path = possible_files[0] 
                
        if not ckpt_path:
            print(f"⚠️ 跳过 Fold {fold}: 未找到匹配的权重文件。")
            continue

        print(f"\n[{fold+1}/12] 正在评估 Fold {fold} (权重: {os.path.basename(ckpt_path)})...")
        
        # 2. 准备 cSCC 数据集
        test_dataset = ViT_SKIN(train=False, fold=fold, flatten=False, ori=True, adj=True, r=4)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
        
        # 检查是有标签用于计算 ARI
        label = test_dataset.label[test_dataset.names[0]] if hasattr(test_dataset, 'label') else None

        # 3. 加载模型 (⚠️ 强制 n_genes=171 匹配您之前的权重)
        model = MorphoGeneST.load_from_checkpoint(
            ckpt_path, 
            n_genes=171, 
            embed_dim=256, 
            learning_rate=1e-4, 
            bake=1, 
            lamb=0.1, 
            zinb_coef=0.1
        ).to(device)
        model.eval()

        preds_list, gts_list, centers_list, positions_list = [], [], [], []

        # 4. 运行推理
        with torch.no_grad():
            for batch in test_loader:
                # ViT_SKIN 返回对应的 7 个张量组合
                patch, position, exp_norm, adj, oris, sfs, center = batch
                
                patch = patch.to(device)
                center = center.to(device).squeeze(0)
                exp_norm = exp_norm.squeeze(0).cpu()

                pred_mse, _ = model(patch, center, aug=False)
                
                # 如果真值基因长度和预测基因长度不一致，截取对齐以防止计算报错
                pred_np = pred_mse.cpu().squeeze()
                if exp_norm.shape[-1] > pred_np.shape[-1]:
                    exp_norm = exp_norm[..., :pred_np.shape[-1]]

                preds_list.append(pred_np)
                gts_list.append(exp_norm)
                centers_list.append(center.cpu())
                positions_list.append(position.cpu().squeeze())

        preds = torch.cat(preds_list, dim=0).numpy()
        gts = torch.cat(gts_list, dim=0).numpy()
        cts = torch.cat(centers_list, dim=0).numpy()
        pos = torch.cat(positions_list, dim=0).numpy()

        # 构建 AnnData 用于分析
        adata_pred = ad.AnnData(preds)
        adata_pred.obsm['spatial'] = cts
        adata_gt = ad.AnnData(gts)
        adata_gt.obsm['spatial'] = cts

        # --- 指标计算 ---
        
        # 1. Mean PCC
        R_array, _ = get_R(adata_pred, adata_gt)
        mean_pcc = np.nanmean(R_array)
        results["mean_pcc"].append(mean_pcc)

        # 2. Overall PCC
        valid_mask = np.isfinite(preds.flatten()) & np.isfinite(gts.flatten())
        overall_pcc, _ = pearsonr(preds.flatten()[valid_mask], gts.flatten()[valid_mask])
        results["overall_pcc"].append(overall_pcc)

        # 3. RMSE
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        results["rmse"].append(rmse)

        # 4. SSIM (空间基因表达图像复原度)
        ssim_val = compute_ssim_spatial(preds, gts, pos)
        results["ssim"].append(ssim_val)

        # 5. Leiden ARI (判断是有有效标签)
        metrics_str = f"Mean PCC: {mean_pcc:.4f} | RMSE: {rmse:.4f} | SSIM: {ssim_val:.4f}"
        if label is not None and not isinstance(label, torch.Tensor):
            try:
                ari_val = cluster_leiden(adata_pred, label)
                results["ari"].append(ari_val)
                print(f"   ↳ {metrics_str} | Leiden ARI: {ari_val:.4f}")
            except Exception as e:
                print(f"   ↳ {metrics_str} | 聚类失败: {e}")
        else:
            print(f"   ↳ {metrics_str} | Leiden ARI: 数据未提供空间聚类标签，跳过")

    # ==========================================
    # 汇总输出平均结果
    # ==========================================
    print("\n" + "="*50)
    print("📈 cSCC 数据集各项性能指标平均表现：")
    
    if len(results["mean_pcc"]) > 0:
        print(f"✅ 平均 Mean PCC   : {np.mean(results['mean_pcc']):.4f}")
        print(f"✅ 平均 Overall PCC: {np.mean(results['overall_pcc']):.4f}")
        print(f"✅ 平均 RMSE       : {np.mean(results['rmse']):.4f}")
        print(f"✅ 平均 SSIM       : {np.mean(results['ssim']):.4f}")
    
    if len(results["ari"]) > 0:
        print(f"✅ 平均 Leiden ARI : {np.mean(results['ari']):.4f}  (共计 {len(results['ari'])} 个有效样本)")
    print("="*50 + "\n")

if __name__ == '__main__':
    # ----------------------------------------------------
    # 🔴 在此修改您的 cSCC 模型权重文件夹路径
    # ----------------------------------------------------
    CKPT_DIRECTORY = "./model_checkpoints_cscc" 
    
    evaluate_cscc_all_folds(ckpt_dir=CKPT_DIRECTORY, gpu_id=0)