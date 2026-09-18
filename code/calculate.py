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

# 导入您的模块
from dataset import ViT_HER2ST
from main import MorphoGeneST
from predict import get_R

import warnings
warnings.filterwarnings("ignore")

def cluster_leiden(adata, label):
    """
    使用 Leiden 算法代替原 KMeans 进行聚类并计算 ARI
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
    """
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

def evaluate_single_run(ckpt_dir: str, gpu_id: int = 0):
    """
    运行单次实验目录的 32 折评估，并返回该次实验的平均指标
    """
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    results = {
        "mean_pcc": [],
        "rmse": [],
        "ssim": []
    }

    print(f"\n>>> 正在加载并评测目录: {ckpt_dir} ...")

    # 为了避免输出太乱，这里可以注释掉每一折的详细输出，只看最终结果
    for fold in range(32):
        exact_path = os.path.join(ckpt_dir, f"Fold{fold}.ckpt")
        ckpt_path = None
        
        if os.path.exists(exact_path):
            ckpt_path = exact_path
        else:
            possible_files = glob.glob(os.path.join(ckpt_dir, f"Fold{fold}*.ckpt"))
            if possible_files:
                ckpt_path = possible_files[0] 
                
        if not ckpt_path:
            continue

        test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

        model = MorphoGeneST.load_from_checkpoint(ckpt_path).to(device)
        model.eval()

        preds_list, gts_list, centers_list, positions_list = [], [], [], []

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
                positions_list.append(position.cpu().squeeze())

        preds = torch.cat(preds_list, dim=0).numpy()
        gts = torch.cat(gts_list, dim=0).numpy()
        pos = torch.cat(positions_list, dim=0).numpy()

        adata_pred = ad.AnnData(preds)
        adata_gt = ad.AnnData(gts)

        # 1. PCC (Mean PCC)
        R_array, _ = get_R(adata_pred, adata_gt)
        mean_pcc = np.nanmean(R_array)
        results["mean_pcc"].append(mean_pcc)

        # 2. RMSE
        rmse = np.sqrt(np.mean((preds - gts) ** 2))
        results["rmse"].append(rmse)

        # 3. SSIM 
        ssim_val = compute_ssim_spatial(preds, gts, pos)
        results["ssim"].append(ssim_val)

    if len(results["mean_pcc"]) == 0:
        print(f"⚠️ 警告: 在 {ckpt_dir} 中未找到任何有效的权重文件！")
        return None

    # 计算该次运行的所有折的均值
    avg_metrics = {
        "pcc": np.mean(results["mean_pcc"]),
        "rmse": np.mean(results["rmse"]),
        "ssim": np.mean(results["ssim"])
    }
    
    print(f"✅ 单次运行完成: PCC={avg_metrics['pcc']:.4f}, RMSE={avg_metrics['rmse']:.4f}, SSIM={avg_metrics['ssim']:.4f}")
    return avg_metrics

if __name__ == '__main__':
    # ----------------------------------------------------
    # 🔴 在此填入您通过不同随机种子跑出来的 3 个模型权重文件夹路径
    # ----------------------------------------------------
    CKPT_DIRECTORIES = [
        "/data/MorphoGeneST0/aa",  # 路径1
        "/data/MorphoGeneST0/bb",  # 路径2
        "/data/MorphoGeneST0/cc"   # 路径3
        "/data/MorphoGeneST0/dd" 
        "/data/MorphoGeneST0/ee" 

    ]
    
    GPU_ID = 0
    
    # 用于收集三次实验的数据
    multi_seed_results = {
        "pcc": [],
        "rmse": [],
        "ssim": []
    }

    # 遍历运行三次实验
    for i, ckpt_dir in enumerate(CKPT_DIRECTORIES):
        print(f"\n========== 开始处理第 {i+1}/3 次随机种子实验 ==========")
        res = evaluate_single_run(ckpt_dir, gpu_id=GPU_ID)
        
        if res is not None:
            multi_seed_results["pcc"].append(res["pcc"])
            multi_seed_results["rmse"].append(res["rmse"])
            multi_seed_results["ssim"].append(res["ssim"])

    # 汇总计算最终的均值和标准差 (Mean ± Std)
    if len(multi_seed_results["pcc"]) > 0:
        final_pcc_mean = np.mean(multi_seed_results["pcc"])
        final_pcc_std = np.std(multi_seed_results["pcc"])
        
        final_rmse_mean = np.mean(multi_seed_results["rmse"])
        final_rmse_std = np.std(multi_seed_results["rmse"])
        
        final_ssim_mean = np.mean(multi_seed_results["ssim"])
        final_ssim_std = np.std(multi_seed_results["ssim"])
        
        print("\n" + "="*60)
        print("🏆 多随机种子实验统计结果总结 (Mean ± SD)可直接用于论文填写:")
        print(f"   PCC ↑ : {final_pcc_mean:.3f} ± {final_pcc_std:.3f}")
        print(f"   RMSE ↓: {final_rmse_mean:.3f} ± {final_rmse_std:.3f}")
        print(f"   SSIM ↑: {final_ssim_mean:.3f} ± {final_ssim_std:.3f}")
        print("="*60 + "\n")
        
        # 为了方便调试，也打印出方差 (Variance)
        print(f"(附) PCC Variance : {np.var(multi_seed_results['pcc']):.6f}")
        print(f"(附) RMSE Variance: {np.var(multi_seed_results['rmse']):.6f}")
        print(f"(附) SSIM Variance: {np.var(multi_seed_results['ssim']):.6f}")
    else:
        print("\n❌ 所有的文件夹均未能成功提取到结果，请检查路径。")