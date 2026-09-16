import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

# 导入您的自定义模块
from main import MorphoGeneST, seed_everything
from dataset import ViT_SKIN

def calculate_metrics(preds, gts):
    """
    计算每个基因在所有 Spots 上的 PCC 和 P-value
    preds, gts 形状为: (N_spots, N_genes)
    """
    n_genes = preds.shape[1]
    pccs = []
    p_vals = []
    
    for i in range(n_genes):
        pred_gene = preds[:, i]
        gt_gene = gts[:, i]
        
        # 避免全零或方差为0的情况导致计算报错
        if np.std(pred_gene) == 0 or np.std(gt_gene) == 0:
            pccs.append(np.nan)
            p_vals.append(np.nan)
            continue
            
        r, p = pearsonr(pred_gene, gt_gene)
        pccs.append(r)
        p_vals.append(p)
        
    pccs = np.array(pccs)
    p_vals = np.array(p_vals)
    
    # 清理 nan 值
    valid_mask = ~np.isnan(pccs)
    valid_pccs = pccs[valid_mask]
    valid_pvals = p_vals[valid_mask]
    
    if len(valid_pccs) == 0:
        return 0.0, (0.0, 0.0), 0.0, n_genes
        
    mean_pcc = np.mean(valid_pccs)
    
    # 计算 95% 置信区间 (基于正态分布近似)
    std_pcc = np.std(valid_pccs)
    n = len(valid_pccs)
    margin_of_error = 1.96 * (std_pcc / np.sqrt(n)) if n > 0 else 0
    ci_lower = mean_pcc - margin_of_error
    ci_upper = mean_pcc + margin_of_error
    
    # 显著相关基因比例 (p < 0.05)
    sig_ratio = np.sum(valid_pvals < 0.05) / n_genes * 100
    
    return mean_pcc, (ci_lower, ci_upper), sig_ratio, n_genes

def evaluate_cscc_folds(checkpoint_dir="./model_checkpoints_cscc", gpu_id=0, num_folds=12):
    """
    遍历评估 cSCC 数据集所有 Fold 的模型权重
    """
    seed_everything(12000)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    for fold in range(num_folds):
        print(f"\n==================================================")
        print(f"🌟 正在评估 Fold {fold} 的测试数据和模型...")
        
        # 查找当前 fold 的 checkpoint
        ckpt_pattern = os.path.join(checkpoint_dir, f"Fold{fold}.ckpt")
        ckpt_files = glob.glob(ckpt_pattern)
        
        if not ckpt_files:
            print(f"❌ 未找到 Fold {fold} 的权重文件，跳过...")
            continue
            
        ckpt_path = ckpt_files[0] 
        print(f"📥 成功找到权重: {ckpt_path}")
        
        # 1. 加载测试数据 (务必与打batch前的 dataset 参数一致)
        test_dataset = ViT_SKIN(
            train=False, 
            fold=fold, 
            flatten=False, 
            ori=True, 
            adj=True, 
            r=4
        )
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
        n_spots = len(test_dataset)
        
        # 2. 初始化模型并加载权重
        print(f"🔄 加载模型结构并注入权重...")
        # ⚠️ 这里 n_genes=171 必须与您成功保存时的维度一致
        model = MorphoGeneST.load_from_checkpoint(
            ckpt_path,
            n_genes=171, 
            embed_dim=256, 
            learning_rate=1e-4, 
            bake=1, 
            lamb=0.1, 
            zinb_coef=0.1
        )
        model.eval()
        model.to(device)
        
        # 3. 执行推理
        preds_list, gts_list = [], []
        with torch.no_grad():
            for batch in test_loader:
                # ViT_SKIN 当 ori=True, adj=True 时的返回顺序
                patch, positions, exp_norm, adj, oris, sfs, center = batch
                
                patch = patch.to(device)
                center = center.to(device).squeeze(0)
                
                # 前向传播 (MorphoGeneST 返回 pred_mse 和 pred_zinb 等)
                pred_mse, _ = model(patch, center, aug=False)
                
                # 转为 CPU numpy 格式
                pred_np = pred_mse.cpu().squeeze(0).numpy()
                gt_np = exp_norm.cpu().squeeze(0).numpy()
                
                # 如果真值基因数量超出了模型预测输出的基因数量 (比如 1000 > 171)
                # 截断以对齐评估 (防止报错)
                if gt_np.shape[-1] > pred_np.shape[-1]:
                    gt_np = gt_np[..., :pred_np.shape[-1]]
                
                preds_list.append(pred_np)
                gts_list.append(gt_np)
                

        
        # 4. 计算指标
        preds = np.concatenate(preds_list, axis=0) if preds_list[0].ndim > 1 else np.stack(preds_list, axis=0)
        gts = np.concatenate(gts_list, axis=0) if gts_list[0].ndim > 1 else np.stack(gts_list, axis=0)
        
        # ========================================================
        # 💡 新增/修改：在这里获取真实的 spot 数量
        n_spots = preds.shape[0]
        # ========================================================
        
        # 4. 计算指标
        mean_pcc, ci, sig_ratio, final_n_genes = calculate_metrics(preds, gts)
        
        print(f"📊 基础统计 | Spots: {n_spots} | 评估 Genes: {final_n_genes}")
        print(f"🎯 PCC 结果 | 均值: {mean_pcc:.4f} | 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"🔬 统计检验 | 显著相关基因比例 (p < 0.05): {sig_ratio:.2f}%")
        
        # 记录结果
        results.append({
            "Fold": fold,
            "ID": test_dataset.names[0], # 获取当前切片的名称 (如 'P2_ST_rep1')
            "Spots": n_spots,
            "PCC": mean_pcc,
            "CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "Sig": sig_ratio
        })
        
    # ==========================
    # 汇总输出 Markdown 表格
    # ==========================
    if not results:
        print("未获取到任何结果，请检查模型权重路径！")
        return
        
    df = pd.DataFrame(results)
    
    # 打印 Markdown 表格
    print("\n\n### 📊 详细测试结果数据表 (cSCC)")
    print("| Fold | Sample ID | Spots (斑点数) | PCC 均值 | 95% 置信区间 | 显著相关基因比例 ($p < 0.05$) |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for _, row in df.iterrows():
        print(f"| Fold {int(row['Fold'])} | {row['ID']} | {int(row['Spots'])} | {row['PCC']:.4f} | {row['CI']} | {row['Sig']:.2f}% |")

    # 计算全局统计算法
    total_spots = df["Spots"].sum()
    weighted_pcc = (df["Spots"] * df["PCC"]).sum() / total_spots
    mean_pcc_arithmetic = df["PCC"].mean()
    mean_sig = df["Sig"].mean()
    
    print("\n---\n")
    print("### 🏆 宏观全局评估指标 (Global Metrics)")
    print(f"*   **Fold (切片) 数量**: {len(df)}")
    print(f"*   **评估斑点总数 (Total Spots)**: {total_spots}")
    print(f"*   **加权平均全局 PCC (Weighted Mean PCC)**: $$\\frac{{\\sum (Spots_i \\times PCC_i)}}{{\\sum Spots_i}} \\approx {weighted_pcc:.4f}$$")
    print(f"*   **简单算术平均全局 PCC (Arithmetic Mean PCC)**: $$\\approx {mean_pcc_arithmetic:.4f}$$")
    print(f"*   **平均显著相关基因占比 ($p < 0.05$)**: $$\\approx {mean_sig:.2f}\%$$")

if __name__ == '__main__':
    # 调用函数，默认 gpu 0，12个fold
    evaluate_cscc_folds(checkpoint_dir="./model_checkpoints_cscc", gpu_id=0, num_folds=12)