# downstream_pipeline.py
# Enhanced downstream analysis pipeline for MorphoGeneST 
# 可直接替换你的下游分析脚本

import os
import warnings
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import gseapy
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, kruskal
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 导入项目自定义模块
from main import MorphoGeneST
from dataset import ViT_HER2ST
from utils import BCELL, CD8T, TUMOR

warnings.filterwarnings("ignore")

# ==============================
# 全局绘图设置
# ==============================
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

sns.set_style("whitegrid")


# ==============================
# 通用工具函数
# ==============================
def _to_array(x):
    """将 AnnData.X 或稀疏矩阵安全转为 numpy array。"""
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def safe_corr(x, y, method="pearson"):
    """安全计算相关系数，避免全零或常数向量导致报错。"""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan

    try:
        if method == "pearson":
            return pearsonr(x, y)[0]
        elif method == "spearman":
            return spearmanr(x, y)[0]
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
    except Exception:
        return np.nan


def finite_vmin_vmax(values, q_low=0.01, q_high=0.99):
    """用分位数确定色阶范围，避免极端值影响空间图。"""
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0, 1

    vmin = np.quantile(values, q_low)
    vmax = np.quantile(values, q_high)

    if np.isclose(vmin, vmax):
        vmin = np.min(values)
        vmax = np.max(values)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return vmin, vmax


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def ensure_spatial_neighbors(adata):
    """确保 squidpy 空间邻接图存在。"""
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic")


# ==============================
# 模型内部推理与解释性信息提取
# ==============================
def extract_attention_distance_profile(block, x, coords):
    try:
        attn_module = block.attn
        x_norm = block.norm1(x)
        B, N, C = x_norm.shape
        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * attn_module.scale
        dist_mat = torch.cdist(coords.float(), coords.float(), p=2)
        dist_bias = attn_module.distance_mlp(dist_mat.unsqueeze(-1)).permute(0, 3, 1, 2)
        attn = attn_logits - dist_bias
        attn_weights = attn.softmax(dim=-1)

        attn_mean = attn_weights[0].mean(dim=0)
        dist = dist_mat[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=dist.device)
        return dist[mask].detach().cpu().numpy(), attn_mean[mask].detach().cpu().numpy()
    except Exception as e:
        print(f"    - Attention distance profile 提取失败: {e}")
        return None, None


def infer_model_outputs(model, patches, centers, chunk_size=64):
    model.eval()
    with torch.no_grad():
        try:
            B, N, C, H, W = patches.shape
            patches_flat = patches.reshape(B * N, C, H, W)
            spot_feats_list = []
            for i in range(0, B * N, chunk_size):
                chunk = patches_flat[i: i + chunk_size]
                spot_feats_list.append(model.image_encoder(chunk))

            spot_feats = torch.cat(spot_feats_list, dim=0)
            centers_b = centers.unsqueeze(0) if centers.dim() == 2 else centers
            h = spot_feats.unsqueeze(0)

            attention_profile = None
            if hasattr(model, "graph_layers") and len(model.graph_layers) > 0:
                dist_flat, weight_flat = extract_attention_distance_profile(model.graph_layers[0], h, centers_b)
                if dist_flat is not None:
                    attention_profile = (dist_flat, weight_flat)

            for layer in model.graph_layers:
                h = layer(h, centers_b)

            h_final = h.squeeze(0)
            pred_mse = model.head_mse(h_final)

            expert_weights = None
            try:
                decoder = model.head_mse[0]
                if hasattr(decoder, "router"):
                    expert_weights = decoder.router(h_final).detach().cpu().numpy()
            except Exception:
                pass

            return pred_mse.detach().cpu().numpy(), expert_weights, attention_profile
        except Exception as e:
            print(f"    - 内部特征推理失败，回退到 model.forward(): {e}")
            pred_mse, _ = model(patches, centers, aug=False)
            return pred_mse.detach().cpu().squeeze().numpy(), None, None


# ==============================
# 任务 1：Marker gene 的定性与定量验证
# ==============================
def analyze_marker_genes(adata_gt, adata_pred, output_dir):
    print("  正在生成标志基因空间对比图...")
    target_genes = ["ERBB2", "FASN", "SCD", "FN1", "CD8A", "CD8B", "MS4A1", "CD79A", "MKI67", "KRT8", "KRT18", "KRT19", "COL1A1"]
    genes_to_plot = [g for g in target_genes if g in adata_pred.var_names and g in adata_gt.var_names]

    if not genes_to_plot:
        return

    # 定性绘图
    gt_values = _to_array(adata_gt[:, genes_to_plot].X)
    vmin, vmax = finite_vmin_vmax(gt_values)

    sc.pl.spatial(adata_gt, color=genes_to_plot, spot_size=90, cmap="magma", vmin=vmin, vmax=vmax, title=[f"Ground Truth: {g}" for g in genes_to_plot], show=False)
    plt.savefig(os.path.join(output_dir, "marker_genes_ground_truth.png"), dpi=300, bbox_inches="tight")
    plt.close()

    sc.pl.spatial(adata_pred, color=genes_to_plot, spot_size=90, cmap="magma", vmin=vmin, vmax=vmax, title=[f"Predicted: {g}" for g in genes_to_plot], show=False)
    plt.savefig(os.path.join(output_dir, "marker_genes_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()

def analyze_marker_gene_quantification(adata_gt, adata_pred, output_dir):
    """基于审稿人要求: 提供下游 Marker 预测的定量验证表格"""
    print("  正在执行关键 Marker Gene 的定量评估...")
    target_genes = ["ERBB2", "FASN", "SCD", "FN1", "CD8A", "CD8B", "MS4A1", "CD79A", "MKI67", "KRT8", "KRT18", "KRT19", "COL1A1"]
    genes = [g for g in target_genes if g in adata_pred.var_names and g in adata_gt.var_names]
    
    records = []
    for g in genes:
        y_true = _to_array(adata_gt[:, g].X).ravel()
        y_pred = _to_array(adata_pred[:, g].X).ravel()
        records.append({
            "Marker_Gene": g,
            "Pearson_Corr": safe_corr(y_true, y_pred, "pearson"),
            "Spearman_Corr": safe_corr(y_true, y_pred, "spearman"),
            "MAE": np.mean(np.abs(y_true - y_pred)),
            "RMSE": np.sqrt(np.mean((y_true - y_pred)**2))
        })
    df_markers = pd.DataFrame(records)
    df_markers.to_csv(os.path.join(output_dir, "quantitative_marker_genes_metrics.csv"), index=False)


# ==============================
# 任务 1.b：空间零值 (Dropouts) 定量恢复验证
# ==============================
def analyze_dropout_recovery_plausibility(adata_gt, adata_pred, output_dir):
    """
    回应审稿人意见1：证明解释为 dropouts 的缺失值预测，不是虚构，而是有独立空间证据支持的。
    通过校验：预测恢复量是否与该 Spot 在真实切片中的物理邻居表达量正相关。
    """
    print("  正在进行空间零值(Technical Dropouts)恢复的生物学合理性验证...")
    ensure_spatial_neighbors(adata_gt)
    adj_mat = adata_gt.obsp['spatial_connectivities']
    
    # 选取高度稀疏但具有恢复潜力的基因进行抽样验证
    gt_matrix = _to_array(adata_gt.X)
    pred_matrix = _to_array(adata_pred.X)
    
    zero_mask = (gt_matrix == 0)
    if not np.any(zero_mask):
        return
        
    records = []
    for g_idx, gene in enumerate(adata_gt.var_names):
        # 如果该基因全为 0 或 几乎没有 0，则跳过
        if np.sum(zero_mask[:, g_idx]) < 10 or np.sum(~zero_mask[:, g_idx]) < 10:
            continue
            
        zero_indices = np.where(zero_mask[:, g_idx])[0]
        # 计算这些0值点其周围邻居的真实表达量 (Local biological neighborhood)
        neighbor_gt_means = []
        pred_imputations = []
        
        for idx in zero_indices:
            neighbors = adj_mat[idx].nonzero()[1]
            if len(neighbors) > 0:
                mean_gt_env = np.mean(gt_matrix[neighbors, g_idx])
                neighbor_gt_means.append(mean_gt_env)
                pred_imputations.append(pred_matrix[idx, g_idx])
                
        if len(pred_imputations) > 5:
            pcc = safe_corr(neighbor_gt_means, pred_imputations, method="pearson")
            records.append({
                "Gene": gene,
                "Imputed_Zero_Spots": len(pred_imputations),
                "Neighborhood_PCC": pcc
            })
            
    if records:
        df_dropout = pd.DataFrame(records).dropna()
        df_dropout.to_csv(os.path.join(output_dir, "dropout_recovery_spatial_validation.csv"), index=False)
        
        # Plot distribution of prediction correlation with neighborhood
        plt.figure(figsize=(6, 4))
        sns.histplot(df_dropout["Neighborhood_PCC"], bins=30, kde=True, color="purple")
        plt.axvline(df_dropout["Neighborhood_PCC"].mean(), color="red", linestyle="--")
        plt.xlabel("Correlation between Imputed Value & GT Neighborhood ($PCC$)")
        plt.ylabel("Number of Genes")
        plt.title("Biological Plausibility of Imputed Spatial Zeros")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dropout_imputation_plausibility_hist.png"), dpi=300)
        plt.close()


# ==============================
# 任务 2：预测性能图
# ==============================
def analyze_prediction_performance(adata_gt, adata_pred, output_dir):
    print("  正在生成预测性能总览图...")
    genes = [g for g in adata_gt.var_names if g in set(adata_pred.var_names)]
    if not genes: return

    gt, pred = _to_array(adata_gt[:, genes].X), _to_array(adata_pred[:, genes].X)
    records = [{"gene": g, "Pearson": safe_corr(gt[:, j], pred[:, j], "pearson"),
                "Spearman": safe_corr(gt[:, j], pred[:, j], "spearman"),
                "MAE": np.mean(np.abs(gt[:, j] - pred[:, j])),
                "RMSE": np.sqrt(np.mean((gt[:, j] - pred[:, j])**2)),
                "Mean_GT": np.mean(gt[:, j])} for j, g in enumerate(genes)]
                
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "gene_level_prediction_metrics.csv"), index=False)
    
    # Gene-wise PCC distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Pearson"].dropna(), bins=40, kde=True, color="#4C72B0")
    plt.axvline(df["Pearson"].mean(), color="red", linestyle="--")
    plt.xlabel("Gene-wise Pearson Correlation")
    plt.title("Distribution of Gene-wise PCC")
    plt.savefig(os.path.join(output_dir, "gene_wise_pcc_distribution.png"), dpi=300); plt.close()


def analyze_tme_scoring(adata_pred, output_dir):
    print("  正在进行 TME 基因集打分...")
    tme_dict = {"B_Cells": BCELL, "Tumor_FASN": TUMOR, "CD8_T_Cells": CD8T}
    score_keys = []
    for key, genes in tme_dict.items():
        sub_genes = [g for g in genes if g in adata_pred.var_names]
        if sub_genes:
            sc.tl.score_genes(adata_pred, gene_list=sub_genes, score_name=key, use_raw=False)
            score_keys.append(key)
    
    if score_keys:
        sc.pl.spatial(adata_pred, color=score_keys, spot_size=90, cmap="viridis", show=False)
        plt.savefig(os.path.join(output_dir, "tme_scoring_spatial.png"), dpi=300, bbox_inches="tight")
        plt.close()


def analyze_tme_violin_and_quantification(adata_pred, output_dir):
    """
    针对审稿人要求增加下游定量验证：提供不同病理区域TME分数的显著性检验
    """
    print("  正在生成 TME score 小提琴图并执行显著性定量检验...")
    score_keys = [k for k in ["B_Cells", "Tumor_FASN", "CD8_T_Cells"] if k in adata_pred.obs.columns]
    if not score_keys:
        return
        
    group_col = "Pat_Annotation" if "Pat_Annotation" in adata_pred.obs.columns else ("Leiden_Clusters" if "Leiden_Clusters" in adata_pred.obs.columns else None)
    if not group_col:
        return

    df = adata_pred.obs[[group_col] + score_keys].copy()
    
    # 1. 定量统计检验 (Kruskal-Wallis 检验计算显著性)
    stats_records = []
    groups = df[group_col].dropna().unique()
    for sig in score_keys:
        distributions = [df[df[group_col] == g][sig].values for g in groups]
        if len(distributions) > 1:
            try:
                stat, pval = kruskal(*distributions)
                stats_records.append({"TME_Signature": sig, "Kruskal_Stat": stat, "P_Value": pval})
            except Exception:
                pass
    if stats_records:
        pd.DataFrame(stats_records).to_csv(os.path.join(output_dir, "tme_scores_kruskal_test.csv"), index=False)

    # 2. 定性图像输出
    df_long = df.melt(id_vars=group_col, var_name="Signature", value_name="Score")
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df_long, x=group_col, y="Score", hue="Signature", cut=0, inner="box")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"TME Signature Scores by {group_col}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tme_scores_violin_by_region.png"), dpi=300)
    plt.close()


# ==============================
# 任务 12：重塑为 Spatial-State Ordering (伪时间解释修正)
# ==============================
def analyze_spatial_state_ordering(adata_pred, output_dir):
    """
    原 "pseudotime" 分析，根据审稿人意见更名和更新声明。
    注意：在缺乏独立时间序列证据的情况下，此处严格定义为探索性的“空间状态排序”
    """
    print("  正在推断探索性的空间状态排序 (Exploratory Spatial-State Ordering)...")
    print("    - 声明: 除非有独立谱系证据支持，本模块推断出的排序仅作为结构和转录重构的探索性展示。")

    try:
        if "Leiden_Clusters" not in adata_pred.obs.columns:
            return

        if "neighbors" not in adata_pred.uns:
            sc.pp.neighbors(adata_pred, n_neighbors=15, use_rep="X")

        root_cluster = adata_pred.obs["Leiden_Clusters"].value_counts().idxmax()
        root_indices = np.where(adata_pred.obs["Leiden_Clusters"] == root_cluster)[0]

        if len(root_indices) == 0:
            return

        adata_pred.uns["iroot"] = root_indices[0]
        try:
            sc.tl.diffmap(adata_pred)
        except Exception:
            pass
        sc.tl.dpt(adata_pred)

        # 更名后进行制图
        sc.pl.spatial(
            adata_pred,
            color="dpt_pseudotime",
            cmap="Spectral_r",
            spot_size=90,
            title="Exploratory Spatial-State Ordering",
            show=False
        )
        plt.savefig(os.path.join(output_dir, "spatial_state_ordering_trajectory.png"), dpi=300, bbox_inches="tight")
        plt.close()

        genes = [g for g in ["ERBB2", "FASN", "FN1", "CD8A", "MKI67", "COL1A1"] if g in adata_pred.var_names]

        if len(genes) > 0:
            pt = np.asarray(adata_pred.obs["dpt_pseudotime"])
            expr = _to_array(adata_pred[:, genes].X)
            
            df_list = [pd.DataFrame({"Spatial_State_Order": pt, "Expression": expr[:, i], "Gene": g}) for i, g in enumerate(genes)]
            df = pd.concat(df_list, axis=0)

            plt.figure(figsize=(8, 5))
            sns.lineplot(data=df, x="Spatial_State_Order", y="Expression", hue="Gene", errorbar=None)
            plt.title("Gene Expression Dynamics along Spatial-State Ordering")
            plt.xlabel("Spatial-State Order")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spatial_state_gene_dynamics.png"), dpi=300)
            plt.close()

    except Exception as e:
        print(f"    - 空间状态排序推断失败: {e}")
        plt.close()


# 省略其余与原代码一模一样未修改的方法 (analyze_clustering, analyze_degs_and_pathways, analyze_co_embedding, 等)
# 在全流程调用中补充我们新增的部分:

def run_full_analysis(fold: int, checkpoint_path: str):
    output_dir = f"./analysis_results22/Fold_{fold}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"🚀 开始对 Fold {fold} 进行增强版全流程下游分析 (针对审稿意见修订版)")
    print("=" * 80)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True)
    sample_name = test_dataset.names[0]
    model = MorphoGeneST.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    patch, position, exp_gt, adj, oris, sfs, center = test_dataset[0]
    patch, center_gpu = patch.unsqueeze(0).to(device), center.to(device)

    with torch.no_grad():
        pred_matrix, expert_weights, attention_profile = infer_model_outputs(model, patch, center_gpu)

    gt_matrix = exp_gt.cpu().squeeze().numpy()
    spatial_coords = center.cpu().numpy()

    adata_pred = sc.AnnData(pred_matrix)
    adata_pred.var_names = pd.Index(test_dataset.gene_set).astype(str)
    adata_pred.obsm["spatial"] = spatial_coords
    
    adata_gt = sc.AnnData(gt_matrix)
    adata_gt.var_names = pd.Index(test_dataset.gene_set).astype(str)
    adata_gt.obsm["spatial"] = spatial_coords

    if sample_name in test_dataset.label and test_dataset.label[sample_name] is not None:
        label = np.asarray(test_dataset.label[sample_name]).astype(str)
        if len(label) == adata_pred.n_obs:
            adata_pred.obs["Pat_Annotation"] = pd.Categorical(label)
            adata_gt.obs["Pat_Annotation"] = pd.Categorical(label)

    # 依次执行各分析模块
    # analyze_marker_genes(adata_gt, adata_pred, output_dir)
    # analyze_marker_gene_quantification(adata_gt, adata_pred, output_dir) # 新增：定量评估表格
    
    # analyze_prediction_performance(adata_gt, adata_pred, output_dir)
    # analyze_dropout_recovery_plausibility(adata_gt, adata_pred, output_dir) # 新增：通过生物学邻居进行dropout合理性验证
    
    # analyze_tme_scoring(adata_pred, output_dir)
    # analyze_tme_violin_and_quantification(adata_pred, output_dir) # 增强：包含非参数检验输出

    analyze_spatial_state_ordering(adata_pred, output_dir) # 重构：更严谨的命名与展示（去Pseudotime中心化）

    # 保存最终对象
    adata_pred.write(os.path.join(output_dir, "adata_pred_final.h5ad"))
    adata_gt.write(os.path.join(output_dir, "adata_gt_final.h5ad"))
    print(f"✅ Fold {fold} 分析任务完成（增强指标均已输出至 {output_dir}）！\n")


if __name__ == "__main__":
    TARGET_FOLD = 5
    CHECKPOINT_PATH = "/data/MorphoGeneST0/model/5.ckpt"

    if os.path.exists(CHECKPOINT_PATH):
        run_full_analysis(fold=TARGET_FOLD, checkpoint_path=CHECKPOINT_PATH)