# downstream_pipeline.py
# Enhanced downstream analysis pipeline for MorphoGeneST
# 可直接替换你的 m/2.py

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

from scipy.stats import spearmanr, pearsonr
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
    """
    将 AnnData.X 或稀疏矩阵安全转为 numpy array。
    """
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def safe_corr(x, y, method="pearson"):
    """
    安全计算相关系数，避免全零或常数向量导致报错。
    """
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
    """
    用分位数确定色阶范围，避免极端值影响空间图。
    """
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
    """
    确保 squidpy 空间邻接图存在。
    """
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, coord_type="generic")


# ==============================
# 模型内部推理与解释性信息提取
# ==============================
def extract_attention_distance_profile(block, x, coords):
    """
    提取某个 SAGTBlock 中 attention weight 与空间距离的关系。

    Parameters
    ----------
    block:
        SAGTBlock
    x:
        shape [1, N, C] 的 spot feature
    coords:
        shape [1, N, 2] 的空间坐标

    Returns
    -------
    dist_flat, weight_flat
    """
    try:
        attn_module = block.attn

        x_norm = block.norm1(x)

        B, N, C = x_norm.shape
        qkv = attn_module.qkv(x_norm)
        qkv = qkv.reshape(
            B,
            N,
            3,
            attn_module.num_heads,
            C // attn_module.num_heads
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * attn_module.scale

        dist_mat = torch.cdist(coords.float(), coords.float(), p=2)
        dist_bias = attn_module.distance_mlp(dist_mat.unsqueeze(-1))
        dist_bias = dist_bias.permute(0, 3, 1, 2)

        attn = attn_logits - dist_bias
        attn_weights = attn.softmax(dim=-1)

        # 对多头 attention 求平均
        attn_mean = attn_weights[0].mean(dim=0)  # [N, N]
        dist = dist_mat[0]                       # [N, N]

        mask = ~torch.eye(N, dtype=torch.bool, device=dist.device)

        dist_flat = dist[mask].detach().cpu().numpy()
        weight_flat = attn_mean[mask].detach().cpu().numpy()

        return dist_flat, weight_flat

    except Exception as e:
        print(f"    - Attention distance profile 提取失败: {e}")
        return None, None


def infer_model_outputs(model, patches, centers, chunk_size=64):
    """
    执行模型推理，同时尽量提取：
    1. prediction
    2. MoE expert router weights
    3. attention distance profile

    Parameters
    ----------
    model:
        MorphoGeneST
    patches:
        shape [1, N, 3, H, W]
    centers:
        shape [N, 2] 或 [1, N, 2]

    Returns
    -------
    pred_matrix:
        numpy array, shape [N, G]
    expert_weights:
        numpy array, shape [N, E] 或 None
    attention_profile:
        tuple(dist_flat, weight_flat) 或 None
    """
    model.eval()

    with torch.no_grad():
        try:
            B, N, C, H, W = patches.shape

            patches_flat = patches.reshape(B * N, C, H, W)

            spot_feats_list = []
            for i in range(0, B * N, chunk_size):
                chunk = patches_flat[i: i + chunk_size]
                feat = model.image_encoder(chunk)
                spot_feats_list.append(feat)

            spot_feats = torch.cat(spot_feats_list, dim=0)  # [N, embed_dim]

            if centers.dim() == 2:
                centers_b = centers.unsqueeze(0)
            else:
                centers_b = centers

            h = spot_feats.unsqueeze(0)  # [1, N, embed_dim]

            # 提取第一层 graph attention 的距离-权重关系
            attention_profile = None
            if hasattr(model, "graph_layers") and len(model.graph_layers) > 0:
                dist_flat, weight_flat = extract_attention_distance_profile(
                    model.graph_layers[0],
                    h,
                    centers_b
                )
                if dist_flat is not None and weight_flat is not None:
                    attention_profile = (dist_flat, weight_flat)

            for layer in model.graph_layers:
                h = layer(h, centers_b)

            h_final = h.squeeze(0)  # [N, embed_dim]

            pred_mse = model.head_mse(h_final)

            expert_weights = None
            try:
                # head_mse = Sequential(MoE_Decoder, ReLU)
                decoder = model.head_mse[0]
                if hasattr(decoder, "router"):
                    expert_weights = decoder.router(h_final).detach().cpu().numpy()
            except Exception as e:
                print(f"    - MoE router weights 提取失败: {e}")

            pred_matrix = pred_mse.detach().cpu().numpy()

            return pred_matrix, expert_weights, attention_profile

        except Exception as e:
            print(f"    - 内部特征推理失败，回退到 model.forward(): {e}")

            pred_mse, _ = model(patches, centers, aug=False)
            pred_matrix = pred_mse.detach().cpu().squeeze().numpy()

            return pred_matrix, None, None


# ==============================
# 任务 1：marker gene 空间可视化
# ==============================
def analyze_marker_genes(adata_gt, adata_pred, output_dir):
    """
    核心标���基因空间可视化：Ground Truth vs Predicted。
    """
    print("  [1/13] 正在生成标志基因空间对比图...")

    target_genes = [
        "ERBB2", "FASN", "SCD", "FN1", "CD8A",
        "CD8B", "MS4A1", "CD79A", "MKI67",
        "KRT8", "KRT18", "KRT19", "COL1A1"
    ]

    genes_to_plot = [g for g in target_genes if g in adata_pred.var_names and g in adata_gt.var_names]

    if len(genes_to_plot) == 0:
        print("    - 没有找到可绘制的 marker genes。")
        return

    gt_values = _to_array(adata_gt[:, genes_to_plot].X)
    vmin, vmax = finite_vmin_vmax(gt_values)

    sc.pl.spatial(
        adata_gt,
        color=genes_to_plot,
        spot_size=90,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        title=[f"Ground Truth: {g}" for g in genes_to_plot],
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "marker_genes_ground_truth.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    sc.pl.spatial(
        adata_pred,
        color=genes_to_plot,
        spot_size=90,
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        title=[f"Predicted: {g}" for g in genes_to_plot],
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "marker_genes_predicted.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ==============================
# 任务 2：预测性能图
# ==============================
def analyze_prediction_performance(adata_gt, adata_pred, output_dir):
    """
    生成 gene-wise PCC、spot-wise PCC、MAE 空间图等预测性能图。
    """
    print("  [2/13] 正在生成预测性能总览图...")

    genes = [g for g in adata_gt.var_names if g in set(adata_pred.var_names)]

    if len(genes) == 0:
        print("    - 没有共同基因，跳过预测性能分析。")
        return

    gt = _to_array(adata_gt[:, genes].X)
    pred = _to_array(adata_pred[:, genes].X)

    records = []

    for j, g in enumerate(genes):
        y_true = gt[:, j]
        y_pred = pred[:, j]

        pcc = safe_corr(y_true, y_pred, method="pearson")
        spm = safe_corr(y_true, y_pred, method="spearman")
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        records.append({
            "gene": g,
            "Pearson": pcc,
            "Spearman": spm,
            "MAE": mae,
            "RMSE": rmse,
            "Mean_GT": np.mean(y_true),
            "Var_GT": np.var(y_true)
        })

    df = pd.DataFrame(records)
    df.to_csv(
        os.path.join(output_dir, "gene_level_prediction_metrics.csv"),
        index=False
    )

    # 1. Gene-wise PCC distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Pearson"].dropna(), bins=40, kde=True, color="#4C72B0")
    mean_pcc = df["Pearson"].mean()
    plt.axvline(
        mean_pcc,
        color="red",
        linestyle="--",
        label=f"Mean = {mean_pcc:.3f}"
    )
    plt.xlabel("Gene-wise Pearson Correlation")
    plt.ylabel("Number of Genes")
    plt.title("Distribution of Gene-wise PCC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gene_wise_pcc_distribution.png"),
        dpi=300
    )
    plt.close()

    # 2. Gene-wise Spearman distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Spearman"].dropna(), bins=40, kde=True, color="#55A868")
    mean_spm = df["Spearman"].mean()
    plt.axvline(
        mean_spm,
        color="red",
        linestyle="--",
        label=f"Mean = {mean_spm:.3f}"
    )
    plt.xlabel("Gene-wise Spearman Correlation")
    plt.ylabel("Number of Genes")
    plt.title("Distribution of Gene-wise Spearman Correlation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gene_wise_spearman_distribution.png"),
        dpi=300
    )
    plt.close()

    # 3. Mean expression vs PCC
    plot_df = df.copy()
    plot_df["Mean_GT_clipped"] = plot_df["Mean_GT"].clip(lower=1e-8)

    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=plot_df,
        x="Mean_GT_clipped",
        y="Pearson",
        alpha=0.7,
        s=25,
        linewidth=0
    )
    plt.xscale("log")
    plt.xlabel("Mean Expression in Ground Truth")
    plt.ylabel("Gene-wise PCC")
    plt.title("Expression Level vs Prediction Accuracy")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "mean_expression_vs_pcc.png"),
        dpi=300
    )
    plt.close()

    # 4. Variance vs PCC
    plot_df["Var_GT_clipped"] = plot_df["Var_GT"].clip(lower=1e-8)

    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=plot_df,
        x="Var_GT_clipped",
        y="Pearson",
        alpha=0.7,
        s=25,
        linewidth=0,
        color="#8172B3"
    )
    plt.xscale("log")
    plt.xlabel("Expression Variance in Ground Truth")
    plt.ylabel("Gene-wise PCC")
    plt.title("Expression Variability vs Prediction Accuracy")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "variance_vs_pcc.png"),
        dpi=300
    )
    plt.close()

    # 5. Top / Bottom predicted genes
    valid_df = df.dropna(subset=["Pearson"]).copy()

    if len(valid_df) > 0:
        top_df = valid_df.sort_values("Pearson", ascending=False).head(20)
        bottom_df = valid_df.sort_values("Pearson", ascending=True).head(20)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(
            data=top_df,
            x="Pearson",
            y="gene",
            ax=axes[0],
            color="#55A868"
        )
        axes[0].set_title("Top 20 Predicted Genes")
        axes[0].set_xlabel("PCC")
        axes[0].set_ylabel("Gene")

        sns.barplot(
            data=bottom_df,
            x="Pearson",
            y="gene",
            ax=axes[1],
            color="#C44E52"
        )
        axes[1].set_title("Bottom 20 Predicted Genes")
        axes[1].set_xlabel("PCC")
        axes[1].set_ylabel("Gene")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "top_bottom_gene_pcc.png"),
            dpi=300
        )
        plt.close()

    # 6. Overall Predicted vs Ground Truth scatter
    flat_gt = gt.ravel()
    flat_pred = pred.ravel()

    finite_mask = np.isfinite(flat_gt) & np.isfinite(flat_pred)
    flat_gt = flat_gt[finite_mask]
    flat_pred = flat_pred[finite_mask]

    if len(flat_gt) > 0:
        rng = np.random.default_rng(0)
        n_sample = min(50000, len(flat_gt))
        idx = rng.choice(len(flat_gt), size=n_sample, replace=False)

        overall_pcc = safe_corr(flat_gt[idx], flat_pred[idx], method="pearson")

        plt.figure(figsize=(5, 5))
        sns.scatterplot(
            x=flat_gt[idx],
            y=flat_pred[idx],
            s=5,
            alpha=0.2,
            linewidth=0
        )
        plt.xlabel("Ground Truth Expression")
        plt.ylabel("Predicted Expression")
        plt.title(f"Predicted vs Ground Truth\nPCC = {overall_pcc:.3f}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "pred_vs_gt_scatter.png"),
            dpi=300
        )
        plt.close()

    # 7. Spot-wise PCC
    spot_pcc = []

    for i in range(gt.shape[0]):
        spot_pcc.append(safe_corr(gt[i, :], pred[i, :], method="pearson"))

    adata_pred.obs["Spot_PCC"] = spot_pcc

    sc.pl.spatial(
        adata_pred,
        color="Spot_PCC",
        cmap="viridis",
        spot_size=90,
        title="Spot-wise Prediction PCC",
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "spot_wise_pcc_spatial.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 8. Spot-wise MAE
    adata_pred.obs["Spot_MAE"] = np.mean(np.abs(gt - pred), axis=1)

    sc.pl.spatial(
        adata_pred,
        color="Spot_MAE",
        cmap="Reds",
        spot_size=90,
        title="Spot-wise Prediction MAE",
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "spot_wise_mae_spatial.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 9. Spot-wise RMSE
    adata_pred.obs["Spot_RMSE"] = np.sqrt(np.mean((gt - pred) ** 2, axis=1))

    sc.pl.spatial(
        adata_pred,
        color="Spot_RMSE",
        cmap="Reds",
        spot_size=90,
        title="Spot-wise Prediction RMSE",
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "spot_wise_rmse_spatial.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ==============================
# 任务 3：GT / Pred / Error 三联图
# ==============================
def analyze_gene_triplet_maps(adata_gt, adata_pred, output_dir):
    """
    对核心 marker genes 绘制 Ground Truth / Prediction / Absolute Error 三联空间图。
    """
    print("  [3/13] 正在生成 GT-Pred-Error 三联图...")

    target_genes = [
        "ERBB2", "FASN", "SCD", "FN1", "CD8A",
        "CD8B", "MS4A1", "CD79A", "MKI67",
        "KRT8", "KRT18", "KRT19", "COL1A1"
    ]

    genes_to_plot = [g for g in target_genes if g in adata_pred.var_names and g in adata_gt.var_names]

    if len(genes_to_plot) == 0:
        print("    - 没有找到可绘制的目标基因。")
        return

    triplet_dir = os.path.join(output_dir, "gene_triplet_maps")
    os.makedirs(triplet_dir, exist_ok=True)

    for g in genes_to_plot:
        gt_vec = _to_array(adata_gt[:, g].X).ravel()
        pred_vec = _to_array(adata_pred[:, g].X).ravel()
        err_vec = np.abs(gt_vec - pred_vec)

        adata_gt.obs[f"{g}_GT"] = gt_vec
        adata_pred.obs[f"{g}_Pred"] = pred_vec
        adata_pred.obs[f"{g}_AbsError"] = err_vec

        expr_values = np.concatenate([gt_vec, pred_vec])
        vmin, vmax = finite_vmin_vmax(expr_values)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        sc.pl.spatial(
            adata_gt,
            color=f"{g}_GT",
            cmap="magma",
            spot_size=90,
            vmin=vmin,
            vmax=vmax,
            title=f"{g} Ground Truth",
            ax=axes[0],
            show=False
        )

        sc.pl.spatial(
            adata_pred,
            color=f"{g}_Pred",
            cmap="magma",
            spot_size=90,
            vmin=vmin,
            vmax=vmax,
            title=f"{g} Predicted",
            ax=axes[1],
            show=False
        )

        sc.pl.spatial(
            adata_pred,
            color=f"{g}_AbsError",
            cmap="Reds",
            spot_size=90,
            title=f"{g} Absolute Error",
            ax=axes[2],
            show=False
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(triplet_dir, f"triplet_GT_Pred_Error_{g}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


# ==============================
# 任务 4：空间域识别
# ==============================
def analyze_clustering(adata_pred, output_dir):
    """
    Leiden 聚类与病理学标注对比。
    """
    print("  [4/13] 正在进行 Leiden 聚类与空间域识别...")

    sc.pp.neighbors(adata_pred, n_neighbors=15, use_rep="X")
    sc.tl.leiden(
        adata_pred,
        resolution=0.8,
        key_added="Leiden_Clusters"
    )

    sc.pl.spatial(
        adata_pred,
        color="Leiden_Clusters",
        spot_size=90,
        palette="tab20",
        title="Leiden Clustering on Predicted Data",
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "leiden_clusters.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    if "Pat_Annotation" in adata_pred.obs.columns:
        sc.pl.spatial(
            adata_pred,
            color=["Pat_Annotation", "Leiden_Clusters"],
            spot_size=90,
            palette="tab20",
            show=False,
            wspace=0.3
        )
        plt.savefig(
            os.path.join(output_dir, "cluster_vs_ground_truth.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    return adata_pred


def analyze_cluster_confusion_matrix(adata_pred, output_dir):
    """
    绘制病理标注与 Leiden cluster 的混淆矩阵，并计算 ARI / NMI。
    """
    print("  [5/13] 正在生成病理标注与聚类混淆矩阵...")

    if "Pat_Annotation" not in adata_pred.obs.columns:
        print("    - 没有 Pat_Annotation，跳过混淆矩阵。")
        return

    if "Leiden_Clusters" not in adata_pred.obs.columns:
        print("    - 没有 Leiden_Clusters，跳过混淆矩阵。")
        return

    obs_df = adata_pred.obs.copy()

    # 过滤 undetermined
    mask = obs_df["Pat_Annotation"].astype(str) != "undetermined"
    obs_eval = obs_df.loc[mask].copy()

    if len(obs_eval) == 0:
        print("    - 过滤 undetermined 后无可用 spot。")
        return

    tab_count = pd.crosstab(
        obs_eval["Pat_Annotation"],
        obs_eval["Leiden_Clusters"]
    )

    tab_norm = pd.crosstab(
        obs_eval["Pat_Annotation"],
        obs_eval["Leiden_Clusters"],
        normalize="index"
    )

    tab_count.to_csv(os.path.join(output_dir, "pathology_vs_leiden_count.csv"))
    tab_norm.to_csv(os.path.join(output_dir, "pathology_vs_leiden_normalized.csv"))

    ari = adjusted_rand_score(
        obs_eval["Pat_Annotation"].astype(str),
        obs_eval["Leiden_Clusters"].astype(str)
    )

    nmi = normalized_mutual_info_score(
        obs_eval["Pat_Annotation"].astype(str),
        obs_eval["Leiden_Clusters"].astype(str)
    )

    metrics_df = pd.DataFrame({
        "Metric": ["ARI", "NMI"],
        "Value": [ari, nmi]
    })
    metrics_df.to_csv(
        os.path.join(output_dir, "cluster_annotation_metrics.csv"),
        index=False
    )

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        tab_norm,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    plt.xlabel("Leiden Clusters")
    plt.ylabel("Pathologist Annotation")
    plt.title(f"Pathology Annotation vs Leiden Clusters\nARI={ari:.3f}, NMI={nmi:.3f}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "pathology_vs_leiden_confusion_matrix.png"),
        dpi=300
    )
    plt.close()


# ==============================
# 任务 6：DEGs 与通路富集
# ==============================
def analyze_degs_and_pathways(adata_pred, output_dir):
    """
    差异表达基因与 GO/KEGG 通路富集。
    """
    print("  [6/13] 正在分析差异基因与 GO/KEGG 通路...")

    if "Leiden_Clusters" not in adata_pred.obs.columns:
        print("    - 没有 Leiden_Clusters，跳过 DEG 分析。")
        return

    deg_dir = os.path.join(output_dir, "deg_pathway")
    os.makedirs(deg_dir, exist_ok=True)

    try:
        # 计算差异基因
        sc.tl.rank_genes_groups(
            adata_pred,
            "Leiden_Clusters",
            method="t-test"
        )

        # ====================================================
        # ⭐ 核心修改 1：DEG heatmap (热图)
        # ====================================================
        sc.pl.rank_genes_groups_heatmap(
            adata_pred,
            n_genes=5,
            groupby="Leiden_Clusters",
            standard_scale='var',  # ⭐ 按基因进行 0-1 标准化，消除绝对表达量的影响
            show=False,
            cmap="viridis"         # ⭐ 去掉了 vmin/vmax，改用适合 0-1 的翠绿色带 (或者也可换成 'magma')
        )
        plt.savefig(
            os.path.join(deg_dir, "deg_heatmap.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ====================================================
        # ⭐ 核心修改 2：DEG dotplot (气泡图)
        # ====================================================
        sc.pl.rank_genes_groups_dotplot(
            adata_pred,
            n_genes=5,
            groupby="Leiden_Clusters",
            standard_scale='var',  # ⭐ 灵魂参数：强制拉开不同 Cluster 间的颜色差异
            cmap="Reds",           # ⭐ 使用经典的红色渐变色
            show=False
        )
        plt.savefig(
            os.path.join(deg_dir, "deg_dotplot.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # ---------- 下方的提取和富集分析代码保持不变 ----------

        # 提取 rank_genes_groups 结果
        result = adata_pred.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        deg_tables = []
        for group in groups:
            tmp = pd.DataFrame({
                "group": group,
                "gene": result["names"][group],
                "score": result["scores"][group],
                "pval": result["pvals"][group],
                "pval_adj": result["pvals_adj"][group],
                "logfoldchange": result["logfoldchanges"][group]
                if "logfoldchanges" in result else np.nan
            })
            deg_tables.append(tmp)

        deg_df = pd.concat(deg_tables, axis=0)
        deg_df.to_csv(
            os.path.join(deg_dir, "rank_genes_groups_all.csv"),
            index=False
        )

        # 每个 cluster 取 top genes 合并做通路
        top_genes = (
            deg_df.sort_values(["group", "pval_adj"])
            .groupby("group")
            .head(50)["gene"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if len(top_genes) == 0:
            print("    - 无 top genes，跳过富集分析。")
            return

        try:
            enr = gseapy.enrichr(
                gene_list=top_genes,
                gene_sets=[
                    "KEGG_2021_Human",
                    "GO_Biological_Process_2021"
                ],
                organism="human",
                outdir=os.path.join(deg_dir, "enrichr_results"),
                cutoff=0.05
            )

            if enr.results is not None and not enr.results.empty:
                enr.results.to_csv(
                    os.path.join(deg_dir, "enrichment_results.csv"),
                    index=False
                )

                # Barplot
                try:
                    gseapy.barplot(
                        enr.results,
                        top_term=10,
                        ofname=os.path.join(deg_dir, "pathway_enrichment_barplot.png")
                    )
                except Exception as e:
                    print(f"    - gseapy barplot 失败: {e}")

                # Dotplot
                try:
                    gseapy.dotplot(
                        enr.results,
                        column="Adjusted P-value",
                        x="Gene_set",
                        top_term=10,
                        figsize=(8, 6),
                        ofname=os.path.join(deg_dir, "pathway_enrichment_dotplot.png")
                    )
                except Exception as e:
                    print(f"    - gseapy dotplot 失败: {e}")

        except Exception as e:
            print(f"    - 通路富集分析失败: {e}")

    except Exception as e:
        print(f"    - DEG 分析失败: {e}")

    plt.close()


# ==============================
# 任务 7：TME scoring
# ==============================
def analyze_tme_scoring(adata_pred, output_dir):
    """
    肿瘤微环境基因集打分与空间图。
    """
    print("  [7/13] 正在进行 TME 基因集打分...")

    tme_dict = {
        "B_Cells": BCELL,
        "Tumor_FASN": TUMOR,
        "CD8_T_Cells": CD8T
    }

    score_keys = []

    for key, genes in tme_dict.items():
        sub_genes = [g for g in genes if g in adata_pred.var_names]

        if len(sub_genes) > 0:
            try:
                sc.tl.score_genes(
                    adata_pred,
                    gene_list=sub_genes,
                    score_name=key,
                    use_raw=False
                )
                score_keys.append(key)
            except Exception as e:
                print(f"    - {key} 打分失败: {e}")

    if len(score_keys) > 0:
        sc.pl.spatial(
            adata_pred,
            color=score_keys,
            spot_size=90,
            cmap="viridis",
            show=False
        )
        plt.savefig(
            os.path.join(output_dir, "tme_scoring_spatial.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


def analyze_tme_violin(adata_pred, output_dir):
    """
    绘制 TME score 在不同病理区域或 Leiden cluster 中的分布。
    """
    print("  [8/13] 正在生成 TME score violin plot...")

    score_keys = [
        k for k in ["B_Cells", "Tumor_FASN", "CD8_T_Cells"]
        if k in adata_pred.obs.columns
    ]

    if len(score_keys) == 0:
        print("    - 没有 TME score，跳过 violin plot。")
        return

    if "Pat_Annotation" in adata_pred.obs.columns:
        group_col = "Pat_Annotation"
    elif "Leiden_Clusters" in adata_pred.obs.columns:
        group_col = "Leiden_Clusters"
    else:
        print("    - 没有可用分组，跳过。")
        return

    df = adata_pred.obs[[group_col] + score_keys].copy()
    df_long = df.melt(
        id_vars=group_col,
        var_name="Signature",
        value_name="Score"
    )

    plt.figure(figsize=(10, 5))
    sns.violinplot(
        data=df_long,
        x=group_col,
        y="Score",
        hue="Signature",
        cut=0,
        inner="box"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"TME Signature Scores by {group_col}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "tme_scores_violin_by_region.png"),
        dpi=300
    )
    plt.close()

    # Boxplot 版本
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_long,
        x=group_col,
        y="Score",
        hue="Signature"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title(f"TME Signature Scores by {group_col}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "tme_scores_boxplot_by_region.png"),
        dpi=300
    )
    plt.close()


# ==============================
# 任务 9：UMAP Co-embedding
# ==============================
def analyze_co_embedding(adata_gt, adata_pred, output_dir):
    """
    真实与预测表达矩阵 co-embedding。
    """
    print("  [9/13] 正在生成 UMAP 潜空间对齐图...")

    try:
        adata_gt_tmp = adata_gt.copy()
        adata_pred_tmp = adata_pred.copy()

        adata_gt_tmp.obs["Source"] = "Ground Truth"
        adata_pred_tmp.obs["Source"] = "Predicted"

        adata_concat = adata_gt_tmp.concatenate(
            adata_pred_tmp,
            batch_key="Batch",
            batch_categories=["Ground Truth", "Predicted"]
        )

        adata_concat.obs["Source"] = adata_concat.obs["Batch"].astype(str)

        sc.pp.pca(adata_concat)
        sc.pp.neighbors(adata_concat, n_neighbors=15, use_rep="X_pca")
        sc.tl.umap(adata_concat)

        color_list = ["Source"]

        if "Leiden_Clusters" in adata_concat.obs.columns:
            try:
                adata_concat.obs["Leiden_Clusters"] = (
                    adata_concat.obs["Leiden_Clusters"]
                    .astype("category")
                    .cat.add_categories(["N/A"])
                    .fillna("N/A")
                )
            except Exception:
                adata_concat.obs["Leiden_Clusters"] = (
                    adata_concat.obs["Leiden_Clusters"]
                    .astype(str)
                    .fillna("N/A")
                )

            color_list.append("Leiden_Clusters")

        sc.pl.umap(
            adata_concat,
            color=color_list,
            wspace=0.3,
            show=False
        )
        plt.savefig(
            os.path.join(output_dir, "umap_co_embedding.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print(f"    - UMAP co-embedding 失败: {e}")
        plt.close()


# ==============================
# 任务 10：Moran's I
# ==============================
def analyze_spatial_autocorrelation(adata_gt, adata_pred, output_dir):
    """
    Moran's I 空间自相关性对比。
    """
    print("  [10/13] 正在对比真实与预测的 Moran's I...")

    try:
        ensure_spatial_neighbors(adata_gt)
        ensure_spatial_neighbors(adata_pred)

        common_genes = [g for g in adata_gt.var_names if g in set(adata_pred.var_names)]

        sq.gr.spatial_autocorr(
            adata_gt,
            mode="moran",
            genes=common_genes,
            n_perms=100,
            n_jobs=4
        )

        sq.gr.spatial_autocorr(
            adata_pred,
            mode="moran",
            genes=common_genes,
            n_perms=100,
            n_jobs=4
        )

        moran_gt = adata_gt.uns["moranI"][["I"]].rename(columns={"I": "Moran_I_GT"})
        moran_pred = adata_pred.uns["moranI"][["I"]].rename(columns={"I": "Moran_I_Pred"})

        moran_df = moran_gt.join(moran_pred, how="inner").dropna()
        moran_df["Delta"] = moran_df["Moran_I_Pred"] - moran_df["Moran_I_GT"]

        moran_df.to_csv(
            os.path.join(output_dir, "morans_i_gene_level.csv")
        )

        r, p = spearmanr(
            moran_df["Moran_I_GT"],
            moran_df["Moran_I_Pred"]
        )

        plt.figure(figsize=(6, 6))
        sns.regplot(
            data=moran_df,
            x="Moran_I_GT",
            y="Moran_I_Pred",
            scatter_kws={"alpha": 0.5, "s": 15},
            line_kws={"color": "red"}
        )
        plt.title(f"Moran's I Correlation\nSpearman R={r:.3f}, p={p:.2e}")
        plt.xlabel("Moran's I Ground Truth")
        plt.ylabel("Moran's I Predicted")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "morans_i_correlation.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # Moran's I delta distribution
        plt.figure(figsize=(6, 4))
        sns.histplot(
            moran_df["Delta"],
            bins=40,
            kde=True,
            color="#C44E52"
        )
        plt.axvline(0, color="black", linestyle="--")
        plt.xlabel("Predicted Moran's I - Ground Truth Moran's I")
        plt.ylabel("Number of Genes")
        plt.title("Distribution of Moran's I Difference")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "morans_i_delta_distribution.png"),
            dpi=300
        )
        plt.close()

    except Exception as e:
        print(f"    - Moran's I 分析失败: {e}")
        plt.close()


# ==============================
# 任务 11：空间邻域富集
# ==============================
def analyze_neighborhood(adata_pred, output_dir):
    """
    空间邻域富集分析。
    """
    print("  [11/13] 正在分析空间邻域富集关系...")

    try:
        if "Leiden_Clusters" not in adata_pred.obs.columns:
            print("    - 没有 Leiden_Clusters，跳过邻域分析。")
            return

        if len(adata_pred.obs["Leiden_Clusters"].unique()) <= 1:
            print("    - 聚类数不足，跳过邻域分析。")
            return

        ensure_spatial_neighbors(adata_pred)

        sq.gr.nhood_enrichment(
            adata_pred,
            cluster_key="Leiden_Clusters"
        )

        sq.pl.nhood_enrichment(
            adata_pred,
            cluster_key="Leiden_Clusters",
            figsize=(6, 6),
            show=False
        )

        plt.savefig(
            os.path.join(output_dir, "neighborhood_enrichment.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    except Exception as e:
        print(f"    - 邻域富集分析失败: {e}")
        plt.close()


# ==============================
# 任务 12：伪时间轨迹
# ==============================
def analyze_pseudotime(adata_pred, output_dir):
    """
    空间伪时间轨迹推断。
    """
    print("  [12/13] 正在推断空间伪时间轨迹...")

    try:
        if "Leiden_Clusters" not in adata_pred.obs.columns:
            print("    - 没有 Leiden_Clusters，跳过伪时间分析。")
            return

        if "neighbors" not in adata_pred.uns:
            sc.pp.neighbors(adata_pred, n_neighbors=15, use_rep="X")

        root_cluster = adata_pred.obs["Leiden_Clusters"].value_counts().idxmax()
        root_indices = np.where(adata_pred.obs["Leiden_Clusters"] == root_cluster)[0]

        if len(root_indices) == 0:
            print("    - 找不到根节点，跳过伪时间分析。")
            return

        adata_pred.uns["iroot"] = root_indices[0]

        try:
            sc.tl.diffmap(adata_pred)
        except Exception:
            pass

        sc.tl.dpt(adata_pred)

        sc.pl.spatial(
            adata_pred,
            color="dpt_pseudotime",
            cmap="Spectral_r",
            spot_size=90,
            title="Spatial Pseudotime Trajectory",
            show=False
        )
        plt.savefig(
            os.path.join(output_dir, "Exploratory Spatial-State Ordering.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # 如果有关键基因，画 pseudotime vs expression
        genes = [
            g for g in ["ERBB2", "FASN", "FN1", "CD8A", "MKI67", "COL1A1"]
            if g in adata_pred.var_names
        ]

        if len(genes) > 0:
            pt = np.asarray(adata_pred.obs["dpt_pseudotime"])
            expr = _to_array(adata_pred[:, genes].X)

            df_list = []
            for i, g in enumerate(genes):
                df_list.append(pd.DataFrame({
                    "Pseudotime": pt,
                    "Expression": expr[:, i],
                    "Gene": g
                }))

            df = pd.concat(df_list, axis=0)

            plt.figure(figsize=(8, 5))
            sns.lineplot(
                data=df,
                x="Pseudotime",
                y="Expression",
                hue="Gene",
                errorbar=None
            )
            plt.title("Gene Expression Dynamics along Pseudotime")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "spatial_state_gene_dynamics.png"),
                dpi=300
            )
            plt.close()

    except Exception as e:
        print(f"    - 伪时间分析失败: {e}")
        plt.close()


# ==============================
# 任务 13：模型解释性图
# ==============================
def analyze_moe_expert_usage(adata_pred, expert_weights, output_dir):
    """
    绘制 MoE expert usage 空间图。
    """
    print("  [13/13] 正在生成 MoE expert usage 图...")

    if expert_weights is None:
        print("    - 未提取到 MoE expert weights，跳过。")
        return

    if len(expert_weights) != adata_pred.n_obs:
        print("    - expert weights 数量与 spots 不一致，跳过。")
        return

    n_experts = expert_weights.shape[1]
    expert_cols = []

    for i in range(n_experts):
        col = f"Expert_{i + 1}_Weight"
        adata_pred.obs[col] = expert_weights[:, i]
        expert_cols.append(col)

    dominant = np.argmax(expert_weights, axis=1).astype(str)
    adata_pred.obs["Dominant_Expert"] = pd.Categorical(
        [f"Expert_{int(i) + 1}" for i in dominant]
    )

    # Expert continuous weights spatial map
    sc.pl.spatial(
        adata_pred,
        color=expert_cols,
        cmap="viridis",
        spot_size=90,
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "moe_expert_weights_spatial.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Dominant expert map
    sc.pl.spatial(
        adata_pred,
        color="Dominant_Expert",
        palette="tab10",
        spot_size=90,
        title="Dominant MoE Expert",
        show=False
    )
    plt.savefig(
        os.path.join(output_dir, "moe_dominant_expert_spatial.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Expert usage proportion
    usage = adata_pred.obs["Dominant_Expert"].value_counts(normalize=True).reset_index()
    usage.columns = ["Expert", "Proportion"]

    usage.to_csv(
        os.path.join(output_dir, "moe_dominant_expert_proportion.csv"),
        index=False
    )

    plt.figure(figsize=(5, 4))
    sns.barplot(
        data=usage,
        x="Expert",
        y="Proportion",
        palette="tab10"
    )
    plt.ylabel("Proportion of Spots")
    plt.title("Dominant MoE Expert Usage")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "moe_dominant_expert_proportion.png"),
        dpi=300
    )
    plt.close()

    # Expert weights by pathology / cluster
    if "Pat_Annotation" in adata_pred.obs.columns:
        group_col = "Pat_Annotation"
    elif "Leiden_Clusters" in adata_pred.obs.columns:
        group_col = "Leiden_Clusters"
    else:
        group_col = None

    if group_col is not None:
        df = adata_pred.obs[[group_col] + expert_cols].copy()
        df_long = df.melt(
            id_vars=group_col,
            var_name="Expert",
            value_name="Weight"
        )

        plt.figure(figsize=(10, 5))
        sns.boxplot(
            data=df_long,
            x=group_col,
            y="Weight",
            hue="Expert"
        )
        plt.xticks(rotation=45, ha="right")
        plt.title(f"MoE Expert Weights by {group_col}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "moe_expert_weights_by_region.png"),
            dpi=300
        )
        plt.close()


def analyze_attention_distance(attention_profile, output_dir):
    """
    绘制 attention weight 与空间距离之间的关系。
    """
    print("  [Extra] 正在生成 Attention weight vs spatial distance 图...")

    if attention_profile is None:
        print("    - 没有 attention profile，跳过。")
        return

    dist_flat, weight_flat = attention_profile

    if dist_flat is None or weight_flat is None:
        print("    - attention profile 为空，跳过。")
        return

    if len(dist_flat) == 0:
        print("    - attention profile 长度为 0，跳过。")
        return

    df = pd.DataFrame({
        "Distance": dist_flat,
        "Attention_Weight": weight_flat
    })

    # 为避免点太多，抽样
    if len(df) > 100000:
        df_plot = df.sample(100000, random_state=0)
    else:
        df_plot = df

    # binned mean
    df["Distance_Bin"] = pd.qcut(
        df["Distance"],
        q=30,
        duplicates="drop"
    )

    bin_df = df.groupby("Distance_Bin").agg(
        Distance=("Distance", "mean"),
        Attention_Weight=("Attention_Weight", "mean")
    ).reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=df_plot,
        x="Distance",
        y="Attention_Weight",
        s=4,
        alpha=0.15,
        linewidth=0
    )
    sns.lineplot(
        data=bin_df,
        x="Distance",
        y="Attention_Weight",
        color="red",
        linewidth=2,
        label="Binned mean"
    )
    plt.xlabel("Spatial Distance Between Spots")
    plt.ylabel("Attention Weight")
    plt.title("Spatial Distance vs Attention Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "attention_weight_vs_spatial_distance.png"),
        dpi=300
    )
    plt.close()

    df.to_csv(
        os.path.join(output_dir, "attention_distance_profile.csv"),
        index=False
    )

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
# ==============================
# 主流程
# ==============================
def analyze_expert_biological_specialization(adata_pred, output_dir):
    """
    专门针对审稿人意见：解析 MoE 专家专门化（expert specialization）的生物学意义。
    对不同专家主导的区域进行差异基因分析，寻找每个专家的生物学印记。
    """
    print("  正在深入解析 MoE 专家专门化 (Expert Specialization) 的生物学意义...")
    
    if "Dominant_Expert" not in adata_pred.obs.columns:
        print("    - 未找到 Dominant_Expert，无法进行专家专门化分析。")
        return

    # 过滤掉包含样本太少的专家类别，避免统计报错
    expert_counts = adata_pred.obs["Dominant_Expert"].value_counts()
    valid_experts = expert_counts[expert_counts > 10].index.tolist()
    
    if len(valid_experts) < 2:
        print("    - 有效的专家类别不足两个，跳过该分析。")
        return

    # 仅保留被有效专家主导的 Spots
    adata_expert = adata_pred[adata_pred.obs["Dominant_Expert"].isin(valid_experts)].copy()
    
    expert_dir = os.path.join(output_dir, "expert_specialization")
    os.makedirs(expert_dir, exist_ok=True)

    try:
        # 1. 以“主导专家”为分组，进行差异基因表达 (DEG) 分析
        sc.tl.rank_genes_groups(adata_expert, groupby="Dominant_Expert", method="t-test")
        
        # 2. 绘制每个专家的 Top 基因热图 (直观展示专家的生物学指纹)
        # 注意：这里去掉了 title 参数，避免 scanpy 传参给报错
        sc.pl.rank_genes_groups_heatmap(
            adata_expert, 
            n_genes=8, 
            groupby="Dominant_Expert", 
            show=False, 
            cmap="bwr",
            vmin=-2, 
            vmax=2
        )
        plt.savefig(os.path.join(expert_dir, "expert_signature_heatmap.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # 3. 提取定量的 DEG 表格供回复审查
        result = adata_expert.uns["rank_genes_groups"]
        groups = result["names"].dtype.names
        deg_tables = []
        
        for group in groups:
            tmp = pd.DataFrame({
                "Expert": group,
                "Marker_Gene": result["names"][group],
                "LogFoldChange": result["logfoldchanges"][group] if "logfoldchanges" in result else np.nan,
                "P_value": result["pvals_adj"][group]
            })
            deg_tables.append(tmp)
            
        deg_df = pd.concat(deg_tables, axis=0)
        # 保存一份每个专家的特征基因列表
        top_expert_genes = deg_df[deg_df["P_value"] < 0.05].groupby("Expert").head(15)
        top_expert_genes.to_csv(os.path.join(expert_dir, "expert_specific_marker_genes.csv"), index=False)
        
        # 4. 分析专家权重与 TME 免疫分数的定量相关性
        score_keys = [k for k in ["B_Cells", "Tumor_FASN", "CD8_T_Cells"] if k in adata_expert.obs.columns]
        expert_weight_cols = [c for c in adata_expert.obs.columns if "Expert_" in c and "_Weight" in c]
        
        if score_keys and expert_weight_cols:
            corr_records = []
            for exp_col in expert_weight_cols:
                for sig in score_keys:
                    w = adata_expert.obs[exp_col].values
                    s = adata_expert.obs[sig].values
                    corr = safe_corr(w, s, method="spearman")
                    corr_records.append({"Expert": exp_col, "TME_Signature": sig, "Spearman_Corr": corr})
                    
            corr_df = pd.DataFrame(corr_records)
            corr_df.to_csv(os.path.join(expert_dir, "expert_tme_correlation.csv"), index=False)
            print("    ✅ 专家专门化定量证明已生成！")
            
    except Exception as e:
        print(f"    - 专家生物学功能解析失败: {e}")
def run_full_analysis(fold: int, checkpoint_path: str):
    """
    从模型加载、推理到所有下游分析的完整流程。
    """
    output_dir = f"./analysis_results00011/Fold_{fold}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"🚀 开始对 Fold {fold} 进行全流程下游分析")
    print(f"📁 结果保存路径: {output_dir}")
    print("=" * 80)

    # ------------------------------
    # 1. 加载数据与模型
    # ------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    test_dataset = ViT_HER2ST(
        train=False,
        fold=fold,
        flatten=False,
        ori=True,
        adj=True
    )

    sample_name = test_dataset.names[0]
    print(f"当前测试样本: {sample_name}")

    model = MorphoGeneST.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()

    # ------------------------------
    # 2. 执行推理
    # ------------------------------
    data = test_dataset[0]

    # 返回顺序：
    # patches, positions, exps, adj, oris, sfs, centers
    patch, position, exp_gt, adj, oris, sfs, center = data

    patch = patch.unsqueeze(0).to(device)      # [1, N, 3, H, W]
    center_gpu = center.to(device)             # [N, 2]

    with torch.no_grad():
        pred_matrix, expert_weights, attention_profile = infer_model_outputs(
            model,
            patch,
            center_gpu
        )

    gt_matrix = exp_gt.cpu().squeeze().numpy()
    spatial_coords = center.cpu().numpy()

    # ------------------------------
    # 3. 构造 AnnData
    # ------------------------------
    adata_pred = sc.AnnData(pred_matrix)
    adata_pred.var_names = pd.Index(test_dataset.gene_set).astype(str)
    adata_pred.var_names_make_unique()
    adata_pred.obsm["spatial"] = spatial_coords

    adata_gt = sc.AnnData(gt_matrix)
    adata_gt.var_names = pd.Index(test_dataset.gene_set).astype(str)
    adata_gt.var_names_make_unique()
    adata_gt.obsm["spatial"] = spatial_coords

    # 添加病理学家标注
    if sample_name in test_dataset.label and test_dataset.label[sample_name] is not None:
        label = test_dataset.label[sample_name]

        try:
            label = np.asarray(label).astype(str)

            if len(label) == adata_pred.n_obs:
                adata_pred.obs["Pat_Annotation"] = pd.Categorical(label)
                adata_gt.obs["Pat_Annotation"] = pd.Categorical(label)
                print("已添加病理学家标注 Pat_Annotation。")
            else:
                print(
                    f"    - 警告: 样本 {sample_name} 的标签长度 "
                    f"({len(label)}) 与 spots 数量 ({adata_pred.n_obs}) 不匹配，跳过添加。"
                )
        except Exception as e:
            print(f"    - 标签添加失败: {e}")

    # 保存基础矩阵
    np.save(os.path.join(output_dir, "pred_matrix.npy"), pred_matrix)
    np.save(os.path.join(output_dir, "gt_matrix.npy"), gt_matrix)
    np.save(os.path.join(output_dir, "spatial_coords.npy"), spatial_coords)

    adata_pred.write(os.path.join(output_dir, "adata_pred_initial.h5ad"))
    adata_gt.write(os.path.join(output_dir, "adata_gt_initial.h5ad"))

    # ------------------------------
    # 4. 下游分析
    # ------------------------------
    # analyze_marker_genes(adata_gt, adata_pred, output_dir)

    # analyze_prediction_performance(adata_gt, adata_pred, output_dir)

    # analyze_gene_triplet_maps(adata_gt, adata_pred, output_dir)

    adata_pred = analyze_clustering(adata_pred, output_dir)

    # analyze_cluster_confusion_matrix(adata_pred, output_dir)

    analyze_degs_and_pathways(adata_pred, output_dir)

    analyze_tme_scoring(adata_pred, output_dir)

    analyze_tme_violin(adata_pred, output_dir)

    # analyze_co_embedding(adata_gt, adata_pred, output_dir)

    # analyze_spatial_autocorrelation(adata_gt, adata_pred, output_dir)

    # analyze_neighborhood(adata_pred, output_dir)

    # analyze_pseudotime(adata_pred, output_dir)
    # analyze_spatial_state_ordering(adata_pred, output_dir)

    analyze_moe_expert_usage(adata_pred, expert_weights, output_dir)
    analyze_expert_biological_specialization(adata_pred, output_dir)
    # analyze_attention_distance(attention_profile, output_dir)

    # 保存最终对象
    adata_pred.write(os.path.join(output_dir, "adata_pred_final.h5ad"))
    adata_gt.write(os.path.join(output_dir, "adata_gt_final.h5ad"))

    print("=" * 80)
    print(f"✅ Fold {fold} 所有分析任务完成！")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 80)
    print()


# ==============================
# 可选：多 Fold 汇总图
# ==============================
def plot_fold_summary(summary_csv, output_dir="./analysis_results33/summary"):
    """
    如果你整理了多折结果，例如：

    Fold,Sample,PCC,ARI,NMI,RMSE
    5,B1,0.45,0.32,0.41,0.88
    11,C1,0.48,0.35,0.43,0.84

    可以用这个函数画多 fold 汇总柱状图。
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(summary_csv)

    if "Fold" not in df.columns:
        raise ValueError("summary_csv 需要包含 Fold 列。")

    metric_cols = [
        c for c in ["PCC", "ARI", "NMI", "RMSE", "MAE"]
        if c in df.columns
    ]

    if len(metric_cols) == 0:
        raise ValueError("summary_csv 至少需要包含 PCC / ARI / NMI / RMSE / MAE 中的一个指标。")

    df_long = df.melt(
        id_vars=[c for c in ["Fold", "Sample"] if c in df.columns],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Value"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_long,
        x="Fold",
        y="Value",
        hue="Metric"
    )
    plt.title("Cross-fold Performance Summary")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "fold_wise_performance_summary.png"),
        dpi=300
    )
    plt.close()

    # 每个指标单独画
    for metric in metric_cols:
        plt.figure(figsize=(6, 4))
        sns.barplot(
            data=df,
            x="Fold",
            y=metric,
            color="#4C72B0"
        )
        plt.title(f"Fold-wise {metric}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"fold_wise_{metric}.png"),
            dpi=300
        )
        plt.close()


# ==============================
# 入口
# ==============================
if __name__ == "__main__":
    # ----------------------------------------------------
    # 请在这里修改 fold 和模型权重路径
    # ----------------------------------------------------
    TARGET_FOLD = 30

    # 请替换成你自己的权重路径
    CHECKPOINT_PATH = "/data/MorphoGeneST0/model_checkpointsljx30/Fold30.ckpt"

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误：找不到模型权重文件：{CHECKPOINT_PATH}")
        print("请检查 CHECKPOINT_PATH 是否正确。")
    else:
        run_full_analysis(
            fold=TARGET_FOLD,
            checkpoint_path=CHECKPOINT_PATH
        )