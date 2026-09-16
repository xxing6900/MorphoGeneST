
import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import gseapy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import adjusted_rand_score
from PIL import Image
from predict import get_R, cluster 
# 导入您项目中的自定义模块
from main import MorphoGeneST
from dataset import ViT_HER2ST
from utils import BCELL, CD8T, TUMOR # 复用您已有的基因集

# --- 全局美化设置 ---
# plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.unicode_minus'] = False
# sns.set_style("whitegrid")


def analyze_marker_genes(adata_gt, adata_pred, output_dir):
    """任务1: 核心标志基因的空间可视化 (风格已对齐)"""
    print("  [1/8] 正在生成标志基因空间对比图并计算 PCC...")
    target_genes = ['IGHA1'] # 这里可按需扩展为 ['FN1', 'GNAS', 'FASN', ...]
    genes_to_plot = [g for g in target_genes if g in adata_pred.var_names]
    
    if not genes_to_plot: 
        print("    ⚠️ 未找到目标基因，跳过任务1。")
        return

    # 获取全局的极值以统一 GT 和 Pred 的颜色刻度
    vmin = adata_gt[:, genes_to_plot].X.min()
    vmax = adata_gt[:, genes_to_plot].X.max()

    # == 计算 PCC 并生成 Pred 标题 ==
    pred_titles = []
    print("    📊 特定基因 PCC 指标:")
    for g in genes_to_plot:
        # 使用 np.array() 转换并摊平，与代码2逻辑一致
        gt_exp = np.array(adata_gt[:, g].X).flatten()
        pred_exp = np.array(adata_pred[:, g].X).flatten()
        
        # 计算 PCC
        pcc_val, _ = pearsonr(gt_exp, pred_exp)
        print(f"      - {g}: PCC = {pcc_val:.4f}")
        
        # 严格对齐代码2的标题格式
        pred_titles.append(f"PCC:{pcc_val:.3f}")
    fig, ax = plt.subplots()
    sc.pl.spatial(adata_pred, 
                    color=genes_to_plot, 
                    size=1.4,             # 稍微加大点的大小，使其更饱满
                    alpha_img=0.5,        # 底部背景亮度
                    frameon=False,        # 彻底去掉外框黑线
                    palette='tab20', 
                    show=False, 
                    cmap='magma', 
                    vmin=vmin, 
                    vmax=vmax, 
                    ax=ax,
                    title="",             # 禁用 scanpy 默认标题
                    legend_loc=None)      # 彻底去掉右侧那一排分类标签

            # 5. 手动添加对齐图片的标题指标
            # 如果是您自己的模型，请将 fontweight 改为 'bold'
    ax.set_title(f"PCC: {pcc_val:.3f}", fontsize=24, fontweight='normal', pad=12)
            
            # 彻底清除坐标轴残留
    ax.set_xlabel("")
    ax.set_ylabel("")
            # 6. 保存 (bbox_inches='tight' 确保不留多余白边)
    plt.savefig(f"{output_dir}/predicted.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close('all')    
    # ===== 画 Ground Truth 图 =====
    # 标题仅保留基因名 (例如 "FN1")
    # sc.pl.spatial(adata_gt, color=genes_to_plot, size=1.4, alpha_img=0.5, frameon=False, 
    #               cmap='magma', vmin=vmin, vmax=vmax, title=[f"{g}" for g in genes_to_plot],
    #               show=False)
    # # 严格对齐代码2的保存路径格式
    # plt.savefig(f"{output_dir}/task1_marker_ground_truth.png", dpi=300, bbox_inches='tight')

    # # ===== 画预测图 Pred =====
    # # 采用紧凑的 PCC 标题
    # sc.pl.spatial(adata_pred, color=genes_to_plot, size=1.4, alpha_img=0.5, frameon=False, 
    #               cmap='magma', vmin=vmin, vmax=vmax, title="",
    #               show=False)
    
    # # 严格对齐代码2的保存路径格式
    # plt.savefig(f"{output_dir}/task1_marker_predicted.png", dpi=300, bbox_inches='tight')
    
    # # 彻底关闭图窗，对齐代码2逻辑
    # plt.close('all')

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def analyze_clustering(adata_pred, output_dir):    
    # 如果有真实标注
    # if 'Pat_Annotation' in adata_pred.obs.columns:
    #     # 1. 提取标签并剔除无效项 (undetermined 等)
    #     labels = adata_pred.obs['Pat_Annotation'].values
    #     valid_mask = ~pd.Series(labels).astype(str).isin(['undetermined', '-1', 'nan', 'NaN']).values
        
    #     # 2. 计算有效类别数，实现“少一类”的聚类效果
    #     valid_labels = labels[valid_mask]
    #     n_clusters = len(np.unique(valid_labels)) 
        
    #     # 3. 执行 KMeans
    #     if 'X_pca' not in adata_pred.obsm:
    #         sc.pp.pca(adata_pred)
            
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=0)
    #     adata_pred.obs['kmeans'] = kmeans.fit_predict(adata_pred.obsm['X_pca']).astype(str)
        
    #     # 4. 计算 ARI
    #     pred_labels_valid = adata_pred.obs['kmeans'].values[valid_mask]
    #     ari_score = adjusted_rand_score(valid_labels, pred_labels_valid)
    #     print(f"    📊 对齐后的 ARI 指标: {ari_score:.4f}")
        
    #     # ---------------------------------------------------------
    #     # 🌟 核心修改：生成纯净版图像 (对准您要求的 image.png 效果)
    #     # ---------------------------------------------------------
    #     fig, ax = plt.subplots(figsize=(6, 6))
        
    #     # 关键参数：legend_loc=None (删除侧边颜标), frameon=False (删除外框), size=1.3 (点大一点)
    #     sc.pl.spatial(adata_pred, 
    #                   color='kmeans', 
    #                   size=1.4, 
    #                   alpha_img=0.5, 
    #                   frameon=False, 
    #                   palette='tab20', 
    #                   show=False, 
    #                   ax=ax,
    #                   title="",        # 去掉自带标题
    #                   legend_loc=None  # ⚠️ 彻底删掉侧边的分类标签图例
    #                  )

    #     # 手动添加位于顶部的“ARI: 0.xx”标题，加粗显示
    #     ax.set_title(f"ARI: {ari_score:.2f}", fontsize=24, fontweight='normal', pad=12)
        
    #     # 移除坐标轴文字
    #     ax.set_xlabel(""); ax.set_ylabel("")

    #     # 保存图片 (bbox_inches='tight' 确保不留白边)
    #     plt.savefig(os.path.join(output_dir, "ARI_Pure_Result.png"), 
    #                 dpi=300, bbox_inches='tight', pad_inches=0)
    #     plt.close()
        
    #     # 为后续分析同步分组名
    #     adata_pred.obs['Leiden_Clusters'] = adata_pred.obs['kmeans']
        
    # else:
        # 无标签时的纯净版输出

    # """任务2: 空间域识别 (纯净版 - 仅保留带指标的图像，无图例，加粗标题)"""
    print("  [2/8] 正在生成纯净版空间聚类对比图...")
    print("    ⚠️ 缺乏真实标签，执行 Leiden 盲聚类...")
    sc.pp.neighbors(adata_pred, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_pred, resolution=0.8, key_added='Leiden_Clusters')
        
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.spatial(adata_pred, color='Leiden_Clusters', size=1.3, alpha_img=0.5, 
              frameon=False, palette='tab20', show=False, ax=ax, title="", legend_loc=None)
    ax.set_title("Predicted Domains", fontsize=20, fontweight='normal', pad=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.savefig(os.path.join(output_dir, "Pure_Domain_Result.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return adata_pred
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

def analyze_co_embedding(adata_gt, adata_pred, output_dir):
    """任务5: 潜空间流形对齐图 (UMAP Co-embedding)"""
    print("  [5/8] 正在生成UMAP潜空间对齐图...")
    adata_gt.obs['Source'] = 'Ground Truth'
    adata_pred.obs['Source'] = 'Predicted'
    adata_concat = adata_gt.concatenate(adata_pred, batch_key='Source')
    
    # Scanpy 标准降维流程: PCA -> Neighbors -> UMAP
    sc.pp.pca(adata_concat)
    sc.pp.neighbors(adata_concat, n_neighbors=15, use_rep='X_pca') 
    sc.tl.umap(adata_concat)
    
    # 填充 Leiden_Clusters 给 Ground Truth 部分 (避免因 NaN 画图报错)
    if 'Leiden_Clusters' in adata_concat.obs.columns:
        adata_concat.obs['Leiden_Clusters'] = adata_concat.obs['Leiden_Clusters'].cat.add_categories(['N/A']).fillna('N/A')
    
    sc.pl.umap(adata_concat, color=['Source', 'Leiden_Clusters'], wspace=0.3, show=False)
    plt.savefig(os.path.join(output_dir, "umap_co_embedding.png"), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_spatial_autocorrelation(adata_gt, adata_pred, output_dir):
    """任务6: 空间自相关性 (Moran's I) 对比"""
    print("  [6/8] 正在对比真实与预测的Moran's I...")
    try:
        sq.gr.spatial_neighbors(adata_gt, coord_type="generic")
        sq.gr.spatial_neighbors(adata_pred, coord_type="generic")
        
        sq.gr.spatial_autocorr(adata_gt, mode="moran", genes=adata_gt.var_names, n_perms=100, n_jobs=4)
        sq.gr.spatial_autocorr(adata_pred, mode="moran", genes=adata_pred.var_names, n_perms=100, n_jobs=4)

        moran_df = pd.DataFrame({
            'Moran_I_GT': adata_gt.uns['moranI']['I'],
            'Moran_I_Pred': adata_pred.uns['moranI']['I']
        }).dropna()
        
        r, p = spearmanr(moran_df['Moran_I_GT'], moran_df['Moran_I_Pred'])
        
        plt.figure(figsize=(6, 6))
        sns.regplot(data=moran_df, x='Moran_I_GT', y='Moran_I_Pred',
                    scatter_kws={'alpha': 0.5, 's': 15}, line_kws={'color': 'red'})
        plt.title(f"Moran's I Correlation (Spearman R={r:.3f})")
        plt.xlabel("Moran's I (Ground Truth)")
        plt.ylabel("Moran's I (Predicted)")
        plt.savefig(os.path.join(output_dir, "morans_i_correlation.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"    - Moran's I 分析失败: {e}")
    plt.close()

def analyze_neighborhood(adata_pred, output_dir):
    """任务7: 微环境空间邻域富集分析"""
    print("  [7/8] 正在分析空间邻域富集关系...")
    try:
        # 确保聚类数大于1
        if len(adata_pred.obs['Leiden_Clusters'].unique()) > 1:
            sq.gr.nhood_enrichment(adata_pred, cluster_key="Leiden_Clusters")
            sq.pl.nhood_enrichment(adata_pred, cluster_key="Leiden_Clusters", figsize=(6,6), show=False)
            plt.savefig(os.path.join(output_dir, "neighborhood_enrichment.png"), dpi=300, bbox_inches='tight')
        else:
            print("    - 跳过邻域分析：聚类数不足。")
    except Exception as e:
        print(f"    - 邻域富集分析失败: {e}")
    plt.close()

def analyze_pseudotime(adata_pred, output_dir):
    """任务8: 肿瘤空间进化轨迹推断"""
    print("  [8/8] 正在推断空间伪时间轨迹...")
    try:
        # 自动选择一个聚类中点最多的cluster作为起点
        root_cluster = adata_pred.obs['Leiden_Clusters'].value_counts().idxmax()
        root_indices = np.where(adata_pred.obs['Leiden_Clusters'] == root_cluster)[0]
        if len(root_indices) > 0:
            adata_pred.uns['iroot'] = root_indices[0]
            sc.tl.dpt(adata_pred)
            # 伪时间空间图 (带灰度背景)
            sc.pl.spatial(adata_pred, color='dpt_pseudotime', size=1.2, alpha_img=0, frameon=False, 
                          cmap='Spectral_r', title="Spatial Pseudotime Trajectory", show=False)
            plt.savefig(os.path.join(output_dir, "pseudotime_trajectory.png"), dpi=300, bbox_inches='tight')
        else:
            print("    - 跳过伪时间分析：找不到根节点。")
    except Exception as e:
        print(f"    - 伪时间分析失败: {e}")
    plt.close()
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

def run_full_analysis(fold: int, checkpoint_path: str):
    """
    主函数，执行从模型加载到所有下游分析的全流程。
    """
    # --- 0. 准备工作 ---
    output_dir = f"./analysis_results23/Fold_{fold}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 开始对 Fold {fold} 进行全流程下游分析，结果将保存在: {output_dir}")
    
    # --- 1. 加载数据与模型，执行推理 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True)
    sample_name = test_dataset.names[0]
    
    model = MorphoGeneST.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    
    data = test_dataset[0]
    patch, _, exp_gt, _, _, _, center = [d for d in data]
    patch = patch.unsqueeze(0).to(device)
    center_gpu = center.to(device)
    
    with torch.no_grad():
        pred_mse, _ = model(patch, center_gpu, aug=False)
        pred_matrix = pred_mse.cpu().squeeze().numpy()
        gt_matrix = exp_gt.squeeze().numpy()
        spatial_coords = center.cpu().numpy()

    # --- 2. 构造图像字典作为背景底图 ---
    print("  [*] 正在加载并处理原始组织学图像底图...")
    # (a) 使用 dataset 内嵌的 get_img 提取原始图像
    orig_img_pil = test_dataset.get_img(sample_name)
    # (b) 将其转为灰度(L)，再转回三通道(RGB)，避免 scanpy 遇到单通道图像报错
    gray_img_array = np.array(orig_img_pil.convert('L').convert('RGB')) / 255.0
    
    # (c) 构建遵从 Scanpy 空间规范的字典
    spatial_dict = {
        sample_name: {
            "images": {"hires": gray_img_array},
            "scalefactors": {
                "tissue_hires_scalef": 1.0,         # 表示 1像素 = 1空间坐标单位
                "spot_diameter_fullres": 110.0      # spot的默认大小 (可按需调整这里的数字)
            }
        }
    }

    # --- 3. 构造 AnnData 对象并注入图像 ---
    # 预测数据
    adata_pred = sc.AnnData(pred_matrix)
    adata_pred.var_names = test_dataset.gene_set
    adata_pred.obsm['spatial'] = spatial_coords
    adata_pred.uns['spatial'] = spatial_dict  # 注入背景底图
    
    # 真实数据
    adata_gt = sc.AnnData(gt_matrix)
    adata_gt.var_names = test_dataset.gene_set
    adata_gt.obsm['spatial'] = spatial_coords
    adata_gt.uns['spatial'] = spatial_dict    # 注入背景底图

    # 添加病理学家标注 (如果存在)
    if sample_name in test_dataset.label and test_dataset.label[sample_name] is not None:
        label = test_dataset.label[sample_name]
        # 确保label长度和spot数量一致
        if len(label) == adata_pred.n_obs:
            adata_pred.obs['Pat_Annotation'] = label
            adata_gt.obs['Pat_Annotation'] = label
        else:
            print(f"    - 警告: 样本 {sample_name} 的标签长度 ({len(label)}) 与 spots 数量 ({adata_pred.n_obs}) 不匹配, 跳过添加。")

    # --- 4. 依次执行所有下游任务 ---
    # analyze_marker_genes(adata_gt, adata_pred, output_dir)
    # adata_pred = analyze_clustering(adata_pred, output_dir) # 更新adata以包含聚类信息
    # analyze_degs_and_pathways(adata_pred, output_dir)
    # analyze_tme_scoring(adata_pred, output_dir)
    # analyze_co_embedding(adata_gt, adata_pred, output_dir)
    # analyze_spatial_autocorrelation(adata_gt, adata_pred, output_dir)
    # analyze_neighborhood(adata_pred, output_dir)
    # analyze_pseudotime(adata_pred, output_dir)
    adata_pred = analyze_clustering(adata_pred, output_dir) # 空间域恢复 (必须开启，且要接收返回值用于后续分组！)
    analyze_tme_scoring(adata_pred, output_dir)             # 免疫映射
    analyze_degs_and_pathways(adata_pred, output_dir)       # 通路分析

    print(f"✅✅✅ Fold {fold} 所有分析任务完成！✅✅✅\n")


if __name__ == '__main__':
    # ----------------------------------------------------
    #  👇👇👇 请在这里修改您的 Fold 和模型权重路径 👇👇👇
    # ----------------------------------------------------
    TARGET_FOLD = 17 # 示例：选择有病理学家标注的切片进行分析，如 5 (B1), 11 (C1), 17 (D1)
    
    # 示例路径，请务必替换为您自己的最新权重文件！
    CHECKPOINT_PATH = "/data/MorphoGeneST0/model_checkpointsljx30/Fold17.ckpt" 

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"错误：找不到模型权重文件 '{CHECKPOINT_PATH}'。请检查路径是否正确。")
    else:
        run_full_analysis(fold=TARGET_FOLD, checkpoint_path=CHECKPOINT_PATH)
        TARGET_FOLD = 11 # 示例：选择有病理学家标注的切片进行分析，如 5 (B1), 11 (C1), 17 (D1)
    
