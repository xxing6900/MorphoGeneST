import os
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ViT_HER2ST
from main import MorphoGeneST

# 解决图像显示中文或负号问题
plt.rcParams['axes.unicode_minus'] = False 

def run_downstream_tasks(fold=5, checkpoint_path=None):
    # 1. 重新实例化数据集并挑出这一个 Fold 的测试数据
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    sample_name = test_dataset.names[0] # 例如 'H1' 或者是 'B1'
    print(f"🔬 开始对 {sample_name} 进行下游生物学分析...")
    
    # 2. 如果有 Checkpoint 则加载，否则直接用你之前保存在列表里的结果（这里以重现推理为例）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MorphoGeneST.load_from_checkpoint(checkpoint_path).to(device)
    model.eval()
    
    data = test_dataset[0] # batch_size = 1
    patch, position, exp_gt, adj, oris, sfs, center = [d for d in data]
    patch = patch.unsqueeze(0).to(device)
    center = center.to(device)
    
    with torch.no_grad():
        pred_mse, _ = model(patch, center, aug=False)
        pred_matrix = pred_mse.cpu().squeeze().numpy()
        
    # 3. 构造完美的 AnnData 对象，专供 Scanpy 分析
    adata_pred = sc.AnnData(pred_matrix)
    adata_pred.var_names = test_dataset.gene_set
    adata_pred.obsm['spatial'] = center.cpu().numpy()
    
    # 获取病理学家的 GT Label (如果有的切片存在的话)
    if test_dataset.label[sample_name] is not None and not isinstance(test_dataset.label[sample_name], torch.Tensor):
        adata_pred.obs['Pat_Annotation'] = test_dataset.label[sample_name]
        
    # ====================================================
    # 任务 1：核心标志基因的空间热图可视化 (去噪展示)
    # ====================================================
    print("📈 生成核心基因的空间分布对比图...")
    target_genes = ['GNAS', 'FASN', 'SCD', 'FN1'] # 提取乳腺癌强相关基因
    genes_to_plot = [g for g in target_genes if g in adata_pred.var_names]
    
    if len(genes_to_plot) > 0:
        sc.pl.spatial(adata_pred, color=genes_to_plot, spot_size=20, 
                      cmap='magma', title=[f"Predicted {g}" for g in genes_to_plot],
                      show=False)
        plt.savefig(f"./results_{sample_name}_marker_genes.png", dpi=300)
    
    # ====================================================
    # 任务 2：完全无监督的聚类，识别空间域
    # ====================================================
    print("🧬 进行 Leiden 聚类与空间域识别...")
    sc.pp.neighbors(adata_pred, n_neighbors=15, use_rep='X')
    sc.tl.leiden(adata_pred, resolution=0.8) # resolution 调节类的多少
    
    # 绘制聚类结果的空间分布
    sc.pl.spatial(adata_pred, color=['leiden'], spot_size=20, show=False)
    plt.savefig(f"./results_{sample_name}_leiden_clusters.png", dpi=300)
    
    # 如果有真实标签，画在一起比对
    if 'Pat_Annotation' in adata_pred.obs.keys():
        sc.pl.spatial(adata_pred, color=['Pat_Annotation', 'leiden'], spot_size=20, show=False)
        plt.savefig(f"./results_{sample_name}_cluster_vs_GT.png", dpi=300)
        
    # ====================================================
    # 任务 3：计算每个空间域的特异性差异基因 (DEGs)
    # ====================================================
    print("📊 寻找每个空间的差异表达基因 (DEGs)...")
    sc.tl.rank_genes_groups(adata_pred, 'leiden', method='t-test')
    sc.pl.rank_genes_groups_heatmap(adata_pred, n_genes=5, groupby='leiden', show=False)
    plt.savefig(f"./results_{sample_name}_DEG_heatmap.png", dpi=300)
    
    # 导出差异基因表格供外部做 Pathway 分析 (GO/KEGG)
    result = adata_pred.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    deg_df = pd.DataFrame({group + '_' + key[:1]: result[key][group]
        for group in groups for key in ['names', 'pvals_adj']})
    deg_df.to_csv(f"./results_{sample_name}_DEGs.csv", index=False)
    
    # ====================================================
    # 任务 4：(结合你 utils 的知识) 构建你的特征基因打分图
    # ====================================================
    print("⚔️ 肿瘤微环境 (TME) 特征打分...")
    from utils import BCELL, TUMOR, CD8T # 直接复用你的先验知识
    tme_dict = {'B_Cells': BCELL, 'Tumor': TUMOR, 'CD8_T_Cells': CD8T}
    
    for key, genes in tme_dict.items():
        sub_genes = [g for g in genes if g in adata_pred.var_names]
        if len(sub_genes) > 0:
            sc.tl.score_genes(adata_pred, gene_list=sub_genes, score_name=key)
            
    score_keys = [k for k in tme_dict.keys() if k in adata_pred.obs.columns]
    if len(score_keys) > 0:
        sc.pl.spatial(adata_pred, color=score_keys, spot_size=20, cmap='viridis', show=False)
        plt.savefig(f"./results_{sample_name}_TME_scoring.png", dpi=300)
        
    print(f"✅ {sample_name} 分析完成，图表已保存！")

# 运行示例（请将相对路径替换为你训练出的最新 ckpt 文件）
if __name__ == '__main__':
    # 你可以把 32 个测完的权重在这个循环里批量过一遍
    ckpt_path = "./model_checkpointsljx25/Fold5-Best-001-0.xxx.ckpt" 
    # run_downstream_tasks(fold=5, checkpoint_path=ckpt_path)