import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
import numpy as np
import anndata as ad
import scipy.stats as stats
from datetime import datetime
from torch.utils.data import DataLoader

# 导入自定义组件
from dataset import ViT_HER2ST 
from predict import get_R, cluster
from modules import PathologyExtractor, SAGTBlock, MoE_Decoder
from zinb import MeanAct, DispAct, ZINB_loss

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# =========================================================================
# 1. 模型定义保持不变
# =========================================================================
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
        
        self.head_mse = nn.Sequential(MoE_Decoder(embed_dim, n_genes, num_experts=4), nn.ReLU())
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


# =========================================================================
# 2. 🌟 修复后的精准寻找权重函数
# =========================================================================
def find_checkpoint(fold, base_dir="./model_checkpointsljx30"):
    """使用正则表达式精确匹配 Fold 编号，避免将 Fold1 定位到 Fold13"""
    search_dirs = [base_dir, "./model_checkpointsljx30"]
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
            
        # 遍历目录下所有 ckpt 文件
        all_ckpts = glob.glob(os.path.join(d, "*.ckpt"))
        for ckpt_path in all_ckpts:
            filename = os.path.basename(ckpt_path)
            # 正则解释: 匹配以 Fold 开头，紧跟一串数字，然后是任意字符，最后以 .ckpt 结尾
            # 例如 "Fold13.ckpt" -> 提取出的数字是 "13"
            match = re.match(r"^Fold(\d+)(.*)\.ckpt$", filename)
            if match:
                # 强制转化为整型对比：严格判断 1 == 1
                ckpt_fold_num = int(match.group(1))
                if ckpt_fold_num == fold:
                    return ckpt_path
                    
    raise FileNotFoundError(f"精准匹配失败：找不到 Fold {fold} 对应的独立模型权重文件！搜索的目录：{search_dirs}")


# =========================================================================
# 3. 评估与输出逻辑保持不变
# =========================================================================
def log_and_print(message, file_handle):
    print(message)
    file_handle.write(message + "\n")

def evaluate_fold(fold, model_class, file_handle, gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    log_and_print(f"\n[{fold}] {'='*50}", file_handle)
    log_and_print(f"🌟 正在加载 Fold {fold} 的测试数据和模型...", file_handle)
    
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    label = test_dataset.label[test_dataset.names[0]]
    
    ckpt_path = find_checkpoint(fold)
    log_and_print(f"📥 成功找到完全对应的权重: {ckpt_path}", file_handle)
    model = model_class.load_from_checkpoint(ckpt_path)
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
    
    adata_pred = ad.AnnData(preds)
    adata_pred.obsm['spatial'] = cts
    adata_gt = ad.AnnData(gts)
    adata_gt.obsm['spatial'] = cts
    
    num_spots = adata_pred.n_obs
    num_genes = adata_pred.n_vars
    
    R_array, P_array = get_R(adata_pred, adata_gt)
    valid_idx = ~np.isnan(R_array)
    valid_R = R_array[valid_idx]
    valid_P = P_array[valid_idx]
    
    mean_pcc = np.mean(valid_R)
    sem_pcc = stats.sem(valid_R)
    ci_pcc = stats.t.interval(0.95, df=len(valid_R)-1, loc=mean_pcc, scale=sem_pcc)
    
    significant_genes_ratio = np.sum(valid_P < 0.05) / len(valid_P)
    
    ari_val = None
    if label is not None and not isinstance(label, torch.Tensor):
        try:
            _, ari_val = cluster(adata_pred, label)
        except Exception:
            pass

    log_and_print(f"📊 基础统计 | Spots: {num_spots} | 保留 Genes: {num_genes}", file_handle)
    log_and_print(f"🎯 PCC 结果 | 均值: {mean_pcc:.4f} | 95% CI: [{ci_pcc[0]:.4f}, {ci_pcc[1]:.4f}]", file_handle)
    log_and_print(f"🔬 统计检验 | 显著相关基因比例 ($p < 0.05$): {significant_genes_ratio*100:.2f}%", file_handle)
    if ari_val is not None:
        log_and_print(f"🎯 ARI 结果 | {ari_val:.4f}", file_handle)
    
    return {
        "fold": fold,
        "spots": num_spots,
        "genes": num_genes,
        "pcc": mean_pcc,
        "ari": ari_val
    }


if __name__ == '__main__':
    evaluated_folds = [0, 1, 2, 3, 4, 5,6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22,23, 24, 25,26, 27, 28, 29,30, 31]
    output_filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    results = []
    total_spots = 0
    
    with open(output_filename, "w", encoding="utf-8") as f:
        log_and_print(f"=== 模型定量评估报告 ===", f)
        log_and_print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        
        for fold in evaluated_folds:
            try:
                res_dict = evaluate_fold(fold=fold, model_class=MorphoGeneST, file_handle=f, gpu_id=0)
                results.append(res_dict)
                total_spots += res_dict["spots"]
            except Exception as e:
                log_and_print(f"❌ 评估 Fold {fold} 失败: {e}", f)

        if len(results) > 0:
            log_and_print("\n" + "="*50, f)
            log_and_print("🏆 宏观评估报告 (Global Evaluation Summary) 🏆", f)
            log_and_print("="*50, f)
            
            all_pccs = [r["pcc"] for r in results]
            all_aris = [r["ari"] for r in results if r["ari"] is not None]
            
            global_mean_pcc = np.mean(all_pccs)
            global_sem_pcc = stats.sem(all_pccs)
            global_ci_pcc = stats.t.interval(0.95, df=len(all_pccs)-1, loc=global_mean_pcc, scale=global_sem_pcc)
            
            log_and_print(f"👥 临床患者数量 (Patients): 8 (HER2+ 数据集)", f)
            log_and_print(f"🧫 评估切片数量 (Sections): {len(results)}", f)
            log_and_print(f"🧬 评估斑点总数 (Total Spots): {total_spots}", f)
            log_and_print(f"🧪 统一保留基因数 (Genes per spot): {results[0]['genes']}\n", f)
            
            log_and_print(f"🌟 跨切片全局平均 PCC:", f)
            log_and_print(f"   => {global_mean_pcc:.4f}  [95% CI: {global_ci_pcc[0]:.4f}, {global_ci_pcc[1]:.4f}]", f)
            
            if len(all_aris) > 0:
                global_mean_ari = np.mean(all_aris)
                global_sem_ari = stats.sem(all_aris)
                global_ci_ari = stats.t.interval(0.95, df=len(all_aris)-1, loc=global_mean_ari, scale=global_sem_ari)
                log_and_print(f"🌟 跨切片全局平均 ARI:", f)
                log_and_print(f"   => {global_mean_ari:.4f}  [95% CI: {global_ci_ari[0]:.4f}, {global_ci_ari[1]:.4f}]", f)
            
            log_and_print("="*50, f)
            
    print(f"\n✅ 所有评估结果已成功提取并保存至文件: {output_filename}")