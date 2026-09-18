import os
import warnings
import torch
import numpy as np
import anndata as ad
from torch.utils.data import DataLoader

# =========================================================================
# 导入你的原始模型文件模块 (假设你的主要训练代码文件名为 train_main.py)
# 如果名字不同，请将 train_main 替换为你的 Python 脚本名（去掉.py）
# =========================================================================
from main import MorphoGeneST, seed_everything
from dataset import ViT_HER2ST 
from predict import get_R, cluster 

warnings.filterwarnings('ignore')

def test_single_fold(ckpt_path, fold=5, gpu_id=0):
    """
    加载已保存的模型权重并在测试集上进行独立评测
    
    参数:
    ckpt_path: str, PyTorch Lightning 自动保存的 .ckpt 文件路径
    fold: int, 测试的 Fold 编号
    gpu_id: int, 使用的 GPU 显卡号
    """
    print(f"\n=======================================================")
    print(f"🚀 开始单折独立测试 | 目标切分 Fold: {fold}")
    print(f"📁 正在加载权重文件: {ckpt_path}")
    print(f"=======================================================\n")
    
    # 1. 设定随机种子以保证测试复现性
    seed_everything(12000)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 2. 加载测试数据集 (train=False)
    print("[1/3] 正在加载测试集数据...")
    test_dataset = ViT_HER2ST(train=False, fold=fold, flatten=False, ori=True, adj=True, prune='Grid')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)
    
    # 提取标签用于计算 ARI
    label = test_dataset.label[test_dataset.names[0]]

    # 3. 加载预训练模型
    print("[2/3] 正在加载模型权重...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重文件: {ckpt_path}")
        
    model = MorphoGeneST.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to(device)

    # 4. 执行推理过程
    print("[3/3] 开始全图推理计算...")
    preds_list, gts_list, centers_list = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            patch, position, exp_norm, adj, oris, sfs, center, *_ = batch
            
            patch = patch.to(device)
            center = center.to(device).squeeze(0)
            exp_norm = exp_norm.squeeze(0).cpu() 
            
            # 开启预测，禁用数据增强策略(aug=False)
            pred_mse, _ = model(patch, center, aug=False)
            
            preds_list.append(pred_mse.cpu().squeeze())
            gts_list.append(exp_norm)
            centers_list.append(center.cpu())

    # 5. 拼接预测结果并构建 AnnData 格式
    preds = torch.cat(preds_list, dim=0).numpy()
    gts = torch.cat(gts_list, dim=0).numpy()
    cts = torch.cat(centers_list, dim=0).numpy()
    
    adata_pred = ad.AnnData(preds)
    adata_pred.obsm['spatial'] = cts
    adata_gt = ad.AnnData(gts)
    adata_gt.obsm['spatial'] = cts

    # 6. 计算评估指标 (PCC & ARI)
    print("\n" + "="*40)
    print("📊 最终评测结果 (Test Results):")
    
    # 平均基因皮尔逊相关系数
    R_array, _ = get_R(adata_pred, adata_gt)
    mean_gene_pcc = np.nanmean(R_array)
    print(f"🎯 Fold {fold} - Mean Gene-wise PCC: {mean_gene_pcc:.4f}")
    
    # ARI 指标计算
    if label is not None and not isinstance(label, torch.Tensor):
        try:
            clus, ari_val = cluster(adata_pred, label)
            print(f"🎯 Fold {fold} - ARI (严格对齐标准):     {ari_val:.4f}")
        except Exception as e:
            print(f"⚠️ 聚类/ARI 计算遭遇异常: {e}")
    print("="*40 + "\n")

    return mean_gene_pcc


if __name__ == '__main__':
    # =====================================================
    # 使用说明：
    # 将模型训练保存的实际路径填入 ckpt_path。
    # 例如：./model_checkpointsljx400/Fold5-Best-065-0.1234.ckpt
    # =====================================================
    
    # 示例: 加载 Fold 5 的单独测试
    FOLD_TO_TEST = 5
    GPU_ID = 0
    CKPT_PATH = f"/data/MorphoGeneST/model/5.ckpt" # 需替换为真实文件名
    # CKPT_PATH1 = f"/data/MorphoGeneST/model_checkpointsljx30/last-v5.ckpt"
    # CKPT_PATH2 = f"/data/MorphoGeneST/model_checkpointsljx35/last-v5.ckpt"
    # CKPT_PATH3 = f"/data/MorphoGeneST/model_checkpointsljx40/last-v5.ckpt"
    # CKPT_PATH4 = f"/data/MorphoGeneST/model_checkpointsljx45/last-v5.ckpt"
    # CKPT_PATH5 = f"/data/MorphoGeneST/model_checkpointsljx400/Fold23-Best-epoch=296-train_loss=0.3349.ckpt"
    # 也可以利用 for 循环测试所有不同的 fold
    # folds = [5, 11, 17, 23, 26, 30]
    # for f in folds:
    #     ckpt_file = f"./model_checkpointsljx204/Fold{f}-Best-xxxx.ckpt" # 填写真实的文件名形式
    #     test_single_fold(ckpt_file, fold=f, gpu_id=0)
    
    test_single_fold(ckpt_path=CKPT_PATH, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
    # test_single_fold(ckpt_path=CKPT_PATH1, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
    # test_single_fold(ckpt_path=CKPT_PATH2, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
    # test_single_fold(ckpt_path=CKPT_PATH3, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
    # test_single_fold(ckpt_path=CKPT_PATH4, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
    # test_single_fold(ckpt_path=CKPT_PATH5, fold=FOLD_TO_TEST, gpu_id=GPU_ID)
