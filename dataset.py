# dataset.py
import os
import glob
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scprep as scp
from PIL import ImageFile, Image
from collections import defaultdict as dfd

# 从 utils 导入邻接矩阵构建工具
from utils import calcADJ

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """
    针对 HER2+ Breast Cancer ST 空间转录组数据集的加载器
    """
    def __init__(self, train=True, fold=0, r=4, flatten=False, ori=True, adj=True, prune='Grid', neighs=4):
        super(ViT_HER2ST, self).__init__()
        
        # 数据集基础路径 (请确保您的文件夹存放符合该结构)
        self.cnt_dir = './data/her2st/data/ST-cnts'
        self.img_dir = './data/her2st/data/ST-imgs'
        self.pos_dir = './data/her2st/data/ST-spotfiles'
        self.lbl_dir = './data/her2st/data/ST-pat/lbl'
        
        # patch 的半径 (尺寸: 2*r x 2*r)
        self.r = 224 // r

        # 尝试加载高变基因列表，通常是 785 个
        try:
            gene_list = list(np.load('./data/her_hvg_cut_1000.npy', allow_pickle=True))
        except FileNotFoundError:
            raise FileNotFoundError("找不到 ./data/her_hvg_cut_1000.npy，请确保该基因列表存在！")
            
        self.gene_list = gene_list
        self.train = train
        self.ori = ori
        self.adj = adj
        self.flatten = flatten

        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]

        # Her2ST的样本划分
        samples = names[1:33]
        te_names = [samples[fold]]
        tr_names = list(set(samples) - set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(f"[{'Train' if train else 'Test'}] 当前处理样本: {self.names}")
        print(f"Loading images for {'Train' if train else 'Test'}...")
        self.img_dict = {i: torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        
        print(f"Loading metadata for {'Train' if train else 'Test'}...")
        self.meta_dict = {i: self.get_meta(i) for i in self.names}
        
        # 标签处理 (用于 ARI 计算)
        self.label = {i: None for i in self.names}
        self.lbl2id = {
            'invasive cancer': 0, 'breast glands': 1, 'immune infiltrate': 2, 
            'cancer in situ': 3, 'connective tissue': 4, 'adipose tissue': 5, 'undetermined': -1
        }
        
        if not train and self.names[0] in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
            self.lbl_dict = {i: self.get_lbl(i) for i in self.names}
            idx = self.meta_dict[self.names[0]].index
            lbl = self.lbl_dict[self.names[0]]
            lbl = lbl.loc[idx, :]['label'].values
            # 存储真实的字符串或ID皆可，用于 utils 中的 KMeans 对比
            self.label[self.names[0]] = lbl 
            
        elif train:
            for i in self.names:
                idx = self.meta_dict[i].index
                if i in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G2', 'H1', 'J1']:
                    lbl = self.get_lbl(i)
                    lbl = lbl.loc[idx, :]['label'].values
                    lbl = torch.Tensor(list(map(lambda x: self.lbl2id[x], lbl)))
                    self.label[i] = lbl
                else:
                    self.label[i] = torch.full((len(idx),), -1)

        self.gene_set = list(gene_list)
        
        # 构建 MSE 所需的归一化基因表达矩阵
        self.exp_dict = {
            i: scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i, m in self.meta_dict.items()
        }
        
        # 构建 ZINB 所需的原始基因 Counts 和 Size Factors
        if self.ori:
            self.ori_dict = {i: m[self.gene_set].values for i, m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
                
        # 提取各个 spot 的物理中心坐标和相对坐标
        self.center_dict = {
            i: np.floor(m[['pixel_x', 'pixel_y']].values).astype(int) 
            for i, m in self.meta_dict.items()
        }
        self.loc_dict = {i: m[['x', 'y']].values for i, m in self.meta_dict.items()}
        
        # 预先计算用于旧版图网络的物理邻接矩阵 (备用或相乘用)
        if self.adj:
            self.adj_dict = {
                i: calcADJ(m, neighs, pruneTag=prune)
                for i, m in self.loc_dict.items()
            }
            
        self.patch_dict = dfd(lambda: None)
        self.id2name = dict(enumerate(self.names))

    def __getitem__(self, index):
        ID = self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1, 0, 2) # [W, H, C]
        
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
            
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        
        if self.adj:
            adj = self.adj_dict[ID]
            
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        
        # 裁剪图像 Spot Patch
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))
                
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r), (y-self.r):(y+self.r), :]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2, 0, 1) # [C, H, W]
            self.patch_dict[ID] = patches

        # 严格按照主模型需要的顺序返回
        data = [patches, positions, exps]
        
        if self.adj:
            data.append(adj)
        else:
            data.append(torch.zeros(1)) # 凑数占位，以防解包报错
            
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        else:
            data += [torch.zeros(1), torch.zeros(1)]
            
        # center 必须放在最后
        data.append(torch.Tensor(centers))
        
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self, name):
        pre = os.path.join(self.img_dir, name[0], name)
        fig_name = os.listdir(pre)[0]
        path = os.path.join(pre, fig_name)
        im = Image.open(path)
        return im

    def get_cnt(self, name):
        path = os.path.join(self.cnt_dir, f"{name}.tsv")
        df = pd.read_csv(path, sep='\t', index_col=0)
        return df

    def get_pos(self, name):
        path = os.path.join(self.pos_dir, f"{name}_selection.tsv")
        df = pd.read_csv(path, sep='\t')

        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        df['id'] = [f"{x[i]}x{y[i]}" for i in range(len(x))]
        return df

    def get_meta(self, name):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        return meta

    def get_lbl(self, name):
        path = os.path.join(self.lbl_dir, f"{name}_labeled_coordinates.tsv")
        df = pd.read_csv(path, sep='\t')

        x = np.around(df['x'].values).astype(int)
        y = np.around(df['y'].values).astype(int)
        df['id'] = [f"{x[i]}x{y[i]}" for i in range(len(x))]
        
        columns_to_drop = ['pixel_x', 'pixel_y', 'x', 'y']
        df.drop([col for col in columns_to_drop if col in df.columns], inplace=True, axis=1)
        df.set_index('id', inplace=True)
        return df
class ViT_SKIN(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self,train=True,r=4,norm=False,fold=0,flatten=True,ori=False,adj=False,prune='NA',neighs=4):
        super(ViT_SKIN, self).__init__()

        self.dir = './data/GSE144240_RAW/'
        self.r = 224//r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        gene_list = list(np.load('data/skin_hvg_cut_1000.npy',allow_pickle=True))

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}

        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)))
                for i,m in self.meta_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
                for i,m in self.meta_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)
