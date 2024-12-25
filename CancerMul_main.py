import torch
from torch.utils.data import DataLoader
import sys, argparse, os, glob, datetime
import pandas as pd
import numpy as np
from torch.utils.data import Dataset 
import time 
import random 
import torch.backends.cudnn as cudnn
import json
import os
from Opt.lookahead import Lookahead
from Opt.radam import RAdam
from utils.loss_func import NLLSurvLoss
from sksurv.metrics import concordance_index_censored
from itertools import chain
torch.multiprocessing.set_sharing_strategy('file_system')
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


    
class BagDataset(Dataset):
    def __init__(self,train_path, rna, dna, status_info,subtype_info,stage_info,  args) -> None:
        super(BagDataset).__init__()
        self.train_path = train_path
        self.rna = rna
        self.dna = dna
        self.status_info = status_info
        self.subtype_info = subtype_info
        self.stage_info = stage_info
        self.args = args

    def get_bag_feats(self,csv_file_df, args):

        feats_csv_path = os.path.join('./example/Patch_feat/Patch_feat', csv_file_df.iloc[0])
        df = pd.read_csv(feats_csv_path)
        
        # 对每个WSI分组排序，每个组为一个region，每个WSI文件名以a1_b1_a2_b2结尾，a1_b1是Region在整个WSI的位置，a2_b2是patch在Region的位置
        df['a1'] = df['img_path'].apply(lambda x: int(x.split('/')[-1].split('_')[0]))
        df['b1'] = df['img_path'].apply(lambda x: int(x.split('/')[-1].split('_')[1]))
        # 同样提取a2、b2为数值列
        df['a2'] = df['img_path'].apply(lambda x: int(x.split('/')[-1].split('_')[2]))
        df['b2'] = df['img_path'].apply(lambda x: int(x.split('/')[-1].split('_')[3].split('.')[0]))
        # 创建一个排序和分组用的a1_b1列
        df['a1_b1'] = df.apply(lambda x: f"{x['a1']}_{x['b1']}", axis=1)
        # 根据a1、b1、a2、b2进行排序
        sorted_df = df.sort_values(['a1', 'b1', 'a2', 'b2'])
        # 按照a1_b1分组
        grouped = sorted_df.groupby('a1_b1')
        # 初始化一个列表来存储每个分组的DataFrame
        group_wsi_feats = [] # [Region_1, Region_2,..., Region_n]

        for img_path, group in grouped:
            # 对每个分组进行操作，这里我们只是简单地收集了每个分组的DataFrame
            del group['a1'], group['b1'], group['a2'], group['b2'], group['a1_b1']
            group = group.iloc[:, :-1]
            group = group.to_numpy()
            group = torch.tensor(np.array(group)).float()
            group_wsi_feats.append(group)

        # RNA表达
        rna_feats = self.rna[csv_file_df.iloc[0][:15]] # TCGA-4B-A93V-01Z-00-DX1
        group_rna_feats = []
        for function_feature in rna_feats:
            # num_pathway.append(len(num_pathway_in_function))
            function_feat = list(chain.from_iterable(function_feature))
            function_feat = torch.tensor(function_feat, dtype=torch.float)
            group_rna_feats.append(function_feat)
               
        #DNA甲基化
        dna_feats = self.dna[csv_file_df.iloc[0][:15]] # TCGA-4B-A93V-01Z-00-DX1
        group_dna_feats = []
        for function_feature in dna_feats:
            # num_pathway.append(len(num_pathway_in_function))
            function_feat = list(chain.from_iterable(function_feature))
            function_feat = torch.tensor(function_feat, dtype=torch.float)
            group_dna_feats.append(function_feat)
        
        survival_time_label = self.status_info.loc[csv_file_df.iloc[0][:15], 'label']
        survival_time = self.status_info.loc[csv_file_df.iloc[0][:15], 'OS_time']
        survival_status = self.status_info.loc[csv_file_df.iloc[0][:15], 'status']
        subtype = self.subtype_info.loc[csv_file_df.iloc[0][:15], 'subtype']
        stage = self.stage_info.loc[csv_file_df.iloc[0][:15], 'stage']
        
        subtype,stage, survival_time_label, survival_time, survival_status = torch.tensor(subtype).type(torch.LongTensor),torch.tensor(stage).type(torch.LongTensor),torch.tensor(survival_time_label).type(torch.LongTensor), torch.tensor(survival_time).type(torch.LongTensor), torch.tensor(survival_status).type(torch.FloatTensor)

        
        return subtype,stage, survival_time_label, survival_time, survival_status, group_wsi_feats, group_rna_feats, group_dna_feats
    

    def dropout_patches(self,feats, p):
        idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
        sampled_feats = np.take(feats, idx, axis=0)
        pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
        pad_feats = np.take(sampled_feats, pad_idx, axis=0)
        sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
        return sampled_feats
    
    def __getitem__(self, idx):
        subtype,stage, survival_time_label, survival_time, survival_status, group_wsi_feats, group_rna_feats, group_dna_feats = self.get_bag_feats(self.train_path.iloc[idx], self.args)
        return  subtype,stage, survival_time_label, survival_time, survival_status, group_wsi_feats, group_rna_feats, group_dna_feats

    def __len__(self):
        return len(self.train_path)


def train(train_loader, milnet, criterion, criterion2,criterion3, optimizer, args, log_path):
    milnet.train()
    total_loss = 0

    print('\n')
    all_risk_scores = np.zeros((len(train_loader)))
    all_censorships = np.zeros((len(train_loader)))
    all_event_times = np.zeros((len(train_loader)))
    all_subtype_tru = np.zeros((len(train_loader)))
    all_subtype_pre = np.zeros((len(train_loader)))
    all_stage_tru = np.zeros((len(train_loader)))
    all_stage_pre = np.zeros((len(train_loader)))
    
    for i,(subtype, stage, survival_time_label, survival_time, survival_status, bag_wsi_feats, bag_rna_feats, bag_dna_feats) in enumerate(train_loader):   
        bag_subtype = subtype.cuda()
        bag_stage = stage.cuda()
        bag_label = survival_time_label.cuda()
        bag_time = survival_time.cuda()
        bag_label_status = survival_status.cuda()
        
        bag_wsi_feats = [group_wsi_feat.cuda().view(-1, args.feats_size) for group_wsi_feat in bag_wsi_feats]
        bag_rna_feats = [group_rna_feats.cuda().unsqueeze(0) for group_rna_feats in bag_rna_feats]
        bag_dna_feats = [group_dna_feats.cuda().unsqueeze(0) for group_dna_feats in bag_dna_feats]
        
        optimizer.zero_grad()

        output = milnet(bag_wsi_feats, bag_rna_feats, bag_dna_feats)
        hazards, S, subtype_pre,stage_pre =  output['logits'], output['S'], output['subtype'], output['stage']

        loss1 =  criterion(hazards, bag_label, bag_label_status)
        loss2 =  criterion2(subtype_pre, bag_subtype)
        loss3 =  criterion3(stage_pre, bag_stage)
        loss = loss1+loss2+loss3

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        all_risk_scores[i] = risk[0]
        all_censorships[i] = bag_label_status.item()
        all_event_times[i] = bag_time.item()
        all_subtype_pre[i] = torch.argmax(subtype_pre)
        all_subtype_tru[i] = bag_subtype.item()
        all_stage_pre[i] = torch.argmax(stage_pre)
        all_stage_tru[i] = bag_stage.item()

  
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        

        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f ' % (i, len(train_loader), loss.item()))
    
    train_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    train_acc_subtype = accuracy_score(all_subtype_tru, all_subtype_pre, normalize=True, sample_weight=None)
    train_acc_stage = accuracy_score(all_stage_tru, all_stage_pre, normalize=True, sample_weight=None)

    return total_loss / len(train_loader), train_c_index, train_acc_subtype,train_acc_stage


def test(test_loader, milnet, criterion,criterion2,criterion3, optimizer, args, log_path, epoch):
    milnet.eval()
    total_loss = 0
    all_risk_scores = np.zeros((len(test_loader)))
    all_censorships = np.zeros((len(test_loader)))
    all_event_times = np.zeros((len(test_loader)))
    all_subtype_tru = np.zeros((len(test_loader)))
    all_subtype_pre = np.zeros((len(test_loader)))
    all_stage_tru = np.zeros((len(test_loader)))
    all_stage_pre = np.zeros((len(test_loader)))

    with torch.no_grad():
        for i,(subtype,stage, survival_time_label, survival_time, survival_status, bag_wsi_feats, bag_rna_feats, bag_dna_feats) in enumerate(test_loader):
            # print(num_pathway_in_function)
            bag_subtype = subtype.cuda()
            bag_stage = stage.cuda()    
            bag_label = survival_time_label.cuda()
            bag_time = survival_time.cuda()
            bag_label_status = survival_status.cuda()

            bag_wsi_feats = [group_wsi_feat.cuda().view(-1, args.feats_size) for group_wsi_feat in bag_wsi_feats]
            bag_rna_feats = [group_rna_feats.cuda().unsqueeze(0) for group_rna_feats in bag_rna_feats]
            bag_dna_feats = [group_dna_feats.cuda().unsqueeze(0) for group_dna_feats in bag_dna_feats]
            
            output = milnet(bag_wsi_feats, bag_rna_feats, bag_dna_feats)
            hazards, S, subtype_pre,stage_pre =  output['logits'], output['S'], output['subtype'], output['stage']
            
            # loss1 =  criterion(hazards, bag_label, bag_label_status)
            # print(bag_subtype,subtype)
            # loss2 =  criterion2(subtype, bag_subtype)
            # loss = loss1+loss2
            
            # total_loss = total_loss + loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_loader), loss.item()))
            sys.stdout.write('\r Testing bag [%d/%d]' % (i, len(test_loader)))

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[i] = risk[0]
            all_censorships[i] = bag_label_status.item()
            all_event_times[i] = bag_time.item()
            all_subtype_pre[i] = torch.argmax(subtype_pre)
            all_subtype_tru[i] = bag_subtype.item()
            all_stage_pre[i] = torch.argmax(stage_pre)
            all_stage_tru[i] = bag_stage.item()

    test_c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    test_cr = classification_report(all_subtype_tru, all_subtype_pre, output_dict=True)
    test_st = classification_report(all_stage_tru, all_stage_pre, output_dict=True)
    
    return total_loss / len(test_loader), test_c_index, test_cr,test_st


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train MambaMIL')
    parser.add_argument('--num_classes', default=4, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_classes_subtype', default=5, type=int, help='Number of subtype classes [2]')
    parser.add_argument('--num_classes_stage', default=4, type=int, help='Number of stage classes [2]')
    parser.add_argument('--WSI_bags_csv', default='./example/slide_info_BLCA.csv', type=str, help='path of the bag csv')
    parser.add_argument('--status_bags_csv', default='./example/status_info_discretization_BLCA1.csv', type=str, help='path of the status csv')
    parser.add_argument('--subtype_csv', default='./example/subtype/BLCA_subtype.csv', type=str, help='path of the subtype csv')
    parser.add_argument('--RNASeq_csv', default='./example/rna_BLCA.csv', type=str, help='path of the RNAseq csv')
    parser.add_argument('--DNASeq_csv', default='./example/dna_BLCA.pkl', type=str, help='path of the DNAseq csv')
    parser.add_argument('--stage_csv', default='./example/BLCA_stage1.csv', type=str, help='path of the stage csv')
    parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=30, type=int, help='Number of total training epochs')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--gpu', type=str, default= '0')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-BLCA-3task', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='mambamil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=0, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    parser.add_argument('--agg', default='no', type=str,help='which agg')
    parser.add_argument('--c_path', nargs='+', default=None, type=str,help='directory to confounders')
    # parser.add_argument('--dir', type=str,help='directory to save logs')

    
    args = parser.parse_args()
    # assert args.model == 'mambamil' 

    # logger
    arg_dict = vars(args)
    dict_json = json.dumps(arg_dict)
    save_path = os.path.join('baseline', datetime.date.today().strftime("%m%d%Y"), str(args.dataset)+'_'+str(args.model)+'_'+str(args.agg )+'_fulltune')

    run = len(glob.glob(os.path.join(save_path, '*')))
    save_path = os.path.join(save_path, str(run))
    os.makedirs(save_path, exist_ok=True)
    save_file = save_path + '/config.json'
    with open(save_file,'w+') as f:
        f.write(dict_json)
    log_path = save_path + '/log.txt'

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    '''
    model 
    1. set require_grad    
    2. choose model and set the trainable params 
    3. load init
    '''
    
    
    # 读取RNAseq信息
    data_RNA = pd.read_csv(args.RNASeq_csv, index_col=0)
    data_RNA = np.log2(data_RNA + 1)
    data_DNA = pd.read_pickle(args.DNASeq_csv)
    data_DNA = data_DNA.apply(pd.to_numeric, errors='coerce').fillna(0)

    KEGG352_filename = './KEGG_RNA_DNA.txt'   # For DNA methylation
    # 使用字典来存储结果

    pathway_dict_RNA = {}
    pathway_dict_DNA = {}
    with open(KEGG352_filename, 'r') as file:
        for line in file:
            # 移除字符串末尾的换行符，并以制表符分割
            parts = line.strip().split('\t')
            # 第一部分是ID，其余的是基因名
            gene_id = parts[0]  # First element is the gene ID
            pathway_category = parts[1]  # Second element is the pathway category
            gene_names_combined = parts[2]  # The third column contains the gene names
            gene_names_combined1 = parts[3]
            gene_names = gene_names_combined.split('/')  # Split gene names by "/"
            gene_names1 = gene_names_combined1.split('/')
            
            # Store the pathway category and gene names together in a list
            pathway_dict_DNA[gene_id] = [pathway_category] + gene_names
            pathway_dict_RNA[gene_id] = [pathway_category] + gene_names1

    # 分别构建 RNA 和 DNA 的 KEGG 信息结构
    KEGG_info_RNA = dict()
    KEGG_info_DNA = dict()

    for k in pathway_dict_RNA:
        KEGG_info_RNA[pathway_dict_RNA[k][0]] = dict()
    for k in pathway_dict_RNA:
        KEGG_info_RNA[pathway_dict_RNA[k][0]][k] = pathway_dict_RNA[k][1:]

    for k in pathway_dict_DNA:
        KEGG_info_DNA[pathway_dict_DNA[k][0]] = dict()
    for k in pathway_dict_DNA:
        KEGG_info_DNA[pathway_dict_DNA[k][0]][k] = pathway_dict_DNA[k][1:]

    time1 = time.time()
    # 预处理KEGG_info以便重用
    # 分别处理 RNA 和 DNA 的 KEGG 信息以便重用
    processed_KEGG_info_RNA = {
        function_name: {
            pathway_name: list(set(gene_list) & set(data_RNA.columns))
            for pathway_name, gene_list in pathway_info.items()
        }
        for function_name, pathway_info in KEGG_info_RNA.items()
    }

    processed_KEGG_info_DNA = {
        function_name: {
            pathway_name: list(set(gene_list) & set(data_DNA.columns))
            for pathway_name, gene_list in pathway_info.items()
        }
        for function_name, pathway_info in KEGG_info_DNA.items()
    }
    
    # 统计基因数目
    num_pathway_in_function_RNA = []
    for function_name, pathway_info in KEGG_info_RNA.items():
        p_num = []
        for pathway_name, gene_list in pathway_info.items():
            p = list(set(gene_list) & set(data_RNA.columns))
            p_num.append(len(p))
        num_pathway_in_function_RNA.append(p_num)

    num_pathway_in_function_DNA = []
    for function_name, pathway_info in KEGG_info_DNA.items():
        p_num = []
        for pathway_name, gene_list in pathway_info.items():
            p = list(set(gene_list) & set(data_DNA.columns))
            p_num.append(len(p))
        num_pathway_in_function_DNA.append(p_num)

    # 提取通路分层信息
    RNA_stratification = {}
    DNA_stratification = {}

    for id in data_RNA.index:
        function_features_RNA = []
        function_features_DNA = []
        
        # 处理 RNA 的 KEGG 信息
        for function_name, pathway_info in processed_KEGG_info_RNA.items():
            pathway_features_RNA = []
            for gene_list in pathway_info.values():
                if gene_list:  # 确保gene_list不为空
                    pathway_features_RNA.append(list(data_RNA.loc[id, gene_list].values))
            function_features_RNA.append(pathway_features_RNA)
        
        # 处理 DNA 的 KEGG 信息
        for function_name, pathway_info in processed_KEGG_info_DNA.items():
            pathway_features_DNA = []
            for gene_list in pathway_info.values():
                if gene_list:  # 确保gene_list不为空
                    pathway_features_DNA.append(list(data_DNA.loc[id, gene_list].values))
            function_features_DNA.append(pathway_features_DNA)
        
        RNA_stratification[id] = function_features_RNA
        DNA_stratification[id] = function_features_DNA
    print('提取通路分层信息完成！', time.time()-time1)
    

    wsi_bags_path = pd.read_csv(args.WSI_bags_csv, index_col=0)
    status_info = pd.read_csv(args.status_bags_csv, index_col=0)
    subtype_info = pd.read_csv(args.subtype_csv, index_col=0)
    stage_info = pd.read_csv(args.stage_csv, index_col=0)
    # 找到两个 DataFrame 中行名（索引）相同的行
    common_index = status_info.index.intersection(subtype_info.index)
    common_index = common_index.intersection(stage_info.index)
    
    wsi_bags_path['data_sliced'] = wsi_bags_path['slide'].str.slice(0, 15)
    common_index2 = pd.Index(wsi_bags_path['data_sliced']).intersection(common_index)

    # 根据相同的行名选取行
    wsi_bags_path = wsi_bags_path[wsi_bags_path['data_sliced'].isin(common_index)]
    
    status_info = status_info.loc[common_index2]
    subtype_info = subtype_info.loc[common_index2]
    stage_info = stage_info.loc[common_index2]
    print('wsi_bags_path')
    print(wsi_bags_path)

    # 初始化KFold，n_splits=5表示五折交叉验证，shuffle=True表示在分层之前会对数据进行洗牌
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    k_count = 0
    k_train_c_index = []
    k_test_c_index = []
    df = pd.DataFrame(columns=['k_fold', 'epoch', 'train_loss_surv', 'train_c_index','train_acc_subtype', 'train_acc_stage', 
                               'test_loss_surv', 'test_c_index', 'test_accuracy_subtype','test_precision_subtype','test_recall_subtype', 'test_f1-score_subtype',
                               'test_accuracy_stage','test_precision_stage','test_recall_stage', 'test_f1-score_stage'])

    # 使用KFold分割数据
    # wsi_bags_path.index 可以获取到DataFrame的索引列表，这里我们基于索引来进行分割
    for train_index, test_index in kf.split(wsi_bags_path):
        print('*'*30)
        print('k-fold: ', k_count)
        k_count += 1
        
        # 根据索引获取训练和测试集
        train_path, test_path = wsi_bags_path.iloc[train_index], wsi_bags_path.iloc[test_index]
                   
        trainset =  BagDataset(train_path, RNA_stratification, DNA_stratification,  status_info, subtype_info, stage_info, args)
        train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)
        testset =  BagDataset(test_path, RNA_stratification, DNA_stratification, status_info, subtype_info, stage_info, args)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

        import Models.MambaMIL.net_3task as mil
        milnet = mil.MambaMIL(input_size=args.feats_size, num_classes=args.num_classes, num_classes_subtype=args.num_classes_subtype,
                              num_classes_stage=args.num_classes_stage,
                              num_pathway_in_function_RNA=num_pathway_in_function_RNA,
                              num_pathway_in_function_DNA=num_pathway_in_function_DNA,
                              embed_dim=512, depth=1, rms_norm=True, residual_in_fp32=True, final_pool_type='mean').cuda()
        

        criterion = NLLSurvLoss()
        criterion2 = torch.nn.CrossEntropyLoss()
        criterion3 = torch.nn.CrossEntropyLoss()
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        original_params = []
        for pname, p in milnet.named_parameters():
            original_params += [p]
        
        optimizer = RAdam(original_params, lr=args.lr, weight_decay=args.weight_decay)

        best_test_c_index = 0
        best_train_c_index = 0
        best_train_acc_subtype = 0
        print("start traing!")

        for epoch in range(args.num_epochs):
            start_time = time.time()
            
            
            train_loss_surv, train_c_index, train_acc_subtype, train_acc_stage = train(train_loader, milnet, criterion,criterion2, criterion3, optimizer, args, log_path) # iterate all bags
            print('\n')
            print('Epoch: {}, train_loss_surv: {:.4f}, train_c_index: {:.4f}, train_acc_subtype: {:.4f}, train_acc_stage: {:.4f}'.format(epoch, train_loss_surv, train_c_index, train_acc_subtype, train_acc_stage))
            print('epoch time:{}'.format(time.time()- start_time))
            test_loss_surv, test_c_index, test_cr, test_st = test(test_loader, milnet, criterion,criterion2, criterion3, optimizer, args, log_path, epoch)
            print('Epoch: {}, test_loss_surv: {:.4f}, test_c_index: {:.4f}, acc_subtype: {:.4f}, precision_subtype: {:.4f}, recall_subtype: {:.4f}, f1-score_subtype: {:.4f},acc_stage: {:.4f}, precision_stage: {:.4f}, recall_stage: {:.4f}, f1-score_stage: {:.4f}'.
                  format(epoch, test_loss_surv, test_c_index, test_cr['accuracy'], test_cr['weighted avg']['precision'], test_cr['weighted avg']['recall'], test_cr['weighted avg']['f1-score'],test_st['accuracy'], test_st['weighted avg']['precision'], test_st['weighted avg']['recall'], test_st['weighted avg']['f1-score']))
            temp = {'k_fold': k_count, 'epoch': epoch,
                            'train_loss_surv': train_loss_surv, 'train_c_index': train_c_index, 'train_acc_subtype': train_acc_subtype,'train_acc_stage': train_acc_stage,
                            'test_loss_surv': test_loss_surv, 'test_c_index': test_c_index, 'test_accuracy_subtype': test_cr['accuracy'],
                            'test_precision_subtype': test_cr['weighted avg']['precision'], 'test_recall_subtype': test_cr['weighted avg']['recall'], 'test_f1-score_subtype': test_cr['weighted avg']['f1-score'],
                            'test_accuracy_stage': test_st['accuracy'],'test_precision_stage': test_st['weighted avg']['precision'], 'test_recall_stage': test_st['weighted avg']['recall'], 'test_f1-score_stage': test_st['weighted avg']['f1-score']}
            
            df = pd.concat([df, pd.DataFrame([temp])], ignore_index=True)
            if test_c_index >= best_test_c_index:
                best_test_c_index = test_c_index
                save_name = os.path.join(save_path, str(k_count)+'_'+str(epoch)+'.pth')
                torch.save(milnet.state_dict(), save_name)
                print('Best model saved at: ' + save_name +'\n')
                with open(log_path,'a+') as log_txt:
                    info = 'Best model saved at: ' + save_name +'\n'
                    log_txt.write(info)
            if epoch == args.num_epochs-1:
                save_name = os.path.join(save_path, str(k_count)+'_'+str(epoch)+'_'+'last.pth')
                torch.save(milnet.state_dict(), save_name)
        
        k_train_c_index.append(best_train_c_index)
        k_test_c_index.append(best_test_c_index)
        with open(log_path,'a+') as log_txt:
            log_txt.write('k_train_c_index' + '\n')
            log_txt.write(str(k_train_c_index)+ '\n')
            log_txt.write('k_test_c_index' + '\n')
            log_txt.write(str(k_test_c_index))
        log_txt.close()
        
        print('k_train_c_index')
        print(k_train_c_index)
        print(np.mean(k_train_c_index))
        print(np.std(k_train_c_index))
        print('k_test_c_index')
        print(k_test_c_index)
        print(np.mean(k_test_c_index))
        print(np.std(k_test_c_index))
        df.to_csv(os.path.join(save_path, str(run))+"result.txt",sep='\t', index=True, header=True)

if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=7 nohup python Train_MambaMIL_Survival_CUP_BLCA33.py --agg no --feats_size 768 --model mambamil > CancerMul_BLCA1216.log 2>&1 &
    # CUDA_VISIBLE_DEVICES=0 python Train_MambaMIL_Survival_BLCA.py --agg no --feats_size 768 --model mambamil