import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle

import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath
import numpy as np
import os

# class ToTensor(object):
#     def __call__(self, sample):
#         img = sample['input']
#         img = VF.to_tensor(img)
#         return {'input': img} 

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std),
    ]
)

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = trnsfrms_val(img)
        img_path = sample['img_path']
        # img = VF.to_tensor(img)
        return {'input': img, 'img_path': img_path} 

# trnsfrms_val = transforms.Compose(
#     [
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean = mean, std = std),
#         ToTensor()
#     ]
# )

class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image, path

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img).convert('RGB')
        sample = {'input': img, 'img_path': temp_path}
        
        if self.transform:
            sample = self.transform(sample)
        return sample


    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=256, shuffle=False, num_workers=16, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(bags_list, model, save_path=None):
    # i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        img_path_list = []
        print(bags_list[i])
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        print(len(csv_file_path))
        dataloader, bag_size = bag_dataset(csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                img_path_list.extend(batch['img_path'])
                patches = batch['input'].float().cuda() 
                feats = model(patches)
                # feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            df['img_path'] = img_path_list
            print(df)
            # print(os.path.join(save_path, bags_list[i].split(os.path.sep)[-1]+'.csv'))
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')            
            # os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            # df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')

if __name__ == '__main__':
    bags_path = './TCGA-BLCA/BLCA_Patch/'
    exist_bag = os.listdir('./TCGA-BLCA/BLCA_Patch_feat')
    exist_bag = [i[:-4] for i in exist_bag]
    print(len(exist_bag))
    print(len(os.listdir(bags_path)))
    bags_list = list(set(os.listdir(bags_path))-set(exist_bag))
    bags_list = [bags_path + i for i in bags_list]
    print(len(bags_list))
    # bags_list = [bags_path + i for i in os.listdir(bags_path)]
    # print('bags_list', bags_list)       
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load('./TransPath/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model = model.cuda ()
    model.eval()
    save_path = './TCGA-BLCA/BLCA_Patch_feat/'
    compute_feats(bags_list, model, save_path)    
    
    # nohup python -u TransPath/compute_feats_cying_BLCA.py > TransPath/compute_feats_cying_BLCA.txt 2>&1 &
    