import sys
sys.path.append(sys.path[0].replace('KAD/data', 'KAD'))

import csv
import json
import logging
import os
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import torch

from data.randaugment import RandomAugment
from io import BytesIO
from configs.default import run_cluster, normalize


class MIMIC_Dataset(Dataset):
    def __init__(self, json_path, csv_path, sty_path, image_res,args):
        self.json_info = json.load(open(json_path,'r'))
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])#40 class for fine-grained query list
        sty_info = pd.read_csv(sty_path)
        self.sty_dict_info = self.csv_to_dict(sty_info)

        if args.colourjitter:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                transforms.RandomGrayscale(),

                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])

        else:
            self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(image_res,scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])    

    
    def csv_to_dict(self,sty_info):
        tui_list = sty_info.iloc[:,0]
        sty_list = sty_info.iloc[:,1]
        sty_dict = defaultdict(list)
        for idx in range(len(tui_list)):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):

        img_path = self.img_path_list[index]
        
        class_label = self.class_list[index] 

        
        entities = self.json_info[index]['entities']
        captions = self.json_info[index]['caption']


        if len(entities) != 0:
            caption_list = ''
            entity_details = ''
            for entity in entities:
                sub_caption = entity['caption']
                sub_entities = entity['entity']
                sub_entity_details = ''
                for sub_entity in sub_entities:
                    try:
                        sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
                    except:
                        sub_entity_details += ' [ENT] ' + sub_entity['Entity']  
                entity_details = entity_details + sub_entity_details + ' [SEP] '
                caption_list = caption_list + sub_caption + ' [SEP] '
        else:
            caption_list = ''
            entity_details = ''
            for sub_caption in captions:
                caption_list = caption_list + sub_caption + ' [SEP] '
            entity_details = caption_list
        
        # img = open_jpg(img_path).convert('RGB')  
        img = Image.open(img_path).convert('RGB') 
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label,
            "caption": caption_list,
            "entity": entity_details
            }
    

class Mergetrain_Dataset(Dataset):
    def __init__(self, json_path, csv_path, sty_path,image_res,args):
        self.json_info = json.load(open(json_path,'r'))
        data_info = pd.read_csv(csv_path)
        if hasattr(args, 'use_dataset'):
            if args.use_dataset == 'all':
                pass
            else:
                dataset_list = args.use_dataset.split('-')
                total_dataset_list = ['mimic', 'cxr14', 'chex', 'vindr']
                removed_dataset_list = list(set(total_dataset_list) - set(dataset_list))
                dataset_dict = {'mimic':0, 'cxr14':1, 'chex':2, 'vindr':3}
                for dataset_name in removed_dataset_list:
                    dataset_idx = dataset_dict[dataset_name]
                    data_info = data_info[data_info.iloc[:,1] != dataset_idx]
                    
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,2:])#60 class for fine-grained query list
        self.label_dataset_list = np.asarray(data_info.iloc[:,1])

        sty_info = pd.read_csv(sty_path)
        
                
        self.sty_dict_info = self.csv_to_dict(sty_info)

        
        self.transform = transforms.Compose([    
            transforms.RandomResizedCrop(image_res,scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToPILImage(),    
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            normalize,
        ])

    
    def csv_to_dict(self,sty_info):
        tui_list = sty_info.iloc[:,0]
        sty_list = sty_info.iloc[:,1]
        sty_dict = defaultdict(list)
        for idx in range(len(tui_list)):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):

        if self.label_dataset_list[index] == 0:

            img_path = self.img_path_list[index]

            class_label = self.class_list[index] 


        
            entities = self.json_info[index]['entities']
            captions = self.json_info[index]['caption']


            if len(entities) != 0:
                caption_list = ''
                entity_details = ''
                for entity in entities:
                    sub_caption = entity['caption']
                    sub_entities = entity['entity']#搞错了 还不是list
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        try:
                            sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
                        except:
                            sub_entity_details += ' [ENT] ' + sub_entity['Entity']  
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
                    caption_list = caption_list + sub_caption + ' [SEP] '
            else:
                caption_list = ''
                entity_details = ''
                for sub_caption in captions:
                    caption_list = caption_list + sub_caption + ' [SEP] '
                entity_details = caption_list
        

        
        else:
            img_path = self.img_path_list[index]
            class_label = self.class_list[index] 
            caption_list = ''
            head = ['normal', 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration',
            'fibrosis',	'No Finding', 'Pleural Other', 'Support Devices', 'Aortic enlargement',
            'Clavicle fracture', 'Enlarged PA', 'ILD', 'Lung cavity', 'Lung cyst', 'Mediastinal shift',	
            'Nodule/Mass', 'Pulmonary fibrosis', 'Rib fracture', 'Other lesion', 'COPD', 'Lung tumor', 'Tuberculosis',
            'Other diseases']
            index_positive = np.where(class_label == 1)
            entity =  np.array(head)[index_positive]
            entity_details = ''
            for sub_entity in entity:
                entity_details = entity_details + sub_entity + ' [SEP] '

        img = Image.open(img_path).convert('RGB') 
        image = self.transform(img)
        label_dataset = self.label_dataset_list[index]

        return {
            "image": image,
            "label": class_label,
            "label_dataset": label_dataset,
            "caption": caption_list,
            "entity": entity_details
            }

 

class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])

        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,[13,7,11,10,15]])

        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


class FT_SixthHospital_Dataset(Dataset):

    def __init__(self, csv_path='train.csv', image_res=512):

        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.max_bit_csv = pd.read_csv('max.csv')
        self.max_bit_list = []
        for img_path in self.img_path_list:
            max_bit = self.max_bit_csv[self.max_bit_csv['img_path'] == img_path].iloc[0]['bit_depth']
            self.max_bit_list.append(int(max_bit))
        
        # self.max_bit_list = np.asarray(max_bit_csv.iloc[:,3])
        self.image_res = image_res
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        
        img = Image.open(img_path).convert('I')
        

        img = np.array(img, dtype=np.float32) 
        img = img / (2**self.max_bit_list[index])
        img = torch.tensor(img, dtype=torch.float32)
        

        img = torch.stack([img, img, img], dim=0)
        
        image = self.transform(img)


        head = ['pneumothorax', 'aortic enlargement', 'rib fracture', 'mass', 'atelectasis', 
 'clavicle fracture', 'collapse', 'mediastinal shift', 'pulmonary fibrosis', 
 'pneumonia', 'blunt', 'pleural effusion', 'cardiomegaly', 'edema', 'fibrosis', 
 'lung tumor', 'copd', 'fracture', 'nodule', 'lung cavity']

        index_positive = np.where(class_label == 1)
        entity =  np.array(head)[index_positive]
        entity_details = ''
        for sub_entity in entity:
            entity_details = entity_details + sub_entity + ' [SEP] '
        
        return {
            "entity": entity_details,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


class TEST_SixthHospital_Dataset(Dataset):
    def __init__(self, csv_path='1000_example.csv', image_res=512):

        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.max_bit_csv = pd.read_csv('max.csv')
        self.max_bit_list = []
        for img_path in self.img_path_list:
            max_bit = self.max_bit_csv[self.max_bit_csv['img_path'] == img_path].iloc[0]['bit_depth']
            self.max_bit_list.append(int(max_bit))
        
        self.image_res = image_res
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        
        img = Image.open(img_path).convert('I')
        
        img = np.array(img, dtype=np.float32) 

        img = img / (2**self.max_bit_list[index])

        img = torch.tensor(img, dtype=torch.float32)

        img = torch.stack([img, img, img], dim=0)
        
        image = self.transform(img)
        
        
        
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


    