import sys
sys.path.append(sys.path[0].replace('KAD/data', 'KAD'))

import os
import sys
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import pydicom
from skimage import exposure
from configs.default import run_cluster, normalize
import torch






class MIMIC_test_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.class_text = ['Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum', 'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion', 'No Finding', 'Pleural Other', 'Support Devices']
        self.class_list = self.class_list[:, (20, 22, 23, 25, 31, 32, 34, 35, 36, 38, 40, 41, 9, 13, 16)]
        
        self.transform = transforms.Compose([                        
                transforms.Resize(image_res, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
        
    def __getitem__(self, index):
        class_label = self.class_list[index]
        img_path = self.img_path_list[index]
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }

    def __len__(self):
        return len(self.img_path_list)

class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
                
        no_finding_label = np.zeros((len(data_info), 1))
        no_finding_label[data_info.iloc[:,2] == 'No Finding'] = 1
        
        self.class_list = np.asarray(data_info.iloc[:,3:])

        self.class_list = np.concatenate((self.class_list, no_finding_label), axis=1)
        
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
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)



class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index]
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class Padchest_Dataset(Dataset):
    def __init__(self, csv_path, image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
        
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):

        img_path = self.img_path_list[index]
            
        class_label = self.class_list[index] 
        
        img = Image.open(img_path).convert('RGB') 

        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class SixthHospital_Dataset(Dataset):
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


class CXR_LT_Dataset(Dataset):
    def __init__(self, csv_path='cxr-lt.csv',image_res=512):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
    
   

   
class Vindr_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
        
        
        
        
class Merge_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
    


class MIMIC_development_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,5])
        self.class_list = np.asarray(data_info.iloc[:,6:])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
    
    

    
class SIIMACR_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

    
class Shenzhen_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,1])
        self.transform = transforms.Compose([                        
                transforms.Resize([image_res,image_res], interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class Openi_Dataset(Dataset):
    def __init__(self, csv_path,image_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,3])
        self.class_list = np.asarray(data_info.iloc[:,4:])

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
            "img_path": img_path,
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)


