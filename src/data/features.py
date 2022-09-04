import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import pandas as pd
import json
import numpy as np
# Move the Train Test Split of the Code here

class Features(Dataset):

    def __init__(self,slide_feature_data,labels_dict):
        """
        Args: 
            slide_feature_data: per slide feature data in the form of list
            labels_dict: labels dict with keys as sample ids
            
        """
        self.slide_data = slide_feature_data
        self.slides = list(slide_feature_data.keys())
        self.labels_dict = labels_dict
    
    def __len__(self):
        
        return len(self.slides)
    
    def __getitem__(self,index):
       
        file_name = self.slides[index]
        label = self.labels_dict[int(self.slides[index])]
        features = self.slide_data[self.slides[index]]
        features = torch.tensor(features)
        label = torch.tensor(label)
        
        return features,label,file_name


class FeatureJson(Dataset):

    def __init__(self,slide_data,labels_dict):

        """
        Args:
            slide_data : json slide paths
            labels_dict : slide level labels dict
        """
        self.slide_list = slide_data
        self.labels_dict = labels_dict


    def __len__(self):

        return len(self.slide_list)
    
    def __getitem__(self,index):

        file_path = self.slide_list[index]
        with open(file_path,"r") as file:
            feature_dictionary = json.loads(file.read())
    
        features = torch.tensor(list(feature_dictionary.values()))
        label = torch.tensor(self.labels_dict[int(((file_path.split('/')[-1]).split(' ')[1]).split('.')[0])])

        return features,label











