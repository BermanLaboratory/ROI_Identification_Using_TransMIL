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
            data_path : path to input data
            transform : transformation function
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
        
        return features,label


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



def per_slide_features(dataset_feature_file_path):

    file_dict = pd.read_pickle(dataset_feature_file_path)
    df = pd.DataFrame(file_dict.items(),columns=['Name','Feature_Value'])
    slide_data = {}
    patch_list = df['Name'].to_list()
    feature_list = df['Feature_Value'].to_list()

    for i in range(len(patch_list)):

        patch = patch_list[i]
        sample_id = ((patch).split(' ')[1]).split('.')[0]
        keys = list(slide_data.keys())
        if (sample_id not in keys):
            slide_data[sample_id] = []
        
        feature_vector = feature_list[i]
        slide_data[sample_id] = slide_data[sample_id] + [feature_vector]
   

    return slide_data

def per_slide_selected(dataset_features,select_patches):

    file_dict = pd.read_pickle(dataset_features)
    df = pd.DataFrame(file_dict.items(),columns=['Name','Feature_Value'])
    slide_data = {}
    patch_list = df['Name'].to_list()
    feature_list = df['Feature_Value'].to_list()
    print(len(patch_list))

    for i in range(len(select_patches)):
        select_patches[i] = select_patches[i].split('/')[-1]

    print(len(patch_list))
    for i in range(len(patch_list)):

        patch = patch_list[i]
        if patch in select_patches:
            sample_id = ((patch).split(' ')[1]).split('.')[0]
            keys = list(slide_data.keys())
            if (sample_id not in keys):
                slide_data[sample_id] = []
            
            feature_vector = feature_list[i]
            slide_data[sample_id] = slide_data[sample_id] + [feature_vector]
        print(i)

    return slide_data


def dataset_labels(csv_file_path):

    labels_df = pd.read_csv(csv_file_path)
    labels_df = labels_df.dropna()
    labels_df.astype(int)
    labels_dict = {}
    files_list = labels_df['Sample ID'].to_list()
    grade = labels_df['Sample Grade'].to_list()

    for i in range(len(files_list)):
        labels_dict[int(files_list[i])] = int(grade[i])
    
    return labels_dict







