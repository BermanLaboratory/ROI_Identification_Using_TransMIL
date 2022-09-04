import json
import yaml
import numpy as np
from addict import Dict
from torchvision import transforms
from sklearn.model_selection import train_test_split
#----> pytorch
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
from glob import glob


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg):

    """
        Returns the callbacks required according to the configuration file
    """

    Mycallbacks = []

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.mode == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = '/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/models',
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
    return Mycallbacks


def data_split_random(seed,indices,dataset_size,validation_split = 0.1,shuffle_dataset = False):

    """
        Inputs:
            seed: Seed to produce random numbers from
            indices: The List of Indices of dataset to split
            dataset_size: The Size of dataset
            validation_split: val split ratio
            shuffle_dataset: To Shuffle datset or not
        Returns:
            Indices split based on the ratio (The split is random)
                train_indices,val_indices
    
    """

    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    return train_indices,val_indices

def data_split_balanced(seed,indices,labels_list,validation_split = 0.1):

    """
        Inputs:
            seed: seed to generate random numbers.
            indices: The List of Indices of dataset to split
            labels_list: List of labels for each image patch
            validation_slit: split ratio required.
        Returns: 
            Indices split based on the ratio(The Dataset Split is balanced)
    """

    train_indices, val_indices, train_labels, val_labels = train_test_split(
                                            indices,
                                            labels_list,
                                            stratify=labels_list,
                                            test_size=validation_split,
                                            random_state=seed
                                            )

    return train_indices,val_indices,train_labels,val_labels

def data_sampler_dict(split_type,indices,random_seed,len_dataset,patch_labels_list,train_split = 0.8 ,validation_split=0.1,test_split = 0.1,data_shuffle = True):
    
    """
        Inputs:
            split_type: can be random or balanced.
            indices: Indices of the dataset
            patch_labels_list: list of labels of each image patch
        Returns:
            train,val and test sampler dictionary is returned based on the type and ratio of split chosen.
    """

    ratio_remaining = 1.0 - validation_split
    ratio_test_adjusted = test_split / ratio_remaining


    if split_type == 'random':
        train_indices_remaining,val_indices = data_split_random(random_seed,indices,len_dataset,validation_split,data_shuffle)
        train_indices,test_indices = data_split_random(random_seed,train_indices_remaining,len(train_indices_remaining),ratio_test_adjusted,data_shuffle)
        
    else:
        train_indices_remaining,val_indices,train_labels_remaining,val_labels = data_split_balanced(random_seed,indices,patch_labels_list,validation_split)
        train_indices,test_indices,_,_ = data_split_balanced(random_seed,train_indices_remaining,train_labels_remaining,ratio_test_adjusted)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    sampler = {'train':train_sampler,'val':valid_sampler,'test':test_sampler}

    return sampler


def read_yaml(fpath=None):
    """
        Args:
            fpath: file path to yaml file
        Returns config dictionary
    """
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


def per_slide_features(dataset_feature_file_path):

    """
        Inputs: 
            dataset_feature_file: Path to the feature file
        Ouputs:
            A dictionary with per slide features. Keys are the sample ids
    """
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

    """   
        This function is used when utilizing densenet extracted features.
         Inputs: 
            dataset_feature_file: Path to the feature file
            selected_patches: List of selected image Patches
        Ouputs:
            A dictionary with per slide features. Keys are the sample ids
    
    """

    file_dict = pd.read_pickle(dataset_features)
    df = pd.DataFrame(file_dict.items(),columns=['Name','Feature_Value'])
    slide_data = {}
    patch_list = df['Name'].to_list()
    feature_list = df['Feature_Value'].to_list()
    
    for i in range(len(select_patches)):
        select_patches[i] = select_patches[i].split('/')[-1]

   
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

    """
        Returns the lables dictionary from the the data csv file
    """

    labels_df = pd.read_csv(csv_file_path)
    labels_df = labels_df.dropna()
    labels_df.astype(int)
    labels_dict = {}
    files_list = labels_df['Sample ID'].to_list()
    grade = labels_df['Sample Grade'].to_list()

    for i in range(len(files_list)):
        labels_dict[int(files_list[i])] = int(grade[i])
    
    return labels_dict