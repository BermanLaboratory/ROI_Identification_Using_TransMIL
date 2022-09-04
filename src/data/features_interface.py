from tkinter import E
import pytorch_lightning as pl
from data.features import FeatureJson, Features
import torch
from utils.utils import *
import os

class FeaturesInterface(pl.LightningDataModule):

    def __init__(self,cfg):

        super().__init__()

        self.cfg = cfg


    def setup(self,stage = None):

        self.labels_dict = dataset_labels(self.cfg.Data.label_dir)

        if (self.cfg.Data.data_type == 'single_file') :

        
         self.slide_data = per_slide_features(self.cfg.Data.data_file)
         self.dataset = Features(self.slide_data,self.labels_dict)
         labels_list = []
         slides = list(self.slide_data.keys())
         for slide in slides:
            labels_list  = labels_list + [self.labels_dict[int(slide)]]

        else:

            json_folder = self.cfg.Data.data_folder
            slide_list = []
            with os.scandir(json_folder) as files:
                for file in files:
                    slide_list.append(file.path)
            self.dataset = FeatureJson(slide_list,self.labels_dict)
            labels_list = []
            for slide in slide_list:
                labels_list = labels_list + [self.labels_dict[int(((slide.split('/')[-1]).split(' ')[1]).split('.')[0])]]


        self.sampler = data_sampler_dict(list(range(len(self.dataset))),self.cfg.General.seed,len(self.dataset),labels_list,self.cfg.Data.train_split,self.cfg.Data.validation_split,self.cfg.Data.test_split,self.cfg.Data.data_shuffle)
    
        if self.cfg.Data.split_test == False :
            slide_data_test = per_slide_features(self.cfg.Data.custom_test_data_file)
            self.dataset_test = Features(slide_data_test,self.labels_dict)
            test_sampler = SubsetRandomSampler(list(range(len(self.dataset_test))))

        self.sampler['test'] = test_sampler


    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.Data.train_dataloader.batch_size, 
                            sampler=self.sampler['train'],num_workers = 1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg.Data.val_dataloader.batch_size,
                                sampler=self.sampler['val'],num_workers = 1)


    def test_dataloader(self):

        if self.cfg.Data.split_test == True:
            return torch.utils.data.DataLoader(self.dataset,batch_size=self.cfg.Data.test_dataloader.batch_size,
                                        sampler = self.sampler['test'],num_workers = 1)
        else:
            return torch.utils.data.DataLoader(self.dataset_test,batch_size=self.cfg.Data.test_dataloader.batch_size,
                                        sampler = self.sampler['test'],num_workers = 1)
