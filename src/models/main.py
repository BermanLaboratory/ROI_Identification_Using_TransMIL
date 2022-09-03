import argparse
from pathlib import Path
import numpy as np
import glob

from data.features_interface import FeaturesInterface
from models.architechture.TransMIL_interface import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

# ----> load Callback

from utils.utils import *
import os


from data.features import *
from torch.utils.data.sampler import SubsetRandomSampler

import yaml
from addict import Dict



#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/config/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = [2])
    parser.add_argument('--fold', default = 0)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    
    wandb_logger = pl_loggers.WandbLogger(name=cfg.General.run_name,project=cfg.General.project_name)
    labels_dict = dataset_labels(cfg.Data.label_dir)
    
 

    if (cfg.Data.data_type == 'pickle') :

        #  print('Hello')
         slide_data = per_slide_features(cfg.Data.data_file)
         dataset = Features(slide_data,labels_dict)

    
    else:

        json_folder = cfg.Data.data_folder
        slide_list = []
        with os.scandir(json_folder) as files:
            for file in files:
                slide_list.append(file.path)

        dataset = FeatureJson(slide_list,labels_dict)
       
   
    validation_split = .1
    test_split = 0.1
    shuffle_dataset = True
    random_seed= 13
    slide_list = []

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]


    test_indices = [183]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    sampler = {'train':train_sampler,'val':valid_sampler,'test':test_sampler}

    #---->Data_Interface 
    DataInterface_dict = {'slide_feature_data':slide_data,
                'labels_dict': labels_dict,
                'sampler': sampler,
                }

    dm = FeaturesInterface(**DataInterface_dict)
    
    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)

    callbacks = load_callbacks(cfg)

    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs= 10,
        gpus=cfg.General.gpus,  
        precision=cfg.General.precision,  
        check_val_every_n_epoch=2,
    )

    #---->train or test
    if cfg.General.mode == 'train':
        # print('Starting Training')
        trainer.fit(model = model, datamodule = dm)
    else:
    
        new_model = model.load_from_checkpoint(cfg.General.weights_file_path,cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml('/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/config/bermanlab.yaml')

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.mode = 'train'
    cfg.Data.fold = args.fold

    #---->main
    main(cfg)
 