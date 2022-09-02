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

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os

def load_callbacks(cfg):

    Mycallbacks = []
    # Make output path
    output_path = cfg.log_path
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )
    Mycallbacks.append(early_stop_callback)

    if cfg.General.server == 'train' :
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = str(cfg.log_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
    return Mycallbacks

from data.features import *
from torch.utils.data.sampler import SubsetRandomSampler

import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)


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
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/logs")
    wandb_logger = pl_loggers.WandbLogger(name='Final Dataset Kimianet 500',project='Multi_Instance_Learning')
    # slide_data = per_slide_features('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/FineTuned_Model_Features_dict.pickle')
    slide_data = per_slide_features('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/KimiaNet_Features_Final.pickle')
    labels_dict = dataset_labels('/mnt/largedrive0/katariap/feature_extraction/data/Dataset/Data.csv')
    dataset_features = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/kimianet_features/KimiaNet_Features_Final.pickle'
    with open('/mnt/largedrive0/katariap/feature_extraction/data/Code/kimianet_feature_extractor/src/data/selected_tiles/selected_clustering_500_final.json', 'r') as f:
        selected = json.load(f)
    # slide_data_500 = per_slide_selected(dataset_features,selected)
    validation_split = .1
    test_split = 0.1
    shuffle_dataset = True
    random_seed= 13
    # dataset = Features(slide_data,labels_dict)
    slide_list = []
    json_folder = '/mnt/largedrive0/katariap/feature_extraction/data/Dataset/DenseNet_Features/'
    with os.scandir(json_folder) as files:
        for file in files:
            slide_list.append(file.path)
        
   
    dataset = FeatureJson(slide_list,labels_dict)
    # dataset = Features(slide_data_500,labels_dict)
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

    #---->Define Data 
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
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=wandb_logger,
        callbacks=cfg.callbacks,
        max_epochs= 10,
        gpus=[0],  
        precision=cfg.General.precision,  
        check_val_every_n_epoch=2,
    )

    #---->train or test
    if cfg.General.server == 'train':
        # print('Starting Training')
        trainer.fit(model = model, datamodule = dm)
    else:
        # model_paths = list(cfg.log_path.glob('*.ckpt'))
        # model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        # for path in model_paths:
        #     print(path)

        new_model = model.load_from_checkpoint('/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/models/Multi_Instance_Learning/x3wwm3y5/checkpoints/epoch=9-step=1770.ckpt',cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml('/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/config/TransMIL.yaml')

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = 'train'
    cfg.Data.fold = args.fold


    # Use Yaml to create a config file

    #---->main
    main(cfg)
 