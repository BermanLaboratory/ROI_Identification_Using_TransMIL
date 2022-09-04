from utils.utils import *
from data.features import *
from models.architechture.TransMIL_interface import ModelInterface
import torch
import argparse
import os

parser = argparse.ArgumentParser(description='Script for Extracting Attention Weights from the Trained Model')
parser.add_argument("cfg",help = 'Path to the configuration File')
parser.add_argument('dst',help= 'Path to the destination Folder')
parser.add_argument('sample_id',help = 'Sample Id of the Whole Slide to extract attention weights for',type = int)
args = parser.parse_args()
config = vars(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # cfg = read_yaml('/mnt/largedrive0/katariap/feature_extraction/data/Code/multi_instance_learning/src/config/bermanlab.yaml')
    cfg = read_yaml(config['cfg'])
    slide_data = per_slide_features(cfg.Data.data_file)

    selected_images = [config['sample_id'].__str__()] #Sample Id to Extract Attention Weights For

    feature_list = torch.tensor(slide_data[selected_images[0]])
    feature_list = feature_list.unsqueeze(0)
    feature_list = feature_list.to(device)


    #------> Load The Model From Checkpoint

    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    model = model.load_from_checkpoint(cfg.General.weights_file_path,cfg=cfg)
    model.to(device)
    torch.set_grad_enabled(False)
    model.eval()

    file_name = selected_images[0]
    results_dict = model.model(data = feature_list)
    attentions_1 = results_dict['attentions_1']
    attentions_2 = results_dict['attentions_2']
    add_length = results_dict['add_length']

    file_name = file_name
    file_name = str(add_length) + '_attention_weights'+ '.npy'  # Add Length will be used for visualization purposes

    file_path_dst = os.path.join(config['dst'],file_name)
    np.save(file_name,attentions_2.cpu().detach().numpy())

