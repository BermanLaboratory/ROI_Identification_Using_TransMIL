import pytorch_lightning as pl
from data.features import FeatureJson, Features
import torch

class FeaturesInterface(pl.LightningDataModule):

    def __init__(self,slide_feature_data,labels_dict,sampler,batch_size=2):

        super().__init__()

        # separate train , test and val batch sizes
        self.batch_size = batch_size
        self.sampler = sampler
        # self.data_module = FeatureJson(slide_feature_data,labels_dict)

        self.data_module = Features(slide_feature_data,labels_dict)

    # All dataset manipulation is done here 
    # how to split etc , apply transforms
    def train_dataloader(self):
        
        return torch.utils.data.DataLoader(self.data_module, batch_size=self.batch_size, 
                            sampler=self.sampler['train'],num_workers = 1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_module, batch_size=self.batch_size,
                                sampler=self.sampler['val'],num_workers = 1)


    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_module,batch_size=self.batch_size,
                                    sampler = self.sampler['test'],num_workers = 1)