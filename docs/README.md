ROI Identification From Whole Slide Images Using Correlated Multi Instance Learning
==============================

Multi Instance Learning is a type of machine learning technique in which the model receives a set of instances as a single bag and each bag has an label associated with it.
Multi Instance Learning helps is solving the weakly supervised classification problems.

[TransMIL](https://arxiv.org/abs/2106.00908) has been used in this project. TransMIL takes into account the correlation between various instances(patches) of the whole slide image. The code for TransMIL's architechture is publically available. The architechture has been used as it is. 

## Running the model:
All the file paths and settings for the model can be modified through the [configuration file](src/config/bermanlab.yaml). 

Parameters of configuration file:

```yaml
General:
    seed: # Seed to Generate all the randomness in the pipeline
    fp16: # Usually the training is in 32 bit precision. 16 bit precision can be used to imporve performance and reduce memory use. 
    amp_level: O2
    precision: 16 
    weights_file_path: # Path to the model checkpoint file. 
    epochs: # Specify max number of epochs while training
    grad_acc: 1 # Grad accumulation to increase batch size. As of now Batch size of one will work. 
    frozen_bn: False
    patience: 10
    mode: # modes to run the model in (train,test)
    log_path: # Path to directory to store the model run logs
    project_name: # Name of the project in the wandb (project to sync the logs with)
    run_name: # Will be used to sync logs with wandb. This will be name of each experiment. Specify unique details of each run for identification.
    

Data:
    data_type: # The data input can be a single vector file for all whole slides or multiple files containing features for each whole slide
    dataset_name: berman_lab_cohort
    data_shuffle: # To shuffle dataset or not.
    data_folder: # Path to the input data folder (if feature vector files are different for each whole slide)
    selected_patches_json: # Path to selected patches file
    data_file: # Path to Extracted Feature File (Features extracted from fine tuned kimianet)
    custom_test_data_file: # Path to a separate test data file (if test split is not from the same dataset)
    label_dir: # Path to the CSV file that Contains Labels associated with each sample id.(Columns: Sample ID and Sample Grade). The grades are binarized to 0 and 1. 
    fold: 0
    nfold: 4
    split_type: # The Dataset split can be random or balanced
    split_test: # Test split from this dataset is required or not
    train_split: 0.8
    validation_split: 0.1 
    test_split: 0.1

    train_dataloader:
        batch_size: 1 
        num_workers: 8

    val_dataloader:
        batch_size: 1
        num_workers: 8

    test_dataloader:
        batch_size: 1
        num_workers: 8

Model:
    name: TransMIL
    n_classes: 2


Optimizer: # Each Optimizer requires different arguments and parameters. The values can be specified depending on the optimizer used. 
    opt: # Any of the Optimizer from the MyOptimizer directory can be utilized.
    lr: # Learning Rate for the optimizer
    opt_eps: null 
    opt_betas: null
    momentum: null 
    weight_decay: 0.00001

Loss:
    base_loss: # Any of the loss function from the list in 'MyLoss' directory can be utilized.

```

To Run Training or Testing of the model:

```shell
python main.py PATH_TO_CONFIG_FILE
```

## Creating Heatmaps:

### 1. Extract Attention Weights of a specific image from the trained model
Use [this file](src/models/extract_attentions.py) to extract the attention weights of a specific whole slide image using sample ids.

The Script will generate .npy file that contains the attention matrix.
The attention matrix extracted contains weights for correlation of each patch with other.
For our use case only self attention weights are required.
Those are the diagonal values. This extraction is done while running the code for Generating Heatmaps.

Sample Shell Command:
```shell
python extract_attentions.py CFG_FILE_PATH DST_FILE_PATH SAMPLE_ID
```
Arguments:
* `cfg`: Path to the Configuration File
* `dst`: Path to the Destination Folder
* `sample_id`: Sample Id of the Whole Slide to Extract Attention weights for.

### 2. Generate Heatmaps Using the Nystrom Attention Weights

The Code For Generating heatmaps is in the form of a jupyter notebook so that the heatmaps can be visualized in the notebook itself and once after tuning the parameters final heatmaps can be saved to the required destination.

Files Required:
1. Extracted Features Pickle File
2. Attention Weights file for the Slide whose heatmap is required.
3. The Whole Slide Image File (Downscaled by 10X):

Steps to Downscale .vsi Image Using Qupath:
1. File -> Export Images -> Original Pixels -> .tiff Format -> Downscale Factor = 10.

If the downscale Factor is changed or the Original Image is to be used for visualization. 
Change the `downsample` factor in the jupyter notebook accordingly. By default it is 10.


`vis_heatmap` is the function used to generate heatmaps for the whole slide image.
Arguments of the Function:
* `alpha`: Range 0 to 1. This is used to handle the opacity of heatmaps over the original whole slide image. (default : 0.5)
* `binarize`: If the visualization of patches below a certain threshold is not required. ( default: False)
* `thresh`: The threshold in case binarize is set to True.



Project Organization
------------

    ├── docs
    |     └── Readme.md     
    ├── models   <- Stores the Checkpoints(weights) for the trained models      
    │
    ├── requirements.txt   <- contains the library requirements for the project
    │                  
    └── src                <- Source code for use in this project.
        ├── __init__.py    
        │
        ├── config
        |      └── bermanlab.yaml  <- Contains all the modifiable configuration options
        |
        ├── data
        |   ├── __init__.py
        |   ├── features_interface.py     <-  Interface to the dataset      
        │   └── features.py               <-  contains the code to the dataset Class
        │
        ├── models          
        │   │ 
        |   ├── __init__.py
        |   |                
        │   ├── architechture
        |   |       ├── MyLoss           <-  Contains all the available Loss Functions
        |   |       ├── MyOptimizer       <-  Contains All available Optimizers
        |   |       ├── TransMIL_interface.py <-  Interface to the main model
        |   |       └── TransMIL.py          <-  Architechture of the model
        |   |
        |   ├── extract_attentions.py  <- Code to Extract the Attentions weights for a wsi using trained model
        |   └── main.py    <- Code to Run Training And Testing On the Datasets.
        │
        ├── utils
        |     ├── __init__.py
        |     └──  utils.py  <-  All the helper functions used through the project are availble in this file
        |
        └── visualization  
                └── heatmaps_visualization.ipynb  <- Code for Generation of Heatmaps (Jupyter Notebook)
    

### References:
1. Shao, Zhuchen, et al. "Transmil: Transformer based correlated multiple instance learning for whole slide image classification." Advances in Neural Information Processing Systems 34 (2021): 2136-2147.

2. Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w 


--------


