ROI Identification From Whole Slide Images Using Correlated Multi Instance Learning
==============================

Mutli Instance Learning is a type of machine learning technique in which the model receives a set of instances as a single bag and each bag has an label associated with it.


## Running the model:
All the file paths and setting for the model can be modified through the [configuration file](). 
The mode to train and test can be changed throught the file.

Parameters required for training:

If the number of patches selected for each image are of different number, only batch size of one works.
The Batch size can be increased in case the number  of patches for each image are same.

Parameters required for testing:

To Run Training or Testing of the model:

```shell
python main.py PATH_TO_CONFIG_FILE
```

## Creating Heatmaps:

### 1. Extract Attention Weights of a specific image from the trained model
Use [this]() to extract the attention weights of a specific whole slide image using sample ids.

The Script will generate .npy file that contains the attention matrix.

Sample Shell Command:
```shell
python extract_attentions.py CFG_FILE_PATH DST_FILE_PATH SAMPLE_ID
```
Arguments:
* `cfg`: Path to the Configuration File
* `dst`: Path to the Destination Folder
* `sample_id`: Sample Id of the Whole Slide to Extract Feature For.

### 2. Generate Heatmaps Using the Nystrom Attention Weights

The Code For Generating heatmaps is in the form of a jupyter notebook so that the heatmaps can be visualized in the notebook itself and once after tuning the parameters final heatmaps can be saved to the required destination.

Files Required:
1. Extracted Features Pickle File
2. Attention Weights file for the Slide whose heatmap is required.
3. The Whole Slide Image File (Downscaled by 10X):

Steps to Downscale .vsi Image Using Qupath:
1. File -> Export Images -> Original Pixels -> .tiff Format -> Downscale Factor = 10.

If the downscale Factor is changed or the Original Image is to be used for visualization. 
Change the `scale` factor in the jupyter notebook accordingly. By default it is 10.


`vis_heatmap` is the function used to generate heatmaps for the whole slide image.
Arguments of the Function:
* `alpha`: Range 0 to 1. This is used to handle the opacity of heatmaps over the original whole slide image. (default : 0.5)
* `binarize`: If the visualization of patches below a certain threshold is not required. ( default: False)
* `thresh`: The threshold in case binarize is set to True.



Project Organization
------------

    ├── docs
    |     └── Readme.md     
    │
    ├── models           
    │
    ├── requirements.txt   
    │                  
    └── src                <- Source code for use in this project.
        ├── __init__.py    
        │
        ├── config
        |      └── bermanlab.yaml
        |
        ├── data
        |   ├── __init__.py
        |   ├── features_interface.py           
        │   └── features.py
        │
        ├── models          
        │   │ 
        |   ├── __init__.py
        |   |                
        │   ├── architechture
        |   |       ├── MyLoss
        |   |       ├── MyOptimizer
        |   |       ├── TransMIL_interface.py
        |   |       └── TransMIL.py
        |   |
        |   ├── extract_attentions.py
        |   └── main.py
        │
        ├── utils
        |     ├── __init__.py
        |     └──  utils.py
        |
        └── visualization  
                └── heatmaps_visualization.ipynb
    

### References:
1. Shao, Zhuchen, et al. "Transmil: Transformer based correlated multiple instance learning for whole slide image classification." Advances in Neural Information Processing Systems 34 (2021): 2136-2147.

2. Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021). https://doi.org/10.1038/s41551-020-00682-w 


--------


