# CMPSC445 Final

This project focuses on detecting breast cancer in high density breast tissue with mammogram images.


If you are beginning with .dcm files, start with the Convert_dicom_to_png.py file to convert the dicom image(s) to png. The data is loaded in png format initially.

You can run python ResNet50_Wt.py through any of your environments. You will need the following files:

----Convert_dicom_to_png.py----(Optional)
--------MammoData.py-----------(required)
--------ResNet50_Wt.py---------(required)

The following dependencies are needed to run the required programs.
###################################################################
(for)MammoData.py
###################################################################
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import time
from datetime import datetime
from IPython import display
import os
import torch
import multiprocessing as mp
import warnings
from pathlib import Path
from torchvision.io import read_image
from multiprocessing import Process, freeze_support
***********************************************************************
######################################################################
(for)ResNet50_Wt.py
#######################################################################
import os
import torch
from torch import nn
import torchvision
from MammoData import get_df, get_data
from multiprocessing import Process, freeze_support
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F
#from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
************************************************************************

## Getting Started
.........................................................................
-In  a terminal window, enter python Convert_dicom_to_png.py -dicomPath(or dir) -pngPath(or dir). Include the -f flag for folder conversions. Convert_dicom_to_png.py -f -dicomdir -pngdir.
-Replace the necessary directories in MammoData.py with your own. We recommend working with "INBREAST" contains high quality images with the relevant Metadata.
-Replace the initial length variable with the split of your choice. It is currently set as 0.7. Make sure to change it at line 94 to reflect the new split as well.
-In ResNet50_Wt, only necessary changes are optional. They include the learning rate, batch size and num_epochs. Scroll to line 244 to access these variables.
.........................................................................
