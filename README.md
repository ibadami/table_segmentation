## Semantic segmentation of table-top:

This assignment took 32 hours of work without including the training runs and initial research for model selection.
All the experiments are done on RTX 2070 super (8GB) and i7 CPU (64 GB). 
The FPS for target video is 60 fps
The FPS for prediction is 98 fps when visualization is turned off.


### Setup
To run the code, first setup a conda environment using the `environment.yml` file

`conda env create -f environment.yml`

Alternatively, you can create conda env first and then install the requirements 
manually from the list on `environment.yml` file.
`conda create -n tabeltopseg python=3.9 anaconda`

For creating videos, you must have ffmpeg installed on your machine:
To install ffmpeg run:
`sudo apt install ffmpeg`

Note: there are certain other dependencies that are already present in my 
machine and hence not actively installed during this project.

The code can be downloaded from here.

I have created the dataset for training using labelme software.
The dataset and pre-trained weights for the table segmentation model can be downloaded 
from here.