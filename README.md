## Semantic segmentation of table-top:

This assignment took 32 hours of work without including the training runs and initial research for model selection.
All the experiments are done on RTX 2070 super (8 GB) GPU and i7 16 core (64 GB) CPU. 
The FPS for input video is 60 fps
The FPS for prediction is 65 fps when visualization is turned off.

### License
Each file shows the original author of the file. The license of the original repository is MIT and hence the current 
repo extends the same license.

### Setup
To run the code, first setup a conda environment using the `environment.yml` file

`conda env create -f environment.yml`

Alternatively, you can create conda env first and then install the requirements 
manually from the list on `environment.yml` file.
`conda create -n tabletopseg python=3.9 anaconda`

Activate the conda environment using:
`conda activate tabletopseg`

For creating videos, you must have ffmpeg installed on your machine:
To install ffmpeg run:
`sudo apt install ffmpeg`

Note: there are certain other dependencies that are already present in my 
machine and hence not actively installed during this project.

The code can be downloaded from here.

I have created the dataset for training using labelme software.
For each video, first image of that video is saved as a training image. Using the labelme software, precise label for the 
table is generated and saved as json.
Download the dataset, pre-trained models and results and extract the folders in the main code folder.

### Approach
This project is a transfer learning performed for semantic segmentation of tabletop using the DDRNet trained on 
cityscape dataset. Please refer to the source code for the model here https://github.com/chenjun2hao/DDRNet.pytorch

#### Training
Once the dataset is generated, train the model using 
python3.9 train.py <--options>
The default parameters will take care of all the option for the present setup.

#### Testing
To infer the new video, use the following command
`python3.9 test.py --visualize True --scaling=2`
- visualize switches on the debugging mode and shows the detection on the fly.
- scaling is used to scale down the input image size for faster inference. Higher the scaling lower the segmentation 
  accuracy will be.
  
Rest of the options in default will work with the current setup.

#### Visualization
Apart from turning on the visualization flag during inference, you can create the visualization using the tools
provided in the visualize.py
When visualize is True in test.py, the full results of original image and segmentation masks will be written in the disk.
Using these images a resultant video will be created by function `create_video_from_images.py`

If the visualization is False during test.py, only detections are saved.
To create the visualization using those detections and the original video, you can use `create_video_from_detections.py` function.

The detections saved during the test.py are 
Refer to `visualize.py` for more information.

#### Evaluation
Due to no ground truth available for the videos, I have not implemented the evaluation script.
But the visual test shows high segmentation accuracy.

Note: If you run into an error or require more information, please contact me.