# CNN Rock Recognition
### Background
Final Project for the Spring 2022 CMSC 478, Rock Recognition ML project.

This project will be training the VGG11 model which is a Convolutional Nueral Net with 8 convolutional layers and 3 fully connected layers. This project will be training the CNN with 3(4) classes with the GPU for cuda acceleration.

0. Agates
1. Quartz
2. Obsidian
3. Granite
4. The Rock (joke class)

The data from this project is taken entirely from online images and repositories of rocks that we had chosen to best suit the project, selecting the images based on the classes we would like to use. The images are run through a transform to normalize everything to size and channels.

Everything about the VGG11 model was teken from the paper where it debuted: https://arxiv.org/pdf/1409.1556.pdf

## Setup
Install dependencies locally, requirements.txt is also avail, use `pip install -r requirements.txt` (Assumes you have Nvidia drivers)
1. `pip install torch`
2. `pip install torchvision`
3. `pip install matplotlib` 
4. `pip install numpy`
5. `pip install opencv-python`
6. `pip install tqdm`


## Files
### models.py
Contains the model VGG11 with all of its layers

### train.py
Does training and validation for the dataset and outputs a model `RockModel.pth` and 2 graphs with accuracy and loss figures.
to run: `python3 train.py`
