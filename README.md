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
If not using the dockerfile install dependencies locally. (Assumes you have Nvidia drivers)
1. `pip install torch`
2. `pip install torchvision`
3. `pip install matplotlib` 
4. `pip install numpy`
5. `pip install opencv-python`
6. `pip install tqdm`


# Dockerfile
Install Nvidia container toolkit to use GPU
1. `distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - && curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list`
2. `sudo apt-get update`
3. `sudo apt-get install -y nvidia-docker2`
4. `sudo systemctl restart docker`
5. `sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi` 

If this shows you an output with your current cuda version and drivers you are in business! 

Build the container with `sudo docker build -t lego .`

Run the container with `sudo docker run -it --rm --gpus all -v home/andrew/internship_summer21:home/andrew/internship_summer21 -w home/andrew/internship_summer21 lego /bin/bash`
 `--rm --gpus all` allows the container to use your GPU for training
 `-t` for an interactive container
 `-w` for the working directory

*Note: adjust the directory for your system, requires an absolute directory.*

## Files
### models.py
Contains the *untrained* model VGG11 with all of its layers

### train.py
Does training and validation for the dataset and outputs a model `LegoModel.pth` and 2 graphs with accuracy and loss figures.
to run: `python3 train.py`

### test.py
*Currently does a SINGLE image*
to run: `python3 test.py`


NOTES:  
1. Discuss the loss function best suited for this project.  
2. OPtimizer?  
3. tranforms best suited for rocks?  
4. Review the pytorch CNN he went over in a class and see about similarities
5. Make a list of the changes from homework 5 that impact the CNN in term sof loss and accuracy
add more if we need to
