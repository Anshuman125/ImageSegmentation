'''Some installations'''

# pip install segmentation-models-pytorch
'''Downloaded dataset from this repo'''
# pip install -U git+https://github.com/albumentations-team/albumentations
# pip install --upgrade opencv-contrib-python
# git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git

'''Imporrting data'''

import sys
sys.path.append('Human-Segmentation-Dataset-master')
# By appending a directory to sys.path, you make it so that Python will consider that directory when looking for modules to import.

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import helper

'''Setup Configurations'''

CSV_FILE = r"Human-Segmentation-Dataset-master/train.csv"
# DATA_DIR = '/content/'

DEVICE = "cpu"
EPOCHS = 25     # an epoch refers to one complete pass through the entire training dataset during the training of a model. During each epoch, the model's parameters are updated based on the error or loss calculated on the training data
LEARNING_RATE = 3e-3
IMAGE_SIZE = 320  # All images might be of different dimensions, this helps to give them all a constant dimension
BATCH_SIZE = 16

ENCODER = 'timm-efficientnet-b0'    # Its primary function is to transform input data into a different representation, often with a lower dimensionality.
# "Timm" refers to the PyTorch Image Models (timm) library. The timm library is a collection of pre-trained models for computer vision tasks in PyTorch.
# "EfficientNet" refers to a family of convolutional neural network architectures. These models are known for scaling the network's depth, width, and resolution in a principled manner.

WEIGHTS = 'imagenet'  # ImageNet is a large-scale dataset used in the field of computer vision for image classification and object detection tasks.

'''Checking if it is imported or not'''

dataframe = pd.read_csv(CSV_FILE)
# Read the CSV FILE
print(dataframe.head())
# I have two columns here. Mask is the ground truth and Image column containing original images

row = dataframe.iloc[4]
# In pandas, iloc indexer is used for integer-location based indexing, allowing you to access a group of rows and columns by their integer position.

image_path = row.images
mask_path = row.masks

image = cv2.imread(image_path)
# Using the imread function of OpenCV (Open source Computer Vision) library of Python to read the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Using the cvtColor function of OpenCV library to convert the colour of the image to view it properly. OpenCV represents images in the BGR color space by default

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
# Using the imread function of OpenCV (Open source Computer Vision) library of Python to read the mask image and the mask is going to be in gray scale as they have only one channel thus it provides simplicity, reduces dimensionality and provides memory efficiency
# I divide the pixel value by 255 to normalise it. The value is now scale between [0, 1]
# Why specifically 255? In JPEG and PNG formats pixel intensities are represented as unsigned integers ranging from 0 to 255. Each colour channel can take value between 0 to 255

'''Using the matplotlib library to see the image and compare it with its ground truth'''

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

ax1.set_title('IMAGE')
ax1.imshow(image)

ax2.set_title('GROUND TRUTH')
ax2.imshow(mask,cmap = 'gray')
plt.show()

'''Splittng it into training dataframe nad validation dataframe'''

train_df, valid_df = train_test_split(dataframe, test_size=0.2, random_state=42)
# Using train_test_split function to spplit the data into train dataframe and valid datatframe
# Test size = 0.2 represents the allocation of data. In this case, 20% of the data will be used for validation, and the remaining 80% will be used for training.
# random_ state= 42 is used to control the randomness of the data splitting process. When you set a specific value for random_state the data will be split in the same way every time you run the code
# 42 here is an arbitrary value and used as a convention

'''Data Augmentation'''

# augmentation is the process of applying various transformations to the existing training dataset to create additional datasets
# albumentation is a popular open source python library for image augmentation
# if there is any augmentation done on any image then its label should not be affected from it but in out case the mask image should also reflect those augmenteed changes
import albumentations as A

def get_train_augs():
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)
  ], is_check_shapes=False)
  # Compose is a function in albumentations library which is used to apply a sequence of augmentations with your data
def get_valid_augs():
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE)
  ], is_check_shapes=False)

'''Creating custom dataset'''

from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  # created a class and collected all important functions inside it

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return (len(self.df))

  def __getitem__(self, index):
    row = self.df.iloc[index]
    image_path = row.images
    mask_path = row.masks
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # here mask already has 2 dimensions height and width but we are going to add one more dimension channel which is one because it is a grayscale image
    # to give image three dimensions
    # many deep learning frameworks especially those designed for image processing there the convention for handling image data is to use a three-dimensional array

    mask = np.expand_dims(mask, axis = -1)
    # the resulting array will have a shape of (height, width, 1)
    # this is often useful when working with image processing libraries or models that expect a certain number of channels even if the mask itself is grayscale

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      # it will return key and mask in dictionary format
      # image as the key and mask as the value
      image = data['image']
      mask = data['mask']

      # h, w, c will be converted to c, h, w as it is the convention

      image = np.transpose(image, (2, 0, 1)).astype(np.float32)
      # The astype method is used to cast the elements of the array to a specified data type
      # converting is done beause of its compatibility with deeplearning and efficiency
      # same is done with mask

      mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

      # it is converted form numpy to tensor because of its compatibility and GPU acceleration with PyTorch
      image = torch.Tensor(image) / 255.0
      mask = torch.round(torch.Tensor(mask) / 255.0)
      # round off the value of mask to either 0 or 1
      # 0 for background, 1 for foreground

      return image, mask

trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

index = 4
image, mask = trainset[index]
helper.show_image(image,mask)
plt.show()

'''Load dataset into batches'''

# DataLoader is used to create batches of data instead of loading the entire dataset at once
from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)
# Dividing the trainset and validset into batches of batch size 16 which I defined in beginning
# Batching helps in memory efficiency and optimization of the program

print(f"Total number of batches in trainloader: {len(trainloader)}")
print(f"Total number of batches in validloader: {len(validloader)}")

# Lets see the shape of a batch
for image, mask in trainloader:
  break
print(f"Shape of one batch of image: {image.shape}")
print(f"Shape of one batch of mask: {mask.shape}")

'''Create Segmentation Model'''

# Image segmentation is the process of dividing an image into meaningful regions for computer vision
from torch import nn
# The nn module provides the building blocks for creating neural network models including layers, loss functions, and optimization algorithms
import segmentation_models_pytorch as smp
# segmentation_models_pytorch is a PyTorch-based library that provides pre-built segmentation models and tools for working with segmentation tasks
from segmentation_models_pytorch.losses import DiceLoss
# Dice Loss is a common loss function used in image segmentation tasks
# Dice Loss = 1 - (2*intersection/union)

class SegmentationModel(nn.Module):

  def __init__(self):

    super(SegmentationModel, self).__init__() # This line calls the constructor of the parent class
    # Calling the parent class constructor using super() in Python is a good practice that ensures proper initialization and sets up the inheritance hierarchy correctly

    self.arc = smp.Unet(
    # This line creates an instance of a U-Net segmentation model using the segmentation_models_pytorch library (smp)
    # U-Net is a popular architecture for image segmentation.

        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        in_channels = 3,
    # in_channels=3 Specifies the number of input channels
    # In typical RGB images, this value is set to 3

        classes = 1,
    # classes=1: Specifies the number of output channels or classes
    # In this case it is set to 1 indicating a binary segmentation task

        activation = None
    # No activation function is applied
    # The raw logits or predictions are returned.

    )

  def forward(self, images, masks = None):
  # This code defines the forward method for a PyTorch neural network class
  # The forward method takes input images and an optional masks tensor as arguments and returns logits (raw model predictions) along with a loss value

    logits = self.arc(images)
    # The model is takes input images and produce segmentation logits as output

    if masks!=None:
      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      # This line computes the Dice Loss between the predicted logits and the ground truth masks using the DiceLoss class
      # The mode='binary' argument indicates that it is a binary segmentation task
      # This line computes the Binary Cross Entropy (BCE) loss with logits between the predicted logits and the ground truth masks

      return logits, loss1+loss2
      # The choice of loss functions depends on the characteristics of the task, the desired model behavior, and the specific challenges posed by the dataset
      # It is a common practice to experiment with different loss functions and combinations to find the most suitable approach for a given segmentation problem

    return logits

model = SegmentationModel()
model.to(DEVICE);
# model.to(DEVICE) is a PyTorch method used to move a model to a specified device
# It makes the model's computations run on the specified hardware device improving performance for large-scale neural network computations

'''Create Train and Validation Function'''

def train_fn(data_loader, model, optimiser):

  model.train()
  # When you're training a neural network it's important to specify whether the model is in training or evaluation mode because certain layers such as dropout and batch normalization behave differently during training and evaluation
  # The model.train() method in PyTorch is used to set the model in training mode

  total_loss = 0.0

  for images, masks in tqdm(data_loader):
  # tqdm a useful tool for visualizing the progress of an iterative process

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)
    # These lines move the tensors images and masks to the specified device (DEVICE = 'cuda').
    # This is a common practice when working with PyTorch models that are already on a specific device (here cuda)

    optimiser.zero_grad()
    # before computing the gradients for the model parameters it is necessary to zero out the gradients from the previous optimization step

    logits, loss = model(images, masks)
    loss.backward()
    # it calculates how much each model parameter contributed to the loss during the forward pass

    optimiser.step()
    # it is responsible for updating the model parameters based on the computed gradients
    # w_new = w_old − (learning_rate × gradient)

    total_loss+=loss.item()

  return (total_loss/len(data_loader))

def eval_fn(data_loader, model):
# during validation no weight upgradation is required
# removed optimizer

  model.eval()
  # it is important to define the model here in eval
  # because we dont want to use here any dropout layer

  total_loss = 0.0

  with torch.no_grad():
  # no gradient computation is done
  # increases efficieny
    for images, masks in tqdm(data_loader):

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits, loss = model(images, masks)

      total_loss+=loss.item()

  return (total_loss/len(data_loader))

'''Train Model'''

optimiser = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
# module 'torch.optim' has no attribute 'legacy' as it had in tensorflow

best_valid_loss = np.Inf

for i in range(EPOCHS):

  train_loss = train_fn(trainloader, model, optimiser)
  valid_loss = eval_fn(validloader, model)

  if valid_loss<best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    # it is used in PyTorch to save the state dictionary of a model to a file.
    # after training a model it is good to save the model's parameters to a file so that it can be loaded later for inference or further training

    print("Saved Model")
    best_valid_loss = valid_loss

  print(f"Epochs : {i+1}, Train Loss : {train_loss}, Valid Loss = : {valid_loss}")

idx = 20
model.load_state_dict(torch.load("best_model.pt"))

image, mask = validset(idx)
logits_masks = model(image.to(DEVICE).unsqueeze(0))
# Adds one batch dimension
# (C, H, W) -> (1, C, H, W)

pred_mask = torch.sigmoid(logits_masks)
pred_mask = (pred_mask > 0.5)*1.0
# Any predicted value greater than 0.5 is considered as 1

helper.show_image(image, mask, pred_mask.detach().squeeze(0))
# move the image to cpu (if using gpu) and then remove the singleton dimension at the 0th position
plt.show()