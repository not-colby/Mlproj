# Remove warning messages
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
#From the file in the same directory
from models import VGG11
matplotlib.style.use('ggplot')

RGB = 3
GRAYSCALE = 1
#For training with GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"[INFO]: Computation device: {device}")
epochs = 10 # number of time you go through the dataset
#Batch size breaks on 1, not sure why
BATCH_SIZE = 30 # how much data you want to train per epoch, it divides your samples by the number
#Hgher batch size is better, keeps overhead down!
NUM_CLASSES = 2 #number of classes
PATH = os.getcwd()
print(PATH)
DATA_PATH = PATH +"/trainData"
VALID_PATH = PATH +"/validateData"

# Transforms based on the VGG paper, technically for pre-trained i think
#Grayscale transform added to keep data to a minimum, must be done at the END
train_transform = transforms.Compose([   
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5], std = [0.5]),
    transforms.Grayscale()
        ])

valid_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.5), std=(0.5)),
     transforms.Grayscale()
     ])
'''
# training dataset and data loader
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, 
                                             transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
# validation dataset and dataloader
valid_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, 
                                           transform=valid_transform)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)
'''

# training dataset and data loader
train_dataset = torchvision.datasets.ImageFolder(root=DATA_PATH, 
                                                transform=train_transform)
                                            
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                shuffle=True, 
                                                batch_size=BATCH_SIZE)

# validation dataset and dataloader
valid_dataset = torchvision.datasets.ImageFolder(root=VALID_PATH, 
                                                transform=valid_transform)


valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                shuffle=True, 
                                                batch_size=BATCH_SIZE)

# instantiate the model
#3 classes, the rocks we want to use
model = VGG11(in_channels=GRAYSCALE, num_classes=NUM_CLASSES) #1 = grayscale 3 = RGB
model.to(device)

print("[INFO]: Batch size: ", BATCH_SIZE)
print("[INFO]: Number of train samples: ", len(train_dataset))
print("[INFO]: Number of validation samples: ", len(valid_dataset))
# classes are detected by folder structure
print("[INFO]: Detected Classes are: ", train_dataset.class_to_idx) 

# the loss function (NO TOUCHY)
criterion = nn.CrossEntropyLoss()
# the optimizer, right from the VGG paper (NO TOUCHY)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)

# training
def train(model, trainloader, optimizer, criterion):
    torch.cuda.empty_cache()
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    train_total = 0
    counter = 0
    #iterate through the data loader, pull images and labels from it
    #propagate images throuh the model, then backpropagate the current loss
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        
        counter += 1
        image, labels = data
        
        #send images and labels to the GPU
        image = image.to(device)#images.cuda() works too
        labels = labels.to(device) #labels.cuda() also works
        
        #clear the gradients of all variables in this optimizer
        #Gradients must be zero beofore the backpropagation "loss.backwards call"
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(image)
        
        # calculate the loss
        loss = criterion(outputs, labels)
        train_total += labels.size(0)

        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1) #preds = predictions
        
        #Calculate the accuracy cont
        train_running_correct += (preds == labels).sum().item()
        train_running_loss += loss.item()

        #ONLY DO IN TRAINING MODE
        loss.backward()
        optimizer.step()


        #Debugging code (Verbose outputs)
        print('preds', preds)
        print('labels', labels)
        print('add', (preds == labels).sum().item())
        #print('outputs: ', outputs)
        print('max', torch.max(outputs, 1))

    #Debugging code
    #print()
    #print('Total number correctly trained (train_running_correct)', train_running_correct)
    #print('Total number incorrectly trained (train_running_loss)', train_running_loss)
    #print('Total number trained (train total): ', train_total)
    

    #loss calculated by loss/repetitions, counter DOES NOT equal total_trained
    '''
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / train_total )
    return epoch_loss, epoch_acc
    '''
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc
    

# validation
def validate(model, testloader, criterion):
    
    model.eval() # eval mode turns off some layers like dropout and BatchNorm not used for eval
    
    # we need two lists to keep track of class-wise accuracy
    #MAGIC NUMBER HERE 10, I THINK IT SHOULD BE NUM_CLASSES
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))

    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    valid_total = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(image)

            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            #Accuracy Continued
            valid_running_correct += (preds == labels).sum().item()
            #valid_total += labels.size(0)
            

            # calculate the accuracy for each class
            correct  = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
            #debugging code
            #print('preds', preds)
            #print('labels', labels)
            #print('add', (preds == labels).sum().item())
            #print('max', torch.max(outputs),1)


    #For debugging
    #print()
    #print('Number of images correctly validated(valid_running_correct): ', valid_running_correct)
    #print('Total number of validated images(valid_total): ', valid_total)
    
    #print('Total correct identification per class: ', class_correct)
    #print('Total identifications per class: ', class_total)

    #Calculating the accruacy
    '''
    epoch_loss = valid_running_loss / len(testloader)
    epoch_acc = 100. * (valid_running_correct / valid_total)
    '''
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    

    # print the accuracy for each class after evey epoch
    # the values should increase as the training goes on
    print('\n')
    #ANOTHER MAGIC NUMBER 10 HERE, NOT SURE IF IT SHOULD ALSO BE NUM_CLASSES
    for i in range(NUM_CLASSES):
        print(f"Accuracy of class {i}: {100*class_correct[i]/class_total[i]}")
    return epoch_loss, epoch_acc

# start the training
# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
for epoch in range(epochs):
    
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_dataloader, 
                                                optimizer, criterion)
    valid_epoch_loss, valid_epoch_acc = validate(model, valid_dataloader,  
                                                criterion)
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print('\n')
    print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
    print('-'*50)

# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, './outputs/rockModel.pth')
 
#Code to view the plots
plt.figure(figsize=(10, 7))
plt.plot(
    train_acc, color='green', linestyle='-', 
    label='train accuracy'
)
plt.plot(
    valid_acc, color='blue', linestyle='-', 
    label='validataion accuracy'
)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./outputs/accuracy.jpg')
plt.show()
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(
    train_loss, color='orange', linestyle='-', 
    label='train loss'
)
plt.plot(
    valid_loss, color='red', linestyle='-', 
    label='validataion loss'
)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./outputs/loss.jpg')
plt.show()

print('TRAINING COMPLETE')
