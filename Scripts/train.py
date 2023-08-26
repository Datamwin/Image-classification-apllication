import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sns
import argparse
import os

def args_paser():
    paser = argparse.ArgumentParser(description='trainer file')
    paser.add_argument('--data_dir', type=str, default='flowers', help='dataset directory')
    paser.add_argument('--gpu', type=bool, default='True', help='True: gpu, False: cpu')
    paser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    paser.add_argument('--epochs', type=int, default=10, help='num of epochs')
    paser.add_argument('--arch', type=str, default='vgg16', help='architecture')
    paser.add_argument('--hidden_units', type=int, default=[4096, 102], help='hidden units for layer')
    paser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='save train model to a file')
    args = paser.parse_args()
    
    return args

def process_data(train_dir, test_dir, valid_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                            transforms.RandomResizedCrop(224), 
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = test_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle = False)
    
    print('Dataloaders have been sucessfully created')
    
    return trainloader, testloader, validloader, train_datasets

def basic_model(arch):
    # Loading a pretrained_network
    print('Use vgg16 or resnet18 as pretrained model.vgg16 is used by default!')
    if arch == None or arch == 'vgg16':
        load_model = models.vgg16(pretrained = True)
        #load_model.name = 'vgg16'
        print('Current model: vgg16 loaded')
    else:
        load_model = models.resnet18 (pretrained = True)
        print('Current model: resnet18 loaded')
        model.name = arch
    
    return load_model


def set_classifier(model, hidden_units):
    if hidden_units == None:
        hidden_units = 512
    input = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input, hidden_units[0])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('output', nn.Linear(hidden_units[0], hidden_units[1])),
                          ('log_softmax', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier
    print('New classifier created sucessfuly')
    return model

 # Creating a train function         
def train_model(epochs, trainloader, validloader, gpu, model, optimizer, criterion):
    if type(epochs) == type(None):
        epochs = 10
        print("Epochs = 10")
        
    if gpu==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    steps = 0
    model.to(device)
    train_loss = 0
    
    for e in range(epochs):
    
        for inputs, labels in trainloader:

            steps += 1
            # Move image and label tensor to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Zeroing the gradients
            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0

            model.eval() # model is in evaluation mode now

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                for inputs, labels in validloader:

                    # move data to GPU
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass with validation data
                    logps = model.forward(inputs)
                    valid_loss += criterion(logps, labels).item()

                    # Accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print('Epoch: {}/{}..'.format(e+1, epochs),
                  'Training Loss: {:.3f}..'.format(train_loss/len(trainloader)),
                  'Valid Loss: {:.3f}..'.format(valid_loss/len(validloader)),
                  'Accuracy: {:.3f}'.format(accuracy/len(validloader)))

            train_loss = 0

            # After validation - model is set back into training mode
            model.train()
            
    return model

# Creating a test function
def test_model(model, testloader, gpu):
    
    if gpu==True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracy = 0
    model.eval()
    
    for inputs, labels in testloader:
    
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass with test data
        logps = model.forward(inputs)

        # accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim = 1)
        equality = top_class == labels.view(*top_class.shape)
        test_accuracy += equality.type(torch.FloatTensor).mean()

    print("Test Accuracy: {:.3f}".format(100 * (test_accuracy/len(testloader))))
    
# Function to save checkpoints                        
def save_checkpoint(model, train_dataset, arch, hidden_units, epochs, optimizer, save_dir):
                  
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'arch': arch,
                  'input_size': model.classifier[0].in_features,
                  'output_size': hidden_units[1], 
                  'classifier': model.classifier, 
                  'mapping': model.class_to_idx,
                  'number_of_epochs': epochs,
                  'optimizer_dict': optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    return torch.save(checkpoint, save_dir) 


def main():
    
    args = args_paser()
    
    # Definition of image data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Dataloaders
    trainloader, testloader, validloader, train_dataset = process_data(train_dir, test_dir, valid_dir)
    
    # Load pretrained model
    model = basic_model(args.arch)
    
    # Freeze parameters of pretrained model avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # Define classifier for the model
    model = set_classifier(model, args.hidden_units)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    print('Criterion and Optimizer sucessfuly defined!')
    
    #Calling the train function to train the model.
    print('Model training...')
    trained_model = train_model(args.epochs, trainloader, validloader, args.gpu, model, optimizer, criterion)
    
    # Calling the test function to test the model
    print('Testing the model...')
    test_model(trained_model, testloader, args.gpu)
    print('Model tested sucessfuly')
    print('Saving the model')
    save_checkpoint(trained_model, train_dataset, args.arch, args.hidden_units, args.epochs, optimizer, args.save_dir)
    print('Model saved!')
    print('Congratulations!') 
if __name__ == '__main__': main()
    
    