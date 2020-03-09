# Import
import numpy as np
import torch
import json
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from get_inputs import get_input_args
from torchvision import datasets, transforms, models

from transform_image import transform
from build_model import build_model
from validation import validation
from save import save

def main():
    
    #get the arguments
    arg = get_input_args()
    
    #load the train, test and validation data
    trainloader, testloader, validloader, train_data = transform(arg.dir)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    #build the model
    model, criterion, optimizer = build_model(arg.hidden_unit, arg.arch, arg.learning_rate)
    #model to gpu
    model.to(arg.gpu)

    #train the network
    
    epochs = arg.epochs
    step = 0
    running_loss = 0
    print_every = 10

    for e in range(epochs):
        for images, labels in trainloader:
            step += 1
            images, labels = images.to(arg.gpu), labels.to(arg.gpu)
       
            optimizer.zero_grad()
       
            logps = model(images)
            loss = criterion(logps, labels)
       
            loss.backward()
            optimizer.step()
       
            running_loss += loss.item()
        
            #test network after some steps and print the current progress
            if step % print_every == 0:
                accuracy , test_loss = validation(model, testloader, arg.gpu, criterion)

                print(f"Epoch: {e+1}/{epochs}.. "
                f"Test loss: {test_loss/len(testloader):.3f}.. "
                f"Test accuracy: {accuracy / len(testloader)*100:.2f}%.. "
                f"The train loss: {running_loss/print_every:.3f}.. ")
            running_loss = 0
            
    #save the model
    model.class_to_idx = train_data.class_to_idx
    torch.save({'epochs': epochs,
            'classifier': model.classifier,
             'optimizer_state_dict': optimizer.state_dict(),
             'state_dict': model.state_dict(),
             'classs_to_idx': model.class_to_idx}, arg.save_dir)    
    
main()