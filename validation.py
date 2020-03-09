#import
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

def validation(model, testdata, device, criterion):
    #set model to eval status
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images,labels in testdata:

            images, labels = images.to(device), labels.to(device)
                    
            logps = model.forward(images) 
            loss = criterion(logps, labels)
            
            test_loss += loss.item()
                
            #calculate the accuracy
                
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
        #reset model to training status
        model.train()
    return accuracy, test_loss