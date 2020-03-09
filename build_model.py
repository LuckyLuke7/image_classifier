#import
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def build_model(hidden_unit, arch, learning_rate):
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)    
    
    for param in model.parameters():
        param.requires_grad = False
    
    #Define our fully connected network
    classifier = nn.Sequential(nn.Linear(25088, hidden_unit),
                               nn.ReLU(),
                               nn.Dropout(p=0.5),
                               nn.Linear(hidden_unit, 102),
                               nn.LogSoftmax(dim=1))

    #Now attach our network to the model we want
    model.classifier = classifier

    #define citerion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    return model, criterion, optimizer

