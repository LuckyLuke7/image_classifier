#import
import torch
from torchvision import datasets, transforms, models


def transform(ImageFolder):
    
    data_dir = ImageFolder
    train_dir = data_dir + 'train/'
    valid_dir = data_dir + 'valid/'
    test_dir = data_dir + 'test/'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=40)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=40)

    return trainloader, testloader, validloader, train_data