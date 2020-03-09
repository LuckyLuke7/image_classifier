#import
from PIL import Image
from torchvision import datasets, transforms, models

def process_image(image):
    '''
    Input:
    image --> Image path
    Output:
    img_tensor --> Tensor ready to test in network
    '''
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor