#imports
import torch
import json
from torchvision import datasets, transforms, models
from get_inputs import get_input_args
from process_image import process_image


def main():
    #get args
    arg = get_input_args()
    
    #enter the image path
    image_path = input("Please enter the path to the image you want to classify:")

    
    #image path for test issues
    #image_path = 'flowers/test/100/image_07896.jpg'    
    
    #load mapping table
    with open(arg.mapping_file, 'r') as f:
        cat_to_name = json.load(f)
    

    #load model type
    model_dict = torch.load(arg.save_dir)
    
    if arg.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arg.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif arg.arch == 'alexnet':
        model = models.alexnet(pretrained=True) 
    
    #load previously trained network
    model.classifier = model_dict['classifier']
    model.load_state_dict(model_dict['state_dict'])
    model.class_to_idx = model_dict['classs_to_idx']
    
    #prepare/process image for later processing through network
    image = process_image(image_path)
    image.unsqueeze_(0)
    
    #move model as well as input to GPU
    model = model.to(arg.gpu)
    image = image.to(arg.gpu)
    
    #run forward propagation
    logps = model.forward(image) 

    #calculate the accuracy
    ps = torch.exp(logps)
    
    #calculate the top k classes
    top_ps, top_class = torch.topk(ps, arg.k, dim=1)
    top_ps = top_ps.cpu().data.numpy().squeeze()
    top_class = top_class.cpu().data.numpy().squeeze()

     
    if arg.k > 1:
        final_probs = {}
        e = 0
        for i in top_class:
            if str(i) not in final_probs:
                final_probs[cat_to_name[str(i)]] = top_ps[e]   
            e += 1  
        x = 1
        for i in final_probs:
            print ("The",x,"most probable class is",i,"\nProbability of",final_probs.get(i)*100,"%\n")
            x += 1
    else:
        print ("The image shows the flower:", cat_to_name[str(top_class)], "\nThe probability is: ",top_ps*100,"%.")
       
main()
