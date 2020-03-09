
import argparse
 
def get_input_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='flowers/', help='Insert the directory of the flowers. default=flowers/')
    parser.add_argument('--save_dir', type=str, default='save_directory', help='Define where to save the trained network. default=save_directory')
    parser.add_argument('--arch', type=str, default='vgg13', help='Define the Model Architecture out of: resnet, alexnet, or vgg. (vgg will be set as default')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Define the Model Architecture learning rate. Default = 0.0005')
    parser.add_argument('--hidden_unit', type=int, default=512, help='Define the Model Architecture hidden unit. Default = 512')
    parser.add_argument('--epochs', type=int, default=1, help='Define the Model Architecture number of epochs. Default = 1')
    parser.add_argument('--gpu', type=str, default='cuda', help='Define the processing element. Default = GPU')
    parser.add_argument('--k', type=int, default=1, help='Define the most probably classes. Default = 1')
    parser.add_argument('--mapping_file', type=str, default='cat_to_name.json', help='Chose the mapping file. Default = cat_to_name.json')

    
    arg = parser.parse_args()

    '''print ("Directory: \n", arg.dir, "\n"
          "Save Directory: \n", arg.save_dir, "\n"
          "Network Architecture: \n", arg.arch, "\n"
          "Learning Rate: \n", arg.learning_rate, "\n"
          "Hidden Units: \n", arg.hidden_unit, "\n"
          "Number of Epochs: \n", arg.epochs, "\n"
          "GPU: \n", arg.gpu, "\n"
          "Number of most probably classes: \n", arg.k)
    '''
    return parser.parse_args()