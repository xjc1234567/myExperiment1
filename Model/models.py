import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict

from Dataset import constants


######################### Evaluator #########################

class Classifier(nn.Module):
    def __init__(self, num_classes, with_softmax=False):
        super(Classifier, self).__init__()        
        in_dims =   (3, constants.IMAGE_SIZE*2, constants.IMAGE_SIZE*2, constants.IMAGE_SIZE)
        out_dims = (constants.IMAGE_SIZE*2, constants.IMAGE_SIZE*2, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.num_classes = num_classes

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),                    
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2)
                )
            )

        self.fc_layers.add_module(
            'fc_1',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE * constants.IMAGE_SIZE//16 *  constants.IMAGE_SIZE//16,
                constants.IMAGE_SIZE*2),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5)
            )
        )

        self.fc_layers.add_module(
            'fc_2',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE*2, num_classes),
            )
        )
        if with_softmax:
            self.fc_layers.add_module('softmax', nn.Sequential(nn.Softmax(dim=1)))

    
    def forward(self, imgs, feature = False):
        out = imgs        
        for i, conv_layer in enumerate(self.conv_layers, start=1):
            out = conv_layer(out)
        out = out.flatten(1, -1)        
        for fc_layer in self.fc_layers:
            img_features = out
            out = fc_layer(out)
        if feature:
            return out, img_features
        return out


class Parameterized(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(Parameterized, self).__init__()        
        self.model = nn.Sequential(
                    nn.Linear(num_inputs, num_inputs*20),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(num_inputs*20, num_inputs*10),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.25),
                    nn.Linear(num_inputs*10, num_classes)
                    )
    
    def forward(self, inputs):             
        return self.model(inputs)

### For the KD experimnt 
class StudentClassifier(nn.Module):
    def __init__(self, num_classes, with_softmax=False):
        super(StudentClassifier, self).__init__()        
        in_dims =   (3, constants.IMAGE_SIZE, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        out_dims = (constants.IMAGE_SIZE, constants.IMAGE_SIZE, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.num_classes = num_classes

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),                    
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2)
                )
            )

        self.fc_layers.add_module(
            'fc_1',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE * constants.IMAGE_SIZE//16 *  constants.IMAGE_SIZE//16,
                constants.IMAGE_SIZE),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5)
            )
        )

        self.fc_layers.add_module(
            'fc_2',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE, num_classes),
            )
        )
        if with_softmax:
            self.fc_layers.add_module('softmax', nn.Sequential(nn.Softmax(dim=1)))

    
    def forward(self, imgs, feature = False):
        out = imgs        
        for i, conv_layer in enumerate(self.conv_layers, start=1):
            out = conv_layer(out)
        out = out.flatten(1, -1)        
        for fc_layer in self.fc_layers:
            img_features = out
            out = fc_layer(out)
        if feature:
            return out, img_features
        return out

if __name__ == "__main__":
    exp_name = str(constants.HONEST) + "_" + str(constants.CURIOUS) + "_" + str(constants.K_Y) + \
               "_" + str(constants.K_S) + "_" + str(int(constants.BETA_X)) + "_" + str(int(constants.BETA_Y)) + \
               "_" + str(constants.SOFTMAX) + "/" + str(constants.IMAGE_SIZE) + "_" + str(constants.RANDOM_SEED)
    save_dir = "results_par/" + constants.DATASET + "/" + exp_name + "/"

    ## Test
    model = Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
    param_G = Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
    model.load_state_dict(torch.load(save_dir + "best_model.pt", map_location=torch.device('cpu')))
    param_G.load_state_dict(torch.load(save_dir + "best_param_G.pt", map_location=torch.device('cpu')))
    for param in param_G.parameters():
        print(param)