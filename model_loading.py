import torch
import torchvision.transforms as transforms
import torchvision
from model import MLP
from utils import put_on_device





def get_transforms(config):
    start_transforms_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        ])
    
    start_transforms_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        ])

    # augmentations = transforms.Compose([
        
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.2),
    #     transforms.ColorJitter(brightness=0.25,contrast=0.1,saturation=0.25,hue=0.25),
    #     transforms.RandomPerspective(distortion_scale=0.1, p=0.1,fill=(1,1,1)),
    #     transforms.RandomAffine(degrees=(-10, 10),translate=(0,0.1), scale=(0.8,1), fill=(1,1,1)),
    #     transforms.GaussianBlur(11,sigma = (0.05,1.))
    #     ])

    augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomRotation(degrees=(-10, 10),fill=(255,255,255)),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3,fill=(255,255,255)),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomAffine(scale=(0.8,1),degrees=0,fill=(255,255,255)),
    transforms.GaussianBlur(11)
    ])

    end_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


    train_transform = transforms.Compose([start_transforms_train,augmentations,end_transforms])


    test_transform = transforms.Compose([start_transforms_test,end_transforms])

    return train_transform,test_transform


def initialize_model(config,device,mode='train',freeze_backbone=True):
    encoder = torchvision.models.resnet50(pretrained=True).to(device)
    fc_in  = encoder.fc.in_features
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze_backbone

    encoder.fc  = MLP(fc_in,config['hidden_sizes'],config['emb_dim'],nonlin=torch.nn.GELU(),residual=False).to(device)

    if mode=='eval':
        encoder.eval()
    

    return {'encoder':encoder}


def load_save(model_dict,config,save_dict,device):
    model_dict['encoder'].load_state_dict(save_dict['encoder'])
    for key,val in model_dict.items():
        val = put_on_device(val,device)
    return model_dict