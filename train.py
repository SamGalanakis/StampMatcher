import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ColorJitter, GaussianBlur
import wandb
import torchvision
from dataloaders import StampDataset
from torch.utils.data import DataLoader
from model.losses import TripletLoss
from tqdm import tqdm
import os 
from model import MLP
from dataloaders import knn_query_manager
from utils import put_on_device

def get_transforms(config):
    start_transforms_train = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomCrop((80, 80)),
        ])
    
    start_transforms_test = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomCrop((80, 80)),
        ])

    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ColorJitter(),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.GaussianBlur(11)
        ])

    end_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_transform = transforms.Compose([start_transforms_train,augmentations,end_transforms])


    test_transform = transforms.Compose([start_transforms_test,end_transforms])

    return train_transform,test_transform


def initialize_model(config,device):
    encoder = torchvision.models.resnet50(pretrained=True).to(device)
    encoder_out  = encoder.fc.out_features
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    projection_head  = MLP(encoder_out,config['hidden_sizes'],config['emb_dim'],nonlin=torch.nn.GELU(),residual=True).to(device)

    return {'projection_head':projection_head,'encoder':encoder}


def load_save(model_dict,config,save_dict,device):
    model_dict['projection_head'].load_state_dict(save_dict['projection_head'])
    for key,val in model_dict.items():
        val = put_on_device(val,device)
    return model_dict


def main():
    run = wandb.init(config='configs/config.yaml',project = 'StampMatcher', entity='samme013')
    config = wandb.config
    

    train_transform,test_transform = get_transforms(config)


    loss_func = TripletLoss(margin = config['triplet_loss_margin'])
    dataset = StampDataset('data/stamps/stamps',train_transform)
    device = 'cuda'
    
    model_dict = initialize_model(config)


    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

    optimizer = torch.optim.Adam(
                model_dict['projection_head'].parameters(), lr=config["lr"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=  config['patience_scheduler'], 
    factor=0.1, min_lr=1E-8,verbose=True)


    save_model_path = 'model_save/'
    last_save_path = None
    best_so_far = 1E+32
    for epoch in range(config["epochs"]):
        print(f"Starting epoch: {epoch}")
        loss_running_avg = 0
        for batch_ind, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            embeddings = [model_dict['encoder'](x) for x in batch]
            projected_embeddings = [model_dict['projection_head'](x) for x in embeddings]
            loss = loss_func(*projected_embeddings)
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            loss_running_avg = (
                loss_running_avg*(batch_ind) + loss_item)/(batch_ind+1)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss)

        wandb.log({"loss_epoch": loss_running_avg,'current_lr':current_lr})
        

        
        if loss_running_avg < best_so_far:
            if last_save_path!=None:
                os.remove(last_save_path)
            print(f'Saving!')   
            savepath = os.path.join(
                save_model_path, f"{wandb.run.name}_e{epoch}_model_dict.pt")
            
            save_dict = {'config': config._items, "projection_head": model_dict['projection_head'].state_dict(
            )}
            torch.save(save_dict, savepath)
            last_save_path = savepath

            best_so_far = min(loss_running_avg,best_so_far)

if __name__ == '__main__':
    main()








