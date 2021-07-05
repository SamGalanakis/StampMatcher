import torch
import torchvision.transforms as transforms
import wandb
from dataloaders import StampDataset
from torch.utils.data import DataLoader
from model.losses import TripletLoss
from tqdm import tqdm
import os 
from test import generate_results
from  model_loading import initialize_model,load_save,get_transforms



def main():
    run = wandb.init(config='configs/config.yaml',project = 'StampMatcher', entity='samme013')
    config = wandb.config
    

    train_transform,test_transform = get_transforms(config)


    loss_func = TripletLoss(margin = config['triplet_loss_margin'])
    dataset = StampDataset(config['train_dir'],train_transform)
    device = 'cuda'
    
    model_dict = initialize_model(config,device=device,mode='train')

    optimizer = torch.optim.Adam(
                model_dict['encoder'].fc.parameters(), lr=config["lr"],weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=  config['patience_scheduler'], 
    factor=0.1, min_lr=1E-8,verbose=True)

    if config['checkpoint_path']:
        print(f'Loading from checkpoint')
        save_dict = torch.load(config['checkpoint_path'])
        model_dict = load_save(model_dict,config,save_dict,device)
        optimizer.load_state_dict(save_dict['optimizer'])
        scheduler.load_state_dict(save_dict['scheduler'])
    else:
        print('Training from scratch')
    dataloader = DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=4,pin_memory=True,drop_last=True)

    


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
            loss = loss_func(*embeddings)
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
            
            save_dict = {'config': config._items,'encoder':model_dict['encoder'].state_dict(),'scheduler':scheduler.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(save_dict, savepath)
            last_save_path = savepath
            best_so_far = min(loss_running_avg,best_so_far)
            figs = generate_results(config,savepath,k=8)
            img_dict = {f'result_{i}':figs[i] for i in range(len(figs))}
            wandb.log(img_dict)

if __name__ == '__main__':
    main()








