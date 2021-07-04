import torch
import os

from torch.serialization import save 
from dataloaders import KnnQueryManager, knn_query_manager
from train import initialize_model,load_save,get_transforms


checkpoint_path  = ''
test_dir = 'data/stamps/test_images'
save_dict = torch.load(checkpoint_path)
config = save_dict['config']
device = 'cuda'
model_dict = initialize_model(config,device)
model_dict = load_save(model_dict,config,save_dict,device)
model = torch.nn.Sequential([model_dict['encoder'],model_dict['projection_head']])
_ , test_transform = get_transforms(config)



knn_query_manager = KnnQueryManager(config['train_dir'],test_transform,model,device)


print()

















