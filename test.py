import torch
import os

from torch.serialization import save 
from dataloaders import KnnQueryManager, knn_query_manager
import matplotlib.pyplot as plt
from PIL import Image
from  model_loading import initialize_model,load_save,get_transforms



def get_query_manager(checkpoint_path):
    save_dict = torch.load(checkpoint_path)
    config = save_dict['config']
    device = 'cuda'
    model_dict = initialize_model(config,device,mode='eval')
    model_dict = load_save(model_dict,config,save_dict,device)
    _ , test_transform = get_transforms(config)



    knn_query_manager = KnnQueryManager(config,config['train_dir'],test_transform,model_dict['encoder'],device)

    return knn_query_manager


def generate_results(config,checkpoint_path,k=8):
    test_dir = 'data/stamps/test_images'
    query_manager = get_query_manager(checkpoint_path)
    run_name = os.path.basename(checkpoint_path).split('_')[0]
    save_folder = f'results/{run_name}'
    figs = []
    try:
        os.mkdir(save_folder)
    except:
        pass
    for img_number,img_path in enumerate(os.listdir(test_dir)):
        path = os.path.join(test_dir,img_path)
        f, axarr = plt.subplots(k+1,1)
        knn,paths = query_manager.query(path,k=k)
        
        plot_paths = [path] + paths
        for index,match_path in enumerate(plot_paths):
            img = Image.open(match_path)
            img = img.resize([100,100])
            axarr[index].imshow(img)
        figs.append(plt.gcf())
        plt.savefig(os.path.join(save_folder,f'result_{img_number}.png'))
    return figs


        

      


    

















