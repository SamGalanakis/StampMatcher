import torch
import os

from torch.serialization import save 
from dataloaders import KnnQueryManager, knn_query_manager
from train import initialize_model,load_save,get_transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
def get_query_manager(checkpoint_path):
    save_dict = torch.load(checkpoint_path)
    config = save_dict['config']
    device = 'cuda'
    model_dict = initialize_model(config,device)
    model_dict = load_save(model_dict,config,save_dict,device)
    model = torch.nn.Sequential(model_dict['encoder'],model_dict['projection_head'])
    _ , test_transform = get_transforms(config)



    knn_query_manager = KnnQueryManager(config,config['train_dir'],test_transform,model,device)

    return knn_query_manager


if __name__ == '__main__':
    test_dir = 'data/stamps/test_images'
    checkpoint_path  = 'model_save/swift-butterfly-25_e19_model_dict.pt'
    query_manager = get_query_manager(checkpoint_path)
    for img_path in os.listdir(test_dir):
        path = os.path.join(test_dir,img_path)
        f, axarr = plt.subplots(4,1)
        knn,paths = query_manager.query(path,k=3)
        
        plot_paths = [path] + paths
        for index,match_path in enumerate(plot_paths):
            img = Image.open(match_path)
            img = img.resize([100,100])
            axarr[index].imshow(img)
        plt.show()
      


    

















