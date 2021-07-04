import torch
from torch.utils.data import Dataset
import os
import io
import random
from utils import custom_pil_loader
from tqdm import tqdm



class KnnQueryManager:
    def __init__(self,root_dir,transforms,model,device='cuda'):
        self.root_dir = root_dir
        self.transforms = transforms
        self.file_paths = os.listdir(root_dir)
        self.device = device
        model.eval()
        model = model.to(device)
        projections = []
        print('Creating index')
        with torch.no_grad():
            for file_path in tqdm(self.file_paths):
                img_path = os.path.join(self.root_dir,
                                    file_path)
                img = custom_pil_loader(img)
                img = self.transforms(img).to(device)
                projection = model(img)
                projections.append(projection)
        self.projections = torch.stack(projections)
    

    def query(self,img_path,k=1,verbose = True):
        img = custom_pil_loader(img_path)
        img = self.transforms(img).to(self.device)
        dist = torch.linalg.norm(self.projections - img, dim=-1)
        knn = dist.topk(k, largest=False)
        if verbose:
            print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))
        return knn
    
    def __len__(self):
        return len(self.file_paths)

        

        
