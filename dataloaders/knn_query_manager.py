import torch
from torch.utils.data import DataLoader,Dataset
import os
import io
import random
from utils import custom_pil_loader
from tqdm import tqdm
import torchvision



class TestLoaderForKnn(Dataset):
    def __init__(self,root_dir,transforms):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.file_paths = os.listdir(root_dir)



    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self,idx):
        img_path = os.path.join(self.root_dir,
                                self.file_paths[idx])

        img = custom_pil_loader(img_path)
        img = self.transforms(img)
        return img

class KnnQueryManager:
    def __init__(self,config,root_dir,transforms,model,device='cuda'):
        self.root_dir = root_dir
        self.config = config
        self.transforms = transforms
        self.file_paths = os.listdir(root_dir)
        self.device = device
        model.eval()
        self.model = model.to(device)
        projections = []

        dataset = TestLoaderForKnn(root_dir,transforms)
        dataloader = DataLoader(dataset,batch_size=200,num_workers=4,pin_memory=True,shuffle=False)
        print('Creating index')
        with torch.no_grad():
            for img in tqdm(dataloader):
                img = img.to(device)
                projection = self.model(img)
                projections.append(projection)
        self.projections = torch.cat(projections)
    

    def query(self,img_path,k=1,verbose = True):
        img = custom_pil_loader(img_path)
        img = self.transforms(img).to(self.device).unsqueeze(0)
        embedding = self.model(img).squeeze()
        dist = torch.linalg.norm(self.projections - embedding, dim=-1)
        knn = dist.topk(k, largest=False)
        indices = knn[1].tolist()
        paths = [os.path.join(self.root_dir,self.file_paths[ind]) for ind in indices]
        return knn,paths

    def __len__(self):
        return len(self.file_paths)

        

        
