import torch
from torch.utils.data import Dataset
import os
import io
import random
from utils import custom_pil_loader
import numpy as np

class StampDataset(Dataset):
    def __init__(self,root_dir,transforms,mode = 'train'):
        super().__init__()
        self.root_dir = root_dir
        self.transforms = transforms
        self.file_paths = os.listdir(root_dir)



    def __len__(self):
        return len(self.file_paths)


    def __getitem__(self,idx):
        main_img_path = os.path.join(self.root_dir,
                                self.file_paths[idx])

        valid_samples = list(range(self.__len__()))
        valid_samples.pop(idx)
        negative_example_idx = random.sample(valid_samples,1)[0]
        nagative_img_path = os.path.join(self.root_dir,
                                self.file_paths[negative_example_idx])
        
        main_image = custom_pil_loader(main_img_path)
        nagative_image = custom_pil_loader(nagative_img_path)
        anchor = self.transforms(main_image)
        positive = self.transforms(main_image)
        negative = self.transforms(nagative_image)

        return anchor,positive,negative
        

        
