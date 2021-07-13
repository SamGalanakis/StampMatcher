import requests
import os
from tqdm import tqdm


out_folder =r'D:\data\stamps'

with open('data\stamps\linkList.txt') as f:
    content = f.readlines()
content = [x.strip() for x in content] 


for line in tqdm(content):
    id,url = line.split(',')
    
    filename, file_extension = os.path.splitext(os.path.basename(url))
    write_path = os.path.join(out_folder,id) + file_extension
    if os.path.isfile(write_path):
        continue
    response = requests.get(url)
    
    
    with open(write_path, "wb") as f:
        f.write(response.content)
