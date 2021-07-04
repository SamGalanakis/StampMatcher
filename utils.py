from PIL import Image
import yaml


def custom_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def put_on_device(x,device):
    try:
        x = x.to(device)
    except:
        pass
    return x

def config_loader(path):
    """Load yaml config"""
    with open(path) as f:
        raw_dict = yaml.load(f,Loader=yaml.FullLoader)
    return {key: raw_dict[key]['value'] for key in raw_dict.keys()}
        