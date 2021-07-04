from PIL import Image

def custom_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")