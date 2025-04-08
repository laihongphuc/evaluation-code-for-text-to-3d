import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm 
except ImportError:
    def tqdm(x):
        return x

from src.model import get_feature_extractor, get_static_from_dataloader, get_feature_from_dataloader

from src.metric import fid_from_stats, clip_score_compute




IMG_FORMAT = ["png", "jpg"]

def cache_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            return os.path.join(folder_path, filename)
    return False


class ImageDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 number_of_images: int,
                 transform: callable = None,
                 split_image: int = 2) -> None:
        super().__init__()
        self.split_image = split_image
        self.images_dir = images_dir 
        # random number of images 
        images_list = [os.path.join(images_dir, name) for name in os.listdir(self.images_dir)] 
        images_list = [path for path in images_list if path.split(".")[-1] in IMG_FORMAT]
        if number_of_images is not None:
            self.images_list = random.sample(images_list, number_of_images)
        else:
            self.images_list = images_list
        self.transform = transform


    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.images_list[index]
        img = Image.open(img_path)
        img = np.array(img)
        h, w, c = img.shape 
        img = img[:, :w//self.split_image, :]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        else: 
            img = None 
        return img
    

def clip_score_helper_function(
        model: torch.nn.Module,
        text_prompt: str, 
        generate_image_dir: str,
        device: str
) -> float: 
    model = model.to(device)
    model.eval()

    generate_dataset = ImageDataset(
        generate_image_dir, None, 
        transform=model.image_preprocessor
    )

    generate_dataloader = DataLoader(
        generate_dataset,
        batch_size=50,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    text_features = model.encode_text(text_prompt)
    image_features = get_feature_from_dataloader(model, generate_dataloader, device)
    score = clip_score_compute(image_features, text_features)
    return score


if __name__ == "__main__":
    df_path = "Text-to-3d-metric.csv"
    df = pd.read_csv(df_path)
    df.fillna("0", inplace=True)

    method = ["SDS", "VSD", "mvdream-baseline", "mvdream-triplane", "mvdream-tensoRF", "mvdream-ms-tensoRF"]
    method_score = []
    
    model = get_feature_extractor("clip", pretrained=True)
    device = torch.device('cuda')

    for m in method:
        current_score = 0
        count = 0
        current_df = df[df[m] != "0"][[m, "Prompt"]]
        for index, row in current_df.iterrows():
            score = clip_score_helper_function(model, row["Prompt"], row[m], device)
            current_score += score 
            count += 1

        method_score.append(current_score / count)
        
    for m, s in zip(method, method_score):
        print(f"{m}: {s}")