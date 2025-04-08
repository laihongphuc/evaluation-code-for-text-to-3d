import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm 
except ImportError:
    def tqdm(x):
        return x

from eval_3d.src.model import get_feature_extractor, get_static_from_dataloader,\
    get_feature_from_dataloader, get_probs_from_dataloader

from eval_3d.src.metric import fid_from_stats, clip_score_compute,\
    inception_variety_from_probs, inception_gain_from_probs, mmd



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

def cache_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".npz"):
            return os.path.join(folder_path, filename)
    return False


def inception_gain_helper_function(
    model: torch.nn.Module,
    generate_image_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda:0"
) -> float: 
    model = model.to(device)
    model.eval()

    generate_dataset = ImageDataset(
        generate_image_dir, None, 
        transform=model.image_preprocessor
    )

    generate_dataloader = DataLoader(
        generate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    generate_dataloader = tqdm(generate_dataloader)

    image_probs = get_probs_from_dataloader(model, generate_dataloader, device)
    score = inception_gain_from_probs(image_probs)
    return score

def clip_score_helper_function(
    clip_model,
    text_prompt: str,
    generate_image_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda:0"
) -> float:
    clip_model = clip_model.to(device)
    clip_model.eval()

    generate_dataset = ImageDataset(
        generate_image_dir, None, 
        transform=clip_model.image_preprocessor
    )

    generate_dataloader = DataLoader(
        generate_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    text_features = clip_model.encode_text(text_prompt)
    image_features = get_feature_from_dataloader(clip_model, generate_dataloader, device)
    score = clip_score_compute(image_features, text_features)
    return score
    
def cmmd_score_helper_function(
    clip_model,
    generate_image_dir: str,
    real_image_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda:0"
) -> float:  
    clip_model = clip_model.to(device)
    clip_model.eval()
    # dataset init
    generate_dataset = ImageDataset(
        generate_image_dir, None, 
        transform=clip_model.image_preprocessor,
    )

    real_dataset = ImageDataset(
        real_image_dir, None, 
        transform=clip_model.image_preprocessor,
        split_image=1
    )

    generate_dataloader = DataLoader(
        generate_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    real_dataloader = DataLoader(
        real_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    real_dataloader = tqdm(real_dataloader)
    generate_dataloader = tqdm(generate_dataloader)
    generate_image_features = get_feature_from_dataloader(clip_model, generate_dataloader, device)
    real_image_features = get_feature_from_dataloader(clip_model, real_dataloader, device)
    score = mmd(generate_image_features, real_image_features)
    return score

def fid_score_helper_function(
    model,
    generate_image_dir: str,
    real_image_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "cuda:0"
) -> float:
    model = model.to(device)
    model.eval()

    generate_dataset = ImageDataset(
        generate_image_dir, None, 
        transform=model.image_preprocessor,
    )

    if real_image_dir is not None:
        real_dataset = ImageDataset(
            real_image_dir, None, 
            transform=model.image_preprocessor,
            split_image=1
        )
    else:
        real_dataset = None

    generate_dataloader = DataLoader(
        generate_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    generate_dataloader = tqdm(generate_dataloader)
    if real_dataset is not None:
        real_dataloader = DataLoader(
            real_dataset, 
            batch_size=batch_size, 
            num_workers=num_workers,
            shuffle=False
        )
        real_dataloader = tqdm(real_dataloader)
    cache_fake, cache_real = cache_folder(generate_image_dir), cache_folder(real_image_dir)
    # cache_fake, cache_real = False, False
    if cache_real is False:
        mean_real, std_real = get_static_from_dataloader(model, real_dataloader, device)
        np.savez(os.path.join(real_image_dir, "stats.npz"), mean=mean_real, std=std_real)
    else:
        mean_real, std_real = np.load(cache_real)["mean"], np.load(cache_real)["std"]
    if cache_fake is False:
        mean_fake, std_fake = get_static_from_dataloader(model, generate_dataloader, device)
        np.savez(os.path.join(generate_image_dir, "stats.npz"), mean=mean_fake, std=std_fake)
    else:
        mean_fake, std_fake = np.load(cache_fake)["mean"], np.load(cache_fake)["std"]

    score = fid_from_stats(mean_real, std_real, mean_fake, std_fake)
    return score