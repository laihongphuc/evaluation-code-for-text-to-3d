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

from src.model import get_feature_extractor, get_static_from_dataloader,\
    get_feature_from_dataloader, get_probs_from_dataloader

from src.metric import fid_from_stats, clip_score_compute,\
    inception_variety_from_probs, inception_gain_from_probs




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


def inception_gain_helper_function(
    model: torch.nn.Module,
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

    generate_dataloader = tqdm(generate_dataloader)

    image_probs = get_probs_from_dataloader(model, generate_dataloader, device)
    score = inception_gain_from_probs(image_probs)
    return score


def fid_score_helper_function(
    model: torch.nn.Module,
    generate_image_dir: str,
    real_image_dir: str,
    device: str
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
        batch_size=50, 
        num_workers=8,
        shuffle=False
    )
    generate_dataloader = tqdm(generate_dataloader)
    if real_dataset is not None:
        real_dataloader = DataLoader(
            real_dataset, 
            batch_size=50, 
            num_workers=8,
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


if __name__ == "__main__":
    # df_path = "/root/Code/Text-to-3D-Metric.csv"
    # df = pd.read_csv(df_path)
    # df.fillna("0", inplace=True)

    # method = ["SDS", "VSD", "mvdream-baseline", "mvdream-triplane", "mvdream-tensoRF", "mvdream-ms-tensoRF"]
    # method = ["mvdream-ms-tensoRF"]
    prompt_list = {
    "a blue motorcycle",
    "a delicious hamburger",
    "a zoomed out DSLR photo of a 3d model of an adorable cottage with a thatched roof",
    "a zoomed out DSLR photo of a beautifully carved wooden knight chess piece",
    "a tiger karate master",
    "a zoomed out DSLR photo of a monkey riding a bike",
    "a DSLR photo of a fox holding a videogame controller",
    "Michelangelo style statue of dog reading news on a cellphone",
    "a chimpanzee with a big grin",
    "a DSLR photo of a corgi puppy",
    "a zoomed out DSLR photo of a squirrel dressed up like a Victorian woman",
    "a pig wearing a backpack",
    "a DSLR photo of a koala wearing a party hat and blowing out birthday candles on a cake",
    "a zoomed out DSLR photo of a chimpanzee holding a cup of hot coffee",
    "a DSLR photo of a quill and ink sitting on a desk",
    "a zoomed out DSLR photo of a pig playing the saxophone",
    "a DSLR photo of a toy robot",
    "a zoomed out DSLR photo of a beautifully carved wooden knight chess piece",
    "a plush toy of a corgi nurse",
    "an astronaut riding a horse",
    "A beautiful dress made out of garbage bags, on a mannequin. Studio lighting, high quality, high resolution.",
    "A blue poison-dart frog sitting on a water lily.",
    "A DSLR photo of a car made out of sushi.",
    "A DSLR photo of a bagel filled with cream cheese and lox.",
    "A DSLR photo of an ice cream sundae.",
    "A DSLR photo of a peacock on a surfboard.",
    "A DSLR photo of a plate piled high with chocolate chip cookies.",
    "A DSLR photo of Neuschwanstein Castle, aerial view.",
    "A DSLR photo of the Imperial State Crown of England.",
    "A DSLR photo of the leaning tower of Pisa, aerial view.",
    "A ripe strawberry.",
    "A silver platter piled high with fruits.",
    "A DSLR photo of a silver candelabra sitting on a red velvet tablecloth, only one candle is lit.",
    "A DSLR photo of Sydney opera house, aerial view.",
    "Michelangelo style statue of an astronaut.",
    }
    method = ["luciddreamer"]
    prompt_dict = {i+1: prompt_list[i] for i in range(len(prompt_list))}
    method_score = []
    
    # model = get_feature_extractor("clip", pretrained=True)
    model = get_feature_extractor("fid", pretrained=True)
    device = torch.device('cuda')
    path = "/home/phuclh/Downloads/Survey/LucidDreamer/output" 
    for m in method:
        current_score = 0
        count = 0
        # current_df = df[df[m] != "0"][[m, "Prompt", "real image"]]
        # for index, row in current_df.iterrows():
        #     real_dir = row["real image"]
        #     print(row["Prompt"])
        #     # score = clip_score_helper_function(model, row["Prompt"], "/root/"+row[m], device)
        #     # score = fid_score_helper_function(model, "/root/"+row[m], "/root/"+real_dir, device)
        #     score = inception_gain_helper_function(model, "/root/"+row[m], device)
        #     current_score += score 
        #     print(score)
        #     count += 1
        for file in os.listdir(path):
            i = int(file.replace("prompt", ""))
            video_file = os.path.join(path, file) 
            output_folder = os.path.join(path, file) + "/images_test"
            score 

        method_score.append((current_score / count, count))
        
    for m, s in zip(method, method_score):
        print(f"{m}: total prompt={s[1]} score={s[0]}")