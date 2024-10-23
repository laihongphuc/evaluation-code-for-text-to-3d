import os
import os.path as osp
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

from src.model import get_feature_extractor, get_static_from_dataloader, get_feature_from_dataloader

from src.metric import fid_from_stats, clip_score_compute


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--metric', type=str, default='fid',
                    choices=['fid', 'clip'], help='Metric to use')
parser.add_argument('--generate-image-dir', type=str, help='Path to inference image from 3Dstudio')
parser.add_argument('--real-image-dir', type=str, default=None, help='Path to real images')
parser.add_argument('--text', type=str, default=None, help='Text to generate image')
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--split-image', type=int, default=2,
                    help='Split image into 2 or 3 parts based on SDS or VSD')
parser.add_argument('--device', type=str, default="cuda:0",
                    help='Device to use. Like cuda, cuda:0 or cpu')


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
    

def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    # model init
    model = get_feature_extractor(args.metric, pretrained=True)
    model = model.to(device)

    # dataset init
    generate_dataset = ImageDataset(
        args.generate_image_dir, None, 
        transform=model.image_preprocessor
    )

    if args.real_image_dir is not None:
        real_dataset = ImageDataset(
            args.real_image_dir, None, 
            transform=model.image_preprocessor
        )
    else:
        real_dataset = None

    generate_dataloader = DataLoader(
        generate_dataset, 
        batch_size=args.batch_size, 
        num_workers=num_workers,
        shuffle=False
    )
    generate_dataloader = tqdm(generate_dataloader)
    if real_dataset is not None:
        real_dataloader = DataLoader(
            real_dataset, 
            batch_size=args.batch_size, 
            num_workers=num_workers,
            shuffle=False
        )
        real_dataloader = tqdm(real_dataloader)

    # metric
    if args.metric == "clip":
        text_features = model.encode_text(args.text)
        image_features = get_feature_from_dataloader(model, generate_dataloader, device)
        score = clip_score_compute(image_features, text_features)
    elif args.metric == "fid":
        mean_real, std_real = get_static_from_dataloader(model, real_dataloader, device)
        mean_fake, std_fake = get_static_from_dataloader(model, generate_dataloader, device)
        score = fid_from_stats(mean_real, std_real, mean_fake, std_fake)
    else:
        raise NotImplementedError(f"Don't support model {args.metric}")
    
    print(f"Compute {args.metric} score: {score}")


if __name__ == "__main__":
    main()