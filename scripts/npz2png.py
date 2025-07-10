"""
Convert a .npz file into a directory of .png files
"""

import argparse
import os

import torch as th
"""
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
"""
from guided_diffusion import logger
"""
from guided_diffusion.script_util_x0 import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
"""

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset

import numpy as np
from PIL import Image

def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path): # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, label_file='./imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    input_file_default  = os.path.join('../scripts/imagenet_dataloader', 'VIRTUAL_imagenet256_labeled.npz')
    parser.add_argument("--input_file", default=input_file_default, type=str, help='the npz file to convert.')
    parser.add_argument("--save_dir", default=None, type=str, help='the directory to save the generate images')
    args = parser.parse_args()

    logger.log("loading dataset...")
    # load .npz of images
    dataset_lr = get_dataset(path=args.input_file, global_rank=0, world_size=1)
    dataloader_lr = th.utils.data.DataLoader(dataset_lr, batch_size=1, shuffle=False, num_workers=16)

    save_dir = args.save_dir if args.save_dir else args.input_file.replace(".npz", "")
    os.makedirs(save_dir, exist_ok=True)
    logger.log("Saving to", save_dir)

    # args.save_png_files=True
    print(logger.get_dir())

    logger.log("sampling...")
    # while len(all_images) * args.batch_size < args.num_samples:
    for i, data in enumerate(dataloader_lr):
        image, label = data
        image = ((image + 1) * 127.5).clamp(0, 255).to(th.uint8)
        image = image.permute(0, 2, 3, 1)
        image = image.squeeze(0)
        image = image.contiguous()
        image = image.detach().cpu().numpy()
        save_filename = f'imagenet256_clean_{i}.png'
        save_path = os.path.join(save_dir, save_filename)
        Image.fromarray(image).save(save_path)
        #save_images(image, label.long(), i, save_dir=save_dir)
        logger.log("Saved image", save_path)
     
        #save_images(image, label.long(), start_idx + len(all_images) * args.batch_size, save_dir=os.path.join(logger.get_dir(), 'lr'))
        #logger.log(f"created {len(all_images) * args.batch_size} samples")

    # dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
