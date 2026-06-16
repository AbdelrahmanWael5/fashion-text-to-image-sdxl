import os
import random
import argparse
import json
import itertools
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Literal, Tuple,List
import torch.utils.data as data
import math
from tqdm.auto import tqdm
from diffusers.training_utils import compute_snr
import torchvision.transforms.functional as TF


class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        captions_file: str,
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size
        self.order = order

        self.norm = transforms.Normalize([0.5], [0.5])

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        captions_path = os.path.join(
            dataroot_path,
            phase,
            captions_file
        )

        self.annotation_pair = {}
        self.im_names = []
        self.c_names = []

        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
    
                file_name = item["file_name"]
                caption = item["raw_caption"]
    
                self.annotation_pair[file_name] = caption
    
                self.im_names.append(file_name)
                self.c_names.append(file_name)
        
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
        
    def __getitem__(self, index):
        c_name = self.c_names[index]
        
        cloth_annotation = self.annotation_pair[c_name]
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name)).convert("RGB")
        cloth = cloth.resize((self.width, self.height))

        
        if self.phase == "train":
            if random.random() > 0.5:
                cloth = self.flip_transform(cloth)


            if random.random()>0.5:
                color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.5)
                fn_idx, b, c, s, h = transforms.ColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,color_jitter.hue)
                
                cloth = TF.adjust_contrast(cloth, c)
                cloth = TF.adjust_brightness(cloth, b)
                cloth = TF.adjust_hue(cloth, h)
                cloth = TF.adjust_saturation(cloth, s)


        result = {}
        result["c_name"] = c_name
        result["cloth"] = self.transform(cloth)  # cloth image for the ClothNet
        result["cloth_captions"] = cloth_annotation

        return result

    def __len__(self):
        return len(self.im_names)