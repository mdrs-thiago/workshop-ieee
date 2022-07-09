from tkinter import Image
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random 
import numpy as np 

from transformers import ViTFeatureExtractor, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, AutoAugment, AutoAugmentPolicy

import torch
from torchvision.datasets import CIFAR10

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path, data_augmentation = False):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        transform = []

        if data_augmentation:
            transform.append(AutoAugment(AutoAugmentPolicy.CIFAR10))

        if feature_extractor.do_resize:
            transform.append(Resize((feature_extractor.size, feature_extractor.size)))

        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))


        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)



def create_CIFAR10_loader(model_name, batch_size = 24, num_workers = 2, ID=False):
  val_loader = DataLoader( 
      CIFAR10('./', download=True, train=False, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
  )

  if ID:
    train_loader = DataLoader( 
      CIFAR10('./', download=True, transform=ViTFeatureExtractorTransforms(model_name)),
      batch_size=batch_size,
      num_workers=num_workers,
      pin_memory=True,
      collate_fn=my_collate
    )
    return train_loader, val_loader
  
  return val_loader
