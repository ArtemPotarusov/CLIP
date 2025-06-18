import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

to_tensor = T.ToTensor()
import typing as tp


class CLIPDataset(Dataset):
    def __init__(self, image_path, image_filenames, captions, tokenizer):
        """
        :image_path -- path to images
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        :tokenizer -- LM Tokenizer 
        """
        self.max_tokenizer_length = 200
        self.truncation = True
        self.padding = True
        self.image_path = image_path
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.tokenizer = tokenizer
        self.encoded_captions = self.tokenizer(
            self.captions,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_tokenizer_length,
        )
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.454, 0.450, 0.412], std=[0.262, 0.2555, 0.273])  # computed on our dataset
        ])  # This should do.

    def __getitem__(self, idx: int) -> tp.Dict[str, tp.Union[torch.Tensor, str]]:
        """
        This one should return dict(keys=['image', 'caption'], value=[Image, Caption])
        """
        item = {
            key: torch.tensor(values[idx]).clone().detach() for key, values in self.encoded_captions.items()
        }
        img_path = self.image_path + '/images/' + self.image_filenames[idx]
        item['image'] = self.transforms(Image.open(img_path).convert("RGB"))
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)
