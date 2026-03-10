"""
dataset/Twitter.py

TwitterDataset for Twitter15 / Twitter17 multimodal sentiment classification.
Modalities: text (BERT tokens) + image (ResNet).

Mirrors CramedDataset interface:
    __getitem__ returns (input_ids, attention_mask, token_type_ids, image, label)

TSV format (TomBERT / Twitter15 standard, tab-separated):
    col 0: sample index
    col 1: sentiment label  (0=negative, 1=neutral, 2=positive)
    col 2: image filename   (no extension, e.g. "107736")
    col 3: tweet text with opinion target masked as $T$
    col 4: opinion target (entity)

Expected directory layout:
    data_root/
    ├── annotations/
    │   ├── train.tsv
    │   ├── dev.tsv
    │   └── test.tsv
    └── twitter2015_images/   (set image_dir in config to override)
        └── *.jpg / *.png
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import BertTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TwitterDataset(Dataset):

    LABEL_MAP = {
        "0": 0, "1": 1, "2": 2,
        "negative": 0, "neutral": 1, "positive": 2,
    }

    def __init__(self, config, mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode   = mode

        ds_cfg         = config["dataset"]
        self.data_root = ds_cfg["data_root"]
        self.max_len   = ds_cfg.get("max_len", 128)
        bert_path      = ds_cfg.get("bert_path", "bert-base-uncased")
        image_dir      = ds_cfg.get("image_dir", "twitter2015_images")

        # image_dir can be an absolute path OR a folder name relative to data_root
        if os.path.isabs(image_dir):
            self.image_root = image_dir
        else:
            self.image_root = os.path.join(self.data_root, image_dir)

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        split_map = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"}
        assert mode in split_map, f"mode must be one of {list(split_map)}"

        # TSV files can sit directly in data_root OR in data_root/annotations/
        direct_path = os.path.join(self.data_root, split_map[mode])
        annot_path  = os.path.join(self.data_root, "annotations", split_map[mode])
        tsv_path    = direct_path if os.path.exists(direct_path) else annot_path

        # ---- load samples ------------------------------------------------
        self.samples = []   # (image_path, text, label_int)

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                # skip header row
                if i == 0:
                    continue
                if len(row) < 5:
                    continue

                label_raw  = row[1].strip()
                image_id   = row[2].strip()   # already has extension e.g. "1860693.jpg"
                tweet_text = row[3].strip()
                entity     = row[4].strip()

                label = self.LABEL_MAP.get(label_raw)
                if label is None:
                    continue

                # image_id may or may not include extension
                if os.path.splitext(image_id)[1]:
                    # extension already present (e.g. "1860693.jpg")
                    image_path = os.path.join(self.image_root, image_id)
                    if not os.path.exists(image_path):
                        continue
                else:
                    # no extension — try common ones
                    image_path = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        candidate = os.path.join(self.image_root, image_id + ext)
                        if os.path.exists(candidate):
                            image_path = candidate
                            break
                    if image_path is None:
                        continue

                # replace $T$ placeholder with actual entity
                text = tweet_text.replace("$T$", entity)
                self.samples.append((image_path, text, label))

        print(f"[TwitterDataset] {mode}: {len(self.samples)} samples loaded.")

        # ---- image transforms --------------------------------------------
        if self.mode == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text, label = self.samples[idx]

        # ---- text modality -----------------------------------------------
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)        # (max_len,)
        attention_mask = enc["attention_mask"].squeeze(0)   # (max_len,)
        token_type_ids = enc["token_type_ids"].squeeze(0)   # (max_len,)

        # ---- image modality ----------------------------------------------
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)                       # (3, 224, 224)

        # ---- one-hot label (matches CramedDataset) ----------------------
        num_classes    = self.config["setting"]["num_class"]
        one_hot        = np.eye(num_classes)[label]
        label_tensor   = torch.FloatTensor(one_hot)

        return input_ids, attention_mask, token_type_ids, image, label_tensor