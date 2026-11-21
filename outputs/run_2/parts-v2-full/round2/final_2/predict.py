"""Recreate a parts-v2 submission from a state_dict and its model settings."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SmallCNN(nn.Module):
    """Teacher's baseline, with an optional delay of the first pooling step."""

    def __init__(self, late_pool=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Identity() if late_pool else nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4 if late_pool else 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Linear(64, 8)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.conv(x))


class ShapeCNN(nn.Module):
    """Two convolutions before the first pool, with 2×2 spatial pooling at the end."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 32), nn.MaxPool2d(2),
            conv_block(32, 48), ResidualBlock(48), nn.MaxPool2d(2),
            conv_block(48, 80), ResidualBlock(80), nn.MaxPool2d(2),
            conv_block(80, 128), ResidualBlock(128),
        )
        self.head = nn.Sequential(nn.Dropout(0.15), nn.Linear(128 * 4, 8))

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.shape[0], 128, 2, 4, 2, 4).mean(dim=(3, 5))
        return self.head(x.flatten(1))


def create_model(architecture, imagenet_state=None):
    if architecture in {"small", "late_pool"}:
        return SmallCNN(late_pool=architecture == "late_pool")
    if architecture == "shape_cnn":
        return ShapeCNN()
    if architecture == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(weights=None)
        if imagenet_state is not None:
            model.load_state_dict(imagenet_state)
        model.fc = nn.Linear(model.fc.in_features, 8)
        return model
    raise ValueError(f"Unknown architecture: {architecture}")


def image_to_tensor(image, architecture, image_size, grayscale=False):
    if grayscale:
        image = ImageOps.grayscale(image).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255)
    if architecture == "resnet18":
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return x.sub_(mean).div_(std)
    return x.sub_(0.5).div_(0.5)


class PartsDataset(Dataset):
    def __init__(self, root, rows, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform
        self.labels = self.rows["label"].tolist() if "label" in self.rows else [-1] * len(rows)
        self.images = []
        for file in self.rows["file"]:
            with Image.open(Path(root) / file) as image:
                self.images.append(image.convert("RGB"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


def predict(data, weights, settings, output):
    data = Path(data)
    cfg = json.loads(Path(settings).read_text(encoding="utf-8"))
    if cfg["dataset_version"] != "parts-v2" or not 1 <= cfg["image_size"] <= 128:
        raise ValueError("Expected parts-v2 and an input size no larger than 128.")
    # Fixed CPU/FP32 inference for both export and offline verification.
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = create_model(cfg["architecture"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()

    test = pd.read_csv(data / "test.csv", dtype={"id": str})
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": str})
    for frame in (test, sample):
        if len(frame) != 960 or frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError("Expected 960 unique IDs in parts-v2 test and sample files.")
    if set(test["id"]) != set(sample["id"]):
        raise ValueError("Test and sample submission IDs differ.")

    def transform(image):
        return image_to_tensor(image, cfg["architecture"], cfg["image_size"], cfg["grayscale"])

    dataset = PartsDataset(data, test, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with torch.inference_mode():
        predictions = [model(images).argmax(dim=1) for images, _ in loader]
    submission = pd.DataFrame({"id": test["id"], "label": torch.cat(predictions).numpy()})
    submission = submission.set_index("id").loc[sample["id"]].reset_index()
    if list(submission.columns) != ["id", "label"] or not submission["label"].between(0, 7).all():
        raise ValueError("Expected id,label with integer class IDs in [0, 7].")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, lineterminator="\n")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory with the parts-v2 CSVs and images/")
    parser.add_argument("--weights", default="model.pt")
    parser.add_argument("--settings", default="model_config.json")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    predict(args.data, args.weights, args.settings, args.output)
