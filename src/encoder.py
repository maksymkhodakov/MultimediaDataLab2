"""
Encoder: згорткова нейронна мережа ResNet-50, попередньо навчена на ImageNet.

Ми відкидаємо два останні шари ResNet-50 (глобальне усереднення та
класифікатор на 1000 класів ImageNet) і беремо вихід останнього
згорткового блоку — тензор 2048 × 7 × 7. Це 49 ділянок зображення,
кожна з яких описана вектором із 2048 чисел.

Ознаки обчислюються ОДИН раз для всіх 8000 зображень і зберігаються на
диск. Під час навчання декодера ResNet не змінюється (заморожена), тому
немає сенсу проганяти через неї зображення в кожній епосі.
"""
import time

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from . import config, data


class ResNetEncoder(nn.Module):
    """ResNet-50 без усереднення та класифікатора: зображення 3×224×224 → 49 векторів по 2048."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # children()[:-2] — усі шари, крім AdaptiveAvgPool2d та Linear (fc).
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.eval()  # режим інференсу: BatchNorm використовує накопичену статистику
        for p in self.parameters():
            p.requires_grad_(False)  # ваги енкодера не навчаємо

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        fmap = self.backbone(images)                 # (B, 2048, 7, 7)
        return fmap.flatten(2).transpose(1, 2)       # (B, 49, 2048): 49 ділянок × 2048 ознак


def image_transform() -> transforms.Compose:
    """
    Попередня обробка зображення так само, як під час навчання ResNet на ImageNet:
    зміна розміру до 224×224, перетворення у тензор [0, 1] та нормалізація
    середніми й стандартними відхиленнями каналів ImageNet.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def features_path(split: str):
    return config.FEATURES_DIR / f"{split}.npy"


@torch.no_grad()
def extract_split(split: str, encoder: ResNetEncoder, device: torch.device, n_images: int) -> None:
    """
    Проганяє всі зображення частини датасету через ResNet-50 і записує ознаки
    у файл .npy формату (N, 49, 2048). Зберігаємо у float16 — це вдвічі менше
    місця на диску (≈1.2 ГБ для train), а точності для навчання декодера достатньо.
    """
    tf = image_transform()
    out = np.lib.format.open_memmap(features_path(split), mode="w+", dtype=np.float16,
                                    shape=(n_images, config.NUM_REGIONS, config.FEATURE_DIM))
    pos = 0
    for images in data.iter_images(split, batch_size=64):
        batch = torch.stack([tf(img) for img in images]).to(device)
        feats = encoder(batch).float().cpu().numpy().astype(np.float16)
        out[pos:pos + len(images)] = feats
        pos += len(images)
    out.flush()
    assert pos == n_images, f"{split}: очікувалось {n_images} зображень, прочитано {pos}"


def extract_all(device: torch.device, sizes: dict[str, int]) -> None:
    """Обчислює ознаки для train/validation/test (пропускає частини, для яких файл уже існує)."""
    todo = [s for s in config.SPLITS if not features_path(s).exists()]
    if not todo:
        return
    encoder = ResNetEncoder().to(device)
    start = time.time()
    for split in todo:
        extract_split(split, encoder, device, sizes[split])
    print(f"Ознаки ResNet-50 обчислено за {time.time() - start:.0f} с")


def load_features(split: str) -> torch.Tensor:
    """Завантажує ознаки частини датасету в оперативну пам'ять як тензор float16 (N, 49, 2048)."""
    return torch.from_numpy(np.load(features_path(split)))


@torch.no_grad()
def encode_pil(image, encoder: ResNetEncoder, device: torch.device) -> torch.Tensor:
    """Ознаки одного довільного зображення (наприклад, власного фото): (1, 49, 2048)."""
    x = image_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    return encoder(x).float()
