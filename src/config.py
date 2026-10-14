"""
Глобальні налаштування лабораторної роботи №2
«Формування підписів до зображень (image captioning) на Flickr8K».

Тут зібрано всі шляхи до файлів, гіперпараметри навчання та опис
експериментів, щоб решта модулів не містила «магічних чисел».
"""
import random
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Шляхи
# ---------------------------------------------------------------------------
# Корінь проєкту — тека, у якій лежить main.py (на рівень вище за src/).
ROOT = Path(__file__).resolve().parent.parent

# Тека з даними (у .gitignore — великі файли не потрапляють у репозиторій).
DATA_DIR = ROOT / "data"
# Копія датасету з Hugging Face Hub (parquet-файли із зображеннями та підписами).
HF_REPO = "jxie/flickr8k"
DATASET_DIR = DATA_DIR / "flickr8k"
# Попередньо обчислені ознаки ResNet-50 для кожного зображення (*.npy).
FEATURES_DIR = DATA_DIR / "features"

# Результати: графіки, таблиці, словник, історії навчання, ваги моделей.
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
# Власні зображення студента (пункт 4 завдання).
OWN_IMAGES_DIR = ROOT / "own_images"

# ---------------------------------------------------------------------------
# Параметри даних
# ---------------------------------------------------------------------------
SPLITS = ("train", "validation", "test")  # стандартне розбиття 6000/1000/1000
CAPTIONS_PER_IMAGE = 5                     # у Flickr8K до кожного фото 5 підписів
MIN_WORD_FREQ = 5         # слово потрапляє у словник, якщо трапилось >= 5 разів
MAX_CAPTION_LEN = 20      # довші підписи обрізаємо (95 % підписів коротші за 19 слів)
IMAGE_SIZE = 224          # розмір входу ResNet-50 → карта ознак 7×7
FEATURE_DIM = 2048        # кількість каналів останнього згорткового блоку ResNet-50
NUM_REGIONS = 49          # 7 × 7 = 49 ділянок зображення

# ---------------------------------------------------------------------------
# Параметри навчання (спільні для всіх експериментів)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 4e-4      # швидкість навчання оптимізатора Adam
GRAD_CLIP = 5.0           # обмеження норми градієнта (захист від «вибуху» градієнтів у LSTM)
DROPOUT = 0.5             # ймовірність dropout перед вихідним шаром
ALPHA_C = 1.0             # вага регуляризації уваги («doubly stochastic attention», Xu et al.)
BEAM_SIZES = (3, 5)       # ширини променя для beam search, які порівнюємо з greedy
SEED = 42                 # зерно генератора випадкових чисел для відтворюваності

# ---------------------------------------------------------------------------
# Опис експериментів (пункт 3 завдання — варіанти з різними параметрами)
# ---------------------------------------------------------------------------
# attention_dim = None означає модель без уваги (Show and Tell).
EXPERIMENTS = {
    "A": dict(name="A. Show and Tell (LSTM 512)",
              embed_dim=512, hidden_dim=512, attention_dim=None),
    "B": dict(name="B. Show, Attend and Tell (LSTM 512)",
              embed_dim=512, hidden_dim=512, attention_dim=512),
    "C": dict(name="C. Show, Attend and Tell (LSTM 256)",
              embed_dim=256, hidden_dim=256, attention_dim=256),
}


def get_device() -> torch.device:
    """Обирає найшвидший доступний пристрій: CUDA (NVIDIA), MPS (Apple Silicon) або CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = SEED) -> None:
    """Фіксує зерна всіх генераторів випадкових чисел, щоб результати можна було відтворити."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    """Створює всі вихідні теки, якщо їх ще немає."""
    for d in (DATA_DIR, FEATURES_DIR, OUTPUT_DIR, FIGURES_DIR, CHECKPOINT_DIR, OWN_IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
