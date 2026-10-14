"""
Робота з датасетом Flickr8K: завантаження, токенізація підписів, словник.

Датасет зберігається у форматі parquet (копія jxie/flickr8k з Hugging Face Hub).
Кожен рядок таблиці — одне зображення (JPEG-байти) та 5 підписів англійською.
"""
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from . import config

CAPTION_COLUMNS = [f"caption_{i}" for i in range(config.CAPTIONS_PER_IMAGE)]


# ---------------------------------------------------------------------------
# Завантаження датасету
# ---------------------------------------------------------------------------
def download_dataset() -> None:
    """Завантажує parquet-файли Flickr8K з Hugging Face Hub (~1.1 ГБ), якщо їх ще немає."""
    if split_files("train"):
        return
    from huggingface_hub import snapshot_download
    snapshot_download(config.HF_REPO, repo_type="dataset", local_dir=config.DATASET_DIR)


def split_files(split: str) -> list[Path]:
    """Повертає відсортований список parquet-файлів для частини датасету (train/validation/test)."""
    return sorted((config.DATASET_DIR / "data").glob(f"{split}-*.parquet"))


def load_captions(split: str) -> list[list[str]]:
    """
    Читає лише текстові стовпці (без зображень — так швидше і менше пам'яті).
    Повертає список довжини N (кількість зображень), де кожен елемент —
    список із 5 «сирих» підписів.
    """
    captions = []
    for path in split_files(split):
        table = pq.read_table(path, columns=CAPTION_COLUMNS).to_pydict()
        captions.extend(zip(*(table[c] for c in CAPTION_COLUMNS)))
    return [list(c) for c in captions]


def iter_images(split: str, batch_size: int = 64):
    """
    Генератор, що по черзі віддає пакети зображень частини датасету
    у вигляді списків PIL.Image. Порядок збігається з load_captions(split).
    Читаємо parquet пакетами, щоб не тримати в пам'яті всі 800 МБ JPEG-ів.
    """
    for path in split_files(split):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["image"]):
            yield [decode_image(item["bytes"]) for item in batch.column("image").to_pylist()]


def load_images(split: str, indices: list[int]) -> list[Image.Image]:
    """Завантажує конкретні зображення за їхніми номерами (для візуалізації прикладів)."""
    wanted = set(indices)
    found, offset = {}, 0
    for path in split_files(split):
        column = pq.read_table(path, columns=["image"]).column("image")
        for i in range(len(column)):
            if offset + i in wanted:
                found[offset + i] = decode_image(column[i].as_py()["bytes"])
        offset += len(column)
    return [found[i] for i in indices]


def decode_image(raw: bytes) -> Image.Image:
    """Перетворює JPEG-байти на RGB-зображення PIL."""
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ---------------------------------------------------------------------------
# Токенізація та словник
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Розбиває підпис на слова: нижній регістр, лише латинські літери (цифри й розділові знаки відкидаємо)."""
    return re.findall(r"[a-z]+", text.lower())


class Vocabulary:
    """
    Словник «слово ↔ індекс».

    Службові токени:
      <pad> — доповнення коротких підписів до однакової довжини в пакеті (індекс 0);
      <s>   — початок речення (з нього декодер починає генерацію);
      </s>  — кінець речення (декодер зупиняється, коли згенерує цей токен);
      <unk> — будь-яке рідкісне слово, якого немає у словнику.
    """
    PAD, START, END, UNK = "<pad>", "<s>", "</s>", "<unk>"

    def __init__(self, words: list[str]):
        self.itos = [self.PAD, self.START, self.END, self.UNK] + words  # index → string
        self.stoi = {w: i for i, w in enumerate(self.itos)}           # string → index
        self.pad_id, self.start_id, self.end_id, self.unk_id = 0, 1, 2, 3

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, captions: list[list[str]], min_freq: int = config.MIN_WORD_FREQ):
        """
        Будує словник за навчальними підписами: залишає слова, що трапились
        щонайменше min_freq разів. Повертає (словник, лічильник частот усіх слів).
        """
        counter = Counter(w for caps in captions for c in caps for w in tokenize(c))
        # Сортуємо за спаданням частоти (а при рівності — за абеткою) для стабільного порядку.
        words = sorted((w for w, n in counter.items() if n >= min_freq), key=lambda w: (-counter[w], w))
        return cls(words), counter

    def encode(self, tokens: list[str], max_len: int = config.MAX_CAPTION_LEN) -> list[int]:
        """Перетворює список слів на індекси: <s> w1 ... wn </s> (з обрізанням до max_len слів)."""
        ids = [self.stoi.get(w, self.unk_id) for w in tokens[:max_len]]
        return [self.start_id] + ids + [self.end_id]

    def decode(self, ids) -> list[str]:
        """Перетворює індекси назад на слова, зупиняючись на </s> і пропускаючи службові токени."""
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_id:
                break
            if i in (self.pad_id, self.start_id):
                continue
            words.append(self.itos[i])
        return words

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.itos[4:], ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8")))


def encode_split(captions: list[list[str]], vocab: Vocabulary,
                 max_len: int = config.MAX_CAPTION_LEN):
    """
    Готує навчальні пари «зображення — підпис» у вигляді тензорів:
      image_index[k] — номер зображення для k-го підпису;
      tokens[k]      — індекси слів k-го підпису, доповнені <pad> до довжини max_len + 2;
      lengths[k]     — справжня довжина k-го підпису разом із <s> та </s>.
    Кожне зображення дає 5 пар (по одній на кожен людський підпис).
    """
    image_index, encoded = [], []
    for img_i, caps in enumerate(captions):
        for c in caps:
            image_index.append(img_i)
            encoded.append(vocab.encode(tokenize(c), max_len))
    lengths = torch.tensor([len(e) for e in encoded])
    tokens = torch.full((len(encoded), max_len + 2), vocab.pad_id, dtype=torch.long)
    for k, e in enumerate(encoded):
        tokens[k, :len(e)] = torch.tensor(e)
    return torch.tensor(image_index), tokens, lengths


def references(captions: list[list[str]]) -> list[list[list[str]]]:
    """Еталонні підписи для метрики BLEU: для кожного зображення — 5 токенізованих речень."""
    return [[tokenize(c) for c in caps] for caps in captions]
