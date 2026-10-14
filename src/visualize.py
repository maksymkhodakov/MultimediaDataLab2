"""
Графіки та ілюстрації для звіту: статистика датасету, криві навчання,
приклади підписів та візуалізація уваги.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")  # малюємо у файли, без вікон
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from . import config

COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def _save(fig, name: str):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _wrap(text: str, width: int = 48) -> str:
    return "\n".join(textwrap.wrap(text, width))


def plot_dataset_stats(counter, vocab_words: list[str], lengths: list[int]):
    """Гістограма довжин підписів та 25 найчастіших слів навчальної вибірки."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.hist(lengths, bins=range(1, max(lengths) + 2), color="#1f77b4", edgecolor="white")
    ax1.axvline(np.percentile(lengths, 95), color="red", ls="--", label="95-й перцентиль")
    ax1.set(title="Довжина підписів (train)", xlabel="кількість слів", ylabel="кількість підписів")
    ax1.legend()
    top = vocab_words[:25]
    ax2.bar(top, [counter[w] for w in top], color="#2ca02c")
    ax2.set(title="25 найчастіших слів (train)", ylabel="частота")
    ax2.tick_params(axis="x", rotation=70)
    return _save(fig, "dataset_stats.png")


def plot_dataset_examples(images, captions):
    """Кілька зображень з датасету разом з усіма п'ятьма людськими підписами."""
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 6.2))
    for ax, img, caps in zip(axes, images, captions):
        # Обрізаємо до однакового розміру 4:3 (без спотворення пропорцій), щоб підписи були на одному рівні.
        ax.imshow(ImageOps.fit(img, (400, 300)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("\n".join(f"{i + 1}. " + _wrap(c.strip().rstrip(" ."), 36) for i, c in enumerate(caps)),
                      fontsize=11, loc="left", labelpad=8)
    return _save(fig, "dataset_examples.png")


def plot_training_curves(histories: dict):
    """Втрати на навчанні та BLEU-4 на валідації по епохах для всіх експериментів."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, h in histories.items():
        epochs = range(1, len(h["train_loss"]) + 1)
        ax1.plot(epochs, h["train_loss"], color=COLORS[key], label=h["name"])
        ax2.plot(epochs, h["val_bleu4"], color=COLORS[key], label=h["name"])
        ax2.scatter([h["best_epoch"]], [h["best_val_bleu4"]], color=COLORS[key], zorder=3)
    ax1.set(title="Втрати на навчанні (cross-entropy)", xlabel="епоха")
    ax2.set(title="BLEU-4 на валідації (greedy)", xlabel="епоха")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    return _save(fig, "training_curves.png")


def plot_captioned_grid(images, model_caps, human_caps, name: str, extra=None):
    """Сітка 3×2: зображення з підписом моделі та одним людським підписом."""
    cols = 2
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.6 * rows))
    for i, ax in enumerate(np.array(axes).flat):
        ax.axis("off")
        if i >= len(images):
            continue
        ax.imshow(ImageOps.fit(images[i], (500, 375)))  # однаковий розмір 4:3 для всіх кадрів
        title = f"модель: {_wrap(model_caps[i], 44)}\nлюдина: {_wrap(human_caps[i], 44)}"
        if extra:
            title += f"\n{extra[i]}"
        ax.set_title(title, fontsize=12.5)
    fig.tight_layout(h_pad=3.0)  # додатковий відступ, щоб довгі заголовки не наїжджали на фото
    return _save(fig, name)


def plot_attention(image: Image.Image, words: list[str], alphas: np.ndarray, name: str):
    """
    Для кожного згенерованого слова показуємо ваги уваги α над 49 ділянками,
    розтягнуті до розміру зображення 224×224 та накладені на фото.
    """
    img = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    n = len(words)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.8 * rows))
    side = int(np.sqrt(config.NUM_REGIONS))  # 7
    for i, ax in enumerate(np.array(axes).flat):
        ax.axis("off")
        if i >= n:
            continue
        att = Image.fromarray(alphas[i].reshape(side, side).astype(np.float32))
        att = np.array(att.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR))
        ax.imshow(img)
        ax.imshow(att, cmap="jet", alpha=0.55)
        ax.set_title(words[i], fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def plot_own_images(images, names, greedy_caps, beam_caps):
    """Власні зображення з підписами greedy та beam search."""
    fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5.6))
    for ax, img, name, g, b in zip(np.atleast_1d(axes), images, names, greedy_caps, beam_caps):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{name}\ngreedy: {_wrap(g)}\nbeam: {_wrap(b)}", fontsize=10)
    fig.tight_layout()
    return _save(fig, "own_images.png")
