"""
Навчання декодерів та оцінювання на валідації.

Функція втрат — перехресна ентропія (cross-entropy) між передбаченим
розподілом наступного слова і правильним словом (позиції <pad> ігноруються).
Для моделей з увагою додаємо регуляризатор Xu et al.:
    λ · Σ_i (1 − Σ_t α_ti)²,
який спонукає модель за все речення «подивитись» на кожну ділянку зображення.

Після кожної епохи рахуємо BLEU-4 на валідаційних зображеннях (greedy)
і зберігаємо ваги найкращої епохи — це захищає від перенавчання.
"""
import json
import time

import torch
from torch import nn

from . import config
from .decoding import greedy_decode
from .metrics import bleu_scores
from .models import build_model, count_parameters


def iterate_batches(n: int, batch_size: int, generator: torch.Generator | None = None):
    """Номери прикладів, розбиті на пакети; при generator — у випадковому порядку."""
    order = torch.randperm(n, generator=generator) if generator is not None else torch.arange(n)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def train_one_epoch(model, optimizer, criterion, features, pairs, device, generator) -> float:
    """
    Одна епоха навчання за всіма 30 000 парами «зображення — підпис».
    Повертає середнє значення перехресної ентропії.
    """
    model.train()
    image_index, tokens, lengths = pairs
    total_loss, total_tokens = 0.0, 0
    for idx in iterate_batches(len(tokens), config.BATCH_SIZE, generator):
        max_len = int(lengths[idx].max())
        captions = tokens[idx, :max_len].to(device)                # обрізаємо зайві <pad>
        feats = features[image_index[idx]].to(device).float()      # float16 → float32

        logits, alphas = model(feats, captions)
        targets = captions[:, 1:]                                  # передбачаємо наступне слово
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        full_loss = loss
        if alphas is not None:
            # Регуляризація уваги: сума ваг кожної ділянки по всіх кроках ≈ 1.
            full_loss = loss + config.ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

        optimizer.zero_grad()
        full_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        n_tokens = int((targets != 0).sum())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


@torch.no_grad()
def generate_all(model, features, vocab, device, batch_size: int = 250):
    """Greedy-підписи для всіх зображень частини датасету."""
    hypotheses = []
    for idx in iterate_batches(len(features), batch_size):
        sentences, _ = greedy_decode(model, features[idx].to(device).float(), vocab)
        hypotheses.extend(sentences)
    return hypotheses


def train_experiment(key: str, cfg: dict, vocab, train_pairs, train_features,
                     val_features, val_refs, device) -> dict:
    """
    Повний цикл навчання одного експерименту (A, B або C).
    Зберігає найкращі ваги в outputs/checkpoints/<key>.pt та історію в outputs/history_<key>.json.
    """
    config.set_seed()
    model = build_model(cfg, len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    generator = torch.Generator().manual_seed(config.SEED)  # однаковий порядок пакетів для всіх моделей

    print(f"{cfg['name']}: параметрів {count_parameters(model):,}")
    history = {"name": cfg["name"], "params": count_parameters(model),
               "train_loss": [], "val_bleu4": [], "epoch_time": []}
    best_bleu, best_epoch = -1.0, 0
    start = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, criterion, train_features, train_pairs, device, generator)
        hyps = generate_all(model, val_features, vocab, device)
        bleu4 = bleu_scores(val_refs, hyps)["BLEU-4"]
        history["train_loss"].append(loss)
        history["val_bleu4"].append(bleu4)
        history["epoch_time"].append(time.time() - t0)
        mark = ""
        if bleu4 > best_bleu:                              # нова найкраща епоха — зберігаємо ваги
            best_bleu, best_epoch = bleu4, epoch
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / f"{key}.pt")
            mark = "  *"
        print(f"  епоха {epoch:2d}: loss {loss:.3f}  val BLEU-4 {bleu4:.3f}  ({time.time() - t0:.0f} с){mark}")

    history.update(best_epoch=best_epoch, best_val_bleu4=best_bleu, total_time=time.time() - start)
    print(f"  найкраща епоха {best_epoch}: val BLEU-4 = {best_bleu:.3f}, час {history['total_time']:.0f} с")
    (config.OUTPUT_DIR / f"history_{key}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def load_trained(key: str, cfg: dict, vocab_size: int, device):
    """Завантажує найкращі збережені ваги експерименту."""
    model = build_model(cfg, vocab_size).to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / f"{key}.pt", map_location=device))
    model.eval()
    return model
