"""
Генерація підписів навченою моделлю: greedy-декодування та beam search.

Greedy  — на кожному кроці беремо найімовірніше слово. Швидко, але одна
          невдала локальна помилка псує все речення.
Beam    — «пошук променем»: тримаємо k найкращих часткових речень, на кожному
          кроці розширюємо кожне на всі слова словника й залишаємо k найкращих
          за сумарною лог-ймовірністю. Наприкінці обираємо найкраще повне речення.
"""
import torch

from . import config
from .data import Vocabulary
from .models import CaptionDecoder

# Максимальна довжина згенерованого речення (у словах, разом із </s>).
MAX_DECODE_LEN = config.MAX_CAPTION_LEN + 1


@torch.no_grad()
def greedy_decode(model: CaptionDecoder, features: torch.Tensor, vocab: Vocabulary,
                  max_len: int = MAX_DECODE_LEN):
    """
    Пакетне greedy-декодування.
      features — (B, 49, 2048) на тому самому пристрої, що й модель.
    Повертає список речень (списки слів) та список матриць уваги (T_i, 49) або None.
    """
    model.eval()
    batch = features.size(0)
    state = model.init_state(features)
    projected = model.prepare(features)
    words = torch.full((batch,), vocab.start_id, dtype=torch.long, device=features.device)

    generated, alphas = [], []
    finished = torch.zeros(batch, dtype=torch.bool, device=features.device)
    for _ in range(max_len):
        logits, state, alpha = model.step(words, state, features, projected)
        words = logits.argmax(dim=1)          # найімовірніше наступне слово
        generated.append(words)
        if alpha is not None:
            alphas.append(alpha)
        finished |= words == vocab.end_id
        if finished.all():                    # усі речення вже закінчились
            break

    generated = torch.stack(generated, dim=1).cpu()
    sentences = [vocab.decode(row) for row in generated]
    if not alphas:
        return sentences, None
    alphas = torch.stack(alphas, dim=1).cpu()
    # Для кожного речення залишаємо ваги уваги лише для його слів.
    return sentences, [alphas[i, :len(s)].numpy() for i, s in enumerate(sentences)]


@torch.no_grad()
def beam_search(model: CaptionDecoder, features: torch.Tensor, vocab: Vocabulary,
                beam_size: int = 3, max_len: int = MAX_DECODE_LEN):
    """
    Beam search для ОДНОГО зображення.
      features — (1, 49, 2048).
    Повертає (список слів, матриця уваги (T, 49) або None).
    """
    model.eval()
    k = beam_size
    device = features.device
    feats = features.expand(k, -1, -1)                       # копія ознак для кожного променя
    state = model.init_state(feats)
    projected = model.prepare(feats)

    seqs = torch.full((k, 1), vocab.start_id, dtype=torch.long, device=device)  # часткові речення
    scores = torch.zeros(k, device=device)                   # сумарні лог-ймовірності
    seq_alphas = torch.zeros(k, 0, config.NUM_REGIONS, device=device)
    complete = []                                            # завершені речення (оцінка, слова, увага)

    for t in range(max_len):
        logits, (h, c), alpha = model.step(seqs[:, -1], state, feats, projected)
        log_probs = torch.log_softmax(logits, dim=1)         # (k, V)
        total = scores.unsqueeze(1) + log_probs              # оцінка кожного продовження
        if t == 0:
            total = total[:1]                                # на першому кроці всі промені однакові
        top_scores, top_idx = total.reshape(-1).topk(k)      # k найкращих (промінь, слово)
        beam_idx = top_idx // model.vocab_size               # з якого променя продовження
        word_idx = top_idx % model.vocab_size                # яким словом продовжили

        seqs = torch.cat([seqs[beam_idx], word_idx.unsqueeze(1)], dim=1)
        if alpha is not None:
            seq_alphas = torch.cat([seq_alphas[beam_idx], alpha[beam_idx].unsqueeze(1)], dim=1)
        h, c = h[beam_idx], c[beam_idx]

        # Промені, що згенерували </s>, переносимо до завершених.
        ended = word_idx == vocab.end_id
        for i in ended.nonzero().flatten().tolist():
            complete.append((top_scores[i].item(), seqs[i].tolist(), seq_alphas[i].cpu()))
        keep = ~ended
        k = int(keep.sum())
        if k == 0:
            break
        seqs, scores, seq_alphas = seqs[keep], top_scores[keep], seq_alphas[keep]
        state = (h[keep], c[keep])
        feats = feats[:k]
        projected = projected[:k] if projected is not None else None

    if not complete:  # жодне речення не закінчилось за max_len кроків — беремо найкраще незавершене
        best = scores.argmax().item()
        complete.append((scores[best].item(), seqs[best].tolist(), seq_alphas[best].cpu()))

    _, ids, alphas = max(complete, key=lambda x: x[0])
    words = vocab.decode(ids)
    return words, (alphas[:len(words)].numpy() if model.use_attention else None)


def caption_features(model, features, vocab, beam_size: int | None = None):
    """Зручна обгортка: підпис одного зображення greedy (beam_size=None) або beam search."""
    if beam_size is None:
        sentences, alphas = greedy_decode(model, features, vocab)
        return sentences[0], (alphas[0] if alphas is not None else None)
    return beam_search(model, features, vocab, beam_size)
