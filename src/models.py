"""
Decoder: рекурентна мережа LSTM, що генерує підпис слово за словом.

Реалізовано дві класичні архітектури в одному класі CaptionDecoder:

1. Show and Tell (Vinyals et al., 2015) — attention_dim = None.
   LSTM бачить зображення лише через початковий стан (h0, c0), обчислений
   з усередненого по 49 ділянках вектора ознак.

2. Show, Attend and Tell (Xu et al., 2015) — attention_dim задано.
   На кожному кроці механізм уваги обчислює ваги α над 49 ділянками
   зображення, і на вхід LSTM додатково подається вектор контексту —
   зважена сума ознак ділянок.
"""
import torch
from torch import nn

from . import config


class BahdanauAttention(nn.Module):
    """
    Адитивна увага (Bahdanau et al., 2015):
        e_i = w^T · tanh(W_a · a_i + W_h · h)   — «оцінка» i-ї ділянки,
        α   = softmax(e)                         — ваги уваги (сума = 1),
        z   = Σ α_i · a_i                        — вектор контексту,
    де a_i — ознаки i-ї ділянки (2048), h — поточний прихований стан LSTM.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(feature_dim, attention_dim)  # W_a
        self.decoder_proj = nn.Linear(hidden_dim, attention_dim)   # W_h
        self.score = nn.Linear(attention_dim, 1)                   # w

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """W_a · a_i не залежить від кроку, тому обчислюємо один раз на все речення."""
        return self.encoder_proj(features)                         # (B, 49, att)

    def forward(self, features, projected, h):
        e = self.score(torch.tanh(projected + self.decoder_proj(h).unsqueeze(1))).squeeze(2)  # (B, 49)
        alpha = torch.softmax(e, dim=1)                                                    # (B, 49)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)                                # (B, 2048)
        return context, alpha


class CaptionDecoder(nn.Module):
    """LSTM-декодер підписів з необов'язковим механізмом уваги."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 attention_dim: int | None = None, feature_dim: int = config.FEATURE_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()
        self.use_attention = attention_dim is not None
        self.vocab_size = vocab_size

        # Embedding: індекс слова → щільний вектор розміру embed_dim.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Початковий стан LSTM з усередненого вектора зображення.
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        lstm_input = embed_dim
        if self.use_attention:
            self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)
            # «Ворота» β = σ(f_β(h)) — модель сама вирішує, наскільки на цьому кроці
            # потрібна інформація із зображення (для службових слів — менше).
            self.f_beta = nn.Linear(hidden_dim, feature_dim)
            lstm_input += feature_dim  # на вхід LSTM: [embedding слова; вектор контексту]

        # LSTMCell — одна комірка LSTM; цикл за словами пишемо вручну, бо на
        # кожному кроці увага залежить від попереднього стану h.
        self.lstm = nn.LSTMCell(lstm_input, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Вихідний шар: прихований стан → оцінки (логіти) для кожного слова словника.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    # ------------------------------------------------------------------ кроки
    def init_state(self, features: torch.Tensor):
        """(h0, c0) = tanh(W · mean(a_1..a_49)) — «перший погляд» на зображення."""
        mean = features.mean(dim=1)
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def prepare(self, features: torch.Tensor):
        """Попередні обчислення, спільні для всіх кроків (проєкція ознак для уваги)."""
        return self.attention.project(features) if self.use_attention else None

    def step(self, words: torch.Tensor, state, features: torch.Tensor, projected):
        """
        Один крок декодера.
          words    — індекси попередніх слів (B,);
          state    — (h, c) з попереднього кроку;
        Повертає логіти наступного слова (B, V), новий стан і ваги уваги (B, 49) або None.
        """
        h, c = state
        x = self.embedding(words)
        alpha = None
        if self.use_attention:
            context, alpha = self.attention(features, projected, h)
            gate = torch.sigmoid(self.f_beta(h))
            x = torch.cat([x, gate * context], dim=1)
        h, c = self.lstm(x, (h, c))
        logits = self.fc(self.dropout(h))
        return logits, (h, c), alpha

    # ----------------------------------------------------------------- навчання
    def forward(self, features: torch.Tensor, captions: torch.Tensor):
        """
        Навчання методом teacher forcing: на вхід кожного кроку подаємо
        ПРАВИЛЬНЕ попереднє слово з еталонного підпису (а не згенероване моделлю).
          features — ознаки зображень (B, 49, 2048);
          captions — індекси слів <s> w1 ... wn </s> <pad>... (B, T).
        Повертає логіти для передбачення captions[:, 1:] — (B, T-1, V)
        та ваги уваги (B, T-1, 49) (або None для моделі без уваги).
        """
        state = self.init_state(features)
        projected = self.prepare(features)
        logits, alphas = [], []
        for t in range(captions.size(1) - 1):
            out, state, alpha = self.step(captions[:, t], state, features, projected)
            logits.append(out)
            if alpha is not None:
                alphas.append(alpha)
        logits = torch.stack(logits, dim=1)
        alphas = torch.stack(alphas, dim=1) if alphas else None
        return logits, alphas


def build_model(cfg: dict, vocab_size: int) -> CaptionDecoder:
    """Створює декодер за описом експерименту з config.EXPERIMENTS."""
    return CaptionDecoder(vocab_size, cfg["embed_dim"], cfg["hidden_dim"], cfg["attention_dim"])


def count_parameters(model: nn.Module) -> int:
    """Кількість параметрів моделі, що навчаються."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
