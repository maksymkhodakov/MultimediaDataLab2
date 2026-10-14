# Лабораторна робота №2
## "Методи обробки мультимедійної інформації"
## Ходаков Максим Олегович ШІ-2

**Тема:** формування підписів до зображень (image captioning) на датасеті Flickr8K.

Encoder — згорткова мережа ResNet-50 (ImageNet), decoder — LSTM.
Порівнюються архітектури Show and Tell та Show, Attend and Tell (з адитивною увагою),
декодування greedy та beam search, якість оцінюється метрикою BLEU-1…4.

### Структура

| Файл | Призначення |
|---|---|
| `main.py` | повний конвеєр: дані → ознаки → навчання → оцінювання → приклади → власні фото |
| `src/config.py` | шляхи, гіперпараметри, опис експериментів A/B/C |
| `src/data.py` | читання Flickr8K (parquet), токенізація, словник |
| `src/encoder.py` | ResNet-50 encoder, обчислення та збереження ознак 49×2048 |
| `src/models.py` | LSTM-декодер з увагою Bahdanau та без неї |
| `src/decoding.py` | greedy-декодування та beam search |
| `src/metrics.py` | BLEU-1…4 |
| `src/train.py` | навчання, валідація, збереження найкращої епохи |
| `src/visualize.py` | графіки, приклади, карти уваги |
| `own_images/` | власні зображення для пункту 4 |
| `outputs/` | результати: `summary.json`, історії навчання, `figures/`, журнали `train_log.txt` (навчання) та `eval_log.txt` (повторне оцінювання) |

### Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py                      # усі етапи (~1.5 год на Apple M1)
python main.py --caption фото.jpg   # підпис до довільного зображення
```

Датасет (~1 ГБ, `jxie/flickr8k` з Hugging Face Hub) завантажується автоматично у `data/`.
