"""
Лабораторна робота №2. Формування підписів до зображень (image captioning) на Flickr8K.

Повний конвеєр:
  1. завантаження датасету Flickr8K та побудова словника;
  2. обчислення ознак зображень згортковою мережею ResNet-50 (encoder);
  3. навчання трьох LSTM-декодерів (Show and Tell та Show, Attend and Tell з різними розмірами);
  4. оцінювання на тестовій вибірці метриками BLEU-1…4 (greedy та beam search);
  5. приклади підписів, візуалізація уваги;
  6. підписи до власних зображень із теки own_images/.

Запуск:
  python main.py                    # усі етапи (вже обчислене береться з диску)
  python main.py --retrain          # навчити моделі заново
  python main.py --caption фото.jpg # підписати довільне зображення найкращою моделлю
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from PIL import Image

from src import config, data, encoder, visualize
from src.decoding import beam_search, greedy_decode
from src.metrics import bleu_scores, human_bleu
from src.train import generate_all, load_trained, train_experiment

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# 1. Датасет і словник
# ---------------------------------------------------------------------------
def prepare_data(summary: dict):
    data.download_dataset()
    captions = {s: data.load_captions(s) for s in config.SPLITS}
    for s in config.SPLITS:
        print(f"{s:<11} зображень {len(captions[s]):5d}, підписів {5 * len(captions[s])}")

    vocab, counter = data.Vocabulary.build(captions["train"])
    vocab.save(config.OUTPUT_DIR / "vocab.json")
    lengths = [len(data.tokenize(c)) for caps in captions["train"] for c in caps]
    words = vocab.itos[4:]
    print(f"Різних слів: {len(counter)}, у словнику (частота >= {config.MIN_WORD_FREQ}): {len(words)}"
          f" (+4 службові токени = {len(vocab)})")
    print(f"Довжина підпису: середня {np.mean(lengths):.1f}, 95% підписів <= "
          f"{int(np.percentile(lengths, 95))} слів, максимум {max(lengths)}")
    print("Найчастіші слова:", ", ".join(words[:25]))

    covered = sum(counter[w] for w in words) / sum(counter.values())
    summary["dataset"] = {
        "sizes": {s: len(captions[s]) for s in config.SPLITS},
        "unique_words": len(counter), "vocab_words": len(words), "vocab_size": len(vocab),
        "mean_len": float(np.mean(lengths)), "p95_len": int(np.percentile(lengths, 95)),
        "max_len": max(lengths), "token_coverage": covered, "top_words": words[:25],
        "truncated_share": float(np.mean(np.array(lengths) > config.MAX_CAPTION_LEN)),
    }
    visualize.plot_dataset_stats(counter, words, lengths)
    visualize.plot_dataset_examples(data.load_images("train", [0, 1, 2]), captions["train"][:3])
    return captions, vocab


# ---------------------------------------------------------------------------
# 3. Навчання / завантаження моделей
# ---------------------------------------------------------------------------
def train_or_load(captions, vocab, device, retrain: bool):
    feats = {s: encoder.load_features(s) for s in config.SPLITS}
    for s in config.SPLITS:
        print(f"{s:<11} ознаки {tuple(feats[s].shape)}")
    train_pairs = data.encode_split(captions["train"], vocab)
    val_refs = data.references(captions["validation"])

    histories, models = {}, {}
    for key, cfg in config.EXPERIMENTS.items():
        hist_path = config.OUTPUT_DIR / f"history_{key}.json"
        ckpt = config.CHECKPOINT_DIR / f"{key}.pt"
        if retrain or not (hist_path.exists() and ckpt.exists()):
            histories[key] = train_experiment(key, cfg, vocab, train_pairs, feats["train"],
                                              feats["validation"], val_refs, device)
        else:
            histories[key] = json.loads(hist_path.read_text(encoding="utf-8"))
            print(f"{cfg['name']}: завантажено (найкраща епоха {histories[key]['best_epoch']}, "
                  f"val BLEU-4 = {histories[key]['best_val_bleu4']:.3f})")
        models[key] = load_trained(key, cfg, len(vocab), device)
    visualize.plot_training_curves(histories)
    return feats, histories, models


# ---------------------------------------------------------------------------
# 4. Оцінювання на тесті
# ---------------------------------------------------------------------------
def beam_all(model, features, vocab, device, k):
    return [beam_search(model, features[i:i + 1].to(device).float(), vocab, k)[0]
            for i in range(len(features))]


def evaluate(models, histories, test_feats, test_refs, vocab, device, summary):
    rows, outputs = [], {}
    for key, model in models.items():
        t0 = time.time()
        hyps = generate_all(model, test_feats, vocab, device)
        rows.append({"key": key, "decoding": "greedy", **bleu_scores(test_refs, hyps),
                     "time": time.time() - t0})
        outputs[(key, "greedy")] = hyps

    best = max(models, key=lambda k: histories[k]["best_val_bleu4"])  # найкраща модель за валідацією
    for key in models:
        for k in config.BEAM_SIZES:
            if key != best and k != config.BEAM_SIZES[0]:
                continue  # k=5 рахуємо лише для найкращої моделі
            t0 = time.time()
            hyps = beam_all(models[key], test_feats, vocab, device, k)
            rows.append({"key": key, "decoding": f"beam k={k}", **bleu_scores(test_refs, hyps),
                         "time": time.time() - t0})
            outputs[(key, f"beam k={k}")] = hyps

    human = human_bleu(test_refs)
    print(f"\n{'модель':<40}{'декодування':<13}{'BLEU-1':>8}{'BLEU-2':>8}{'BLEU-3':>8}{'BLEU-4':>8}{'час, с':>8}")
    for r in rows:
        print(f"{config.EXPERIMENTS[r['key']]['name']:<40}{r['decoding']:<13}"
              + "".join(f"{r[f'BLEU-{n}']:8.3f}" for n in range(1, 5)) + f"{r['time']:8.0f}")
    print(f"{'Людина (підпис 1 проти 4 інших)':<53}" + "".join(f"{human[f'BLEU-{n}']:8.3f}" for n in range(1, 5)))

    # Різноманітність: скільки різних слів і різних речень генерує модель.
    diversity = {f"{k}|{d}": {"unique_words": len({w for h in hyps for w in h}),
                              "unique_sentences": len({" ".join(h) for h in hyps}),
                              "mean_len": float(np.mean([len(h) for h in hyps]))}
                 for (k, d), hyps in outputs.items()}
    summary["test"] = {"rows": rows, "human": human, "best": best, "diversity": diversity}
    return best, outputs


# ---------------------------------------------------------------------------
# 5. Приклади на тестових зображеннях і увага
# ---------------------------------------------------------------------------
def test_examples(model, best_key, outputs, captions, test_refs, test_feats, vocab, device, summary):
    hyps = outputs[(best_key, f"beam k={config.BEAM_SIZES[0]}")]
    smooth = SmoothingFunction().method1
    # BLEU-4 кожного окремого речення — щоб знайти вдалі та невдалі приклади.
    sent_bleu = [sentence_bleu(refs, h, smoothing_function=smooth) for refs, h in zip(test_refs, hyps)]
    order = np.argsort(sent_bleu)

    rng = random.Random(config.SEED)
    groups = {
        "random": rng.sample(range(len(hyps)), 6),
        "good": rng.sample(list(order[-60:]), 6),
        "bad": rng.sample(list(order[:60]), 6),
    }
    summary["examples"] = {}
    for name, idx in groups.items():
        idx = [int(i) for i in idx]
        imgs = data.load_images("test", idx)
        model_caps = [" ".join(hyps[i]) for i in idx]
        human_caps = [captions["test"][i][0].strip().rstrip(" .") for i in idx]
        extra = [f"BLEU-4 речення = {sent_bleu[i]:.2f}" for i in idx]
        visualize.plot_captioned_grid(imgs, model_caps, human_caps, f"test_{name}.png", extra)
        greedy = outputs[(best_key, "greedy")]  # для порівняння greedy та beam search
        summary["examples"][name] = [{"index": i, "model": m, "greedy": " ".join(greedy[i]),
                                      "human": captions["test"][i], "bleu4": sent_bleu[i]}
                                     for i, m in zip(idx, model_caps)]

    # Візуалізація уваги для двох вдалих прикладів.
    summary["attention"] = []
    for n, i in enumerate(summary["examples"]["good"][:2] + summary["examples"]["random"][:1]):
        i = i["index"]
        words, alphas = beam_search(model, test_feats[i:i + 1].to(device).float(), vocab, config.BEAM_SIZES[0])
        visualize.plot_attention(data.load_images("test", [i])[0], words, alphas, f"attention_{n + 1}.png")
        summary["attention"].append({"index": i, "caption": " ".join(words)})


# ---------------------------------------------------------------------------
# 6. Власні зображення
# ---------------------------------------------------------------------------
def caption_file(path, model, vocab, device, enc=None, beam_size=None):
    """Підпис до довільного JPG/PNG-файлу: greedy (beam_size=None) або beam search."""
    enc = enc or encoder.ResNetEncoder().to(device)
    image = Image.open(path).convert("RGB")
    feats = encoder.encode_pil(image, enc, device)
    if beam_size is None:
        sentences, alphas = greedy_decode(model, feats, vocab)
        return " ".join(sentences[0]), (alphas[0] if alphas else None), image
    words, alphas = beam_search(model, feats, vocab, beam_size)
    return " ".join(words), alphas, image


def own_images(models, best_key, vocab, device, summary):
    files = sorted(p for p in config.OWN_IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        print("\nУ теці own_images/ немає зображень — пункт 4 пропущено.")
        return
    enc = encoder.ResNetEncoder().to(device)
    summary["own"] = []
    images, greedy_caps, beam_caps = [], [], []
    print("\nВласні зображення:")
    for n, path in enumerate(files):
        g, _, image = caption_file(path, models[best_key], vocab, device, enc)
        b, alphas, _ = caption_file(path, models[best_key], vocab, device, enc, config.BEAM_SIZES[0])
        # Для порівняння — підписи всіх трьох моделей (beam k=3).
        per_model = {k: caption_file(path, m, vocab, device, enc, config.BEAM_SIZES[0])[0]
                     for k, m in models.items()}
        print(f"  {path.name:<18} greedy: {g}\n  {'':<18} beam:   {b}")
        summary["own"].append({"file": path.name, "greedy": g, "beam": b, "models": per_model})
        images.append(image)
        greedy_caps.append(g)
        beam_caps.append(b)
        if alphas is not None:
            visualize.plot_attention(image, b.split(), alphas, f"attention_own_{n + 1}.png")
    visualize.plot_own_images(images, [p.name for p in files], greedy_caps, beam_caps)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Image captioning на Flickr8K")
    parser.add_argument("--retrain", action="store_true", help="навчити моделі заново")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="кількість епох")
    parser.add_argument("--caption", type=str, help="шлях до зображення, яке треба підписати")
    args = parser.parse_args()
    config.EPOCHS = args.epochs

    config.ensure_dirs()
    config.set_seed()
    device = config.get_device()
    print("Пристрій:", device)

    if args.caption:  # швидкий режим: лише підписати одне зображення
        vocab = data.Vocabulary.load(config.OUTPUT_DIR / "vocab.json")
        summary = json.loads((config.OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))
        key = summary["test"]["best"]
        model = load_trained(key, config.EXPERIMENTS[key], len(vocab), device)
        enc = encoder.ResNetEncoder().to(device)
        print("greedy:", caption_file(args.caption, model, vocab, device, enc)[0])
        print("beam:  ", caption_file(args.caption, model, vocab, device, enc, config.BEAM_SIZES[0])[0])
        return

    summary = {"device": str(device)}
    captions, vocab = prepare_data(summary)
    encoder.extract_all(device, {s: len(captions[s]) for s in config.SPLITS})
    feats, histories, models = train_or_load(captions, vocab, device, args.retrain)
    summary["histories"] = histories

    test_refs = data.references(captions["test"])
    best, outputs = evaluate(models, histories, feats["test"], test_refs, vocab, device, summary)
    # Для прикладів і візуалізації уваги беремо найкращу модель з увагою.
    att_key = max((k for k in models if models[k].use_attention), key=lambda k: histories[k]["best_val_bleu4"])
    summary["attention_key"] = att_key
    test_examples(models[att_key], att_key, outputs, captions, test_refs, feats["test"], vocab, device, summary)
    own_images(models, best, vocab, device, summary)

    (config.OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    print("\nГотово. Результати — у теці outputs/")


if __name__ == "__main__":
    main()
