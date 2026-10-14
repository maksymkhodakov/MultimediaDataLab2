"""
Метрика BLEU (Bilingual Evaluation Understudy, Papineni et al., 2002).

BLEU-n — середнє геометричне точностей n-грам порядку 1..n (частка n-грам
згенерованого речення, що трапляються хоча б в одному з еталонних підписів),
помножене на штраф за надто коротке речення (brevity penalty).
Рахуємо на рівні корпусу (corpus BLEU), як у статтях Show and Tell / Show, Attend and Tell.
"""
from nltk.translate.bleu_score import corpus_bleu


def bleu_scores(references: list[list[list[str]]], hypotheses: list[list[str]]) -> dict[str, float]:
    """
    references — для кожного зображення список еталонних підписів (кожен — список слів);
    hypotheses — для кожного зображення згенерований підпис (список слів).
    Повертає словник {'BLEU-1': ..., ..., 'BLEU-4': ...}.
    """
    result = {}
    for n in range(1, 5):
        weights = tuple([1.0 / n] * n)  # рівні ваги для n-грам порядку 1..n
        result[f"BLEU-{n}"] = corpus_bleu(references, hypotheses, weights=weights)
    return result


def human_bleu(references: list[list[list[str]]]) -> dict[str, float]:
    """
    «Людський» BLEU: перший людський підпис порівнюємо з чотирма іншими.
    Показує, наскільки різняться між собою описи різних людей —
    верхня орієнтовна межа для метрики (тільки з 4 еталонами замість 5).
    """
    return bleu_scores([refs[1:] for refs in references], [refs[0] for refs in references])
