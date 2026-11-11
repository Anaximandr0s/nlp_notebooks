# NLP Notebooks: Evaluating Language Models

Three hands-on notebooks to **see NLP models work live** and to **learn how to judge them**:
which metric to use, how it is computed, and **when it lies**.

Made for the course *Évaluation des métriques de l'IA* — Master Data & AI, INPT Rabat (2026/2027).

| # | Notebook | Task | Models compared | Main metrics |
|---|---|---|---|---|
| 01 | [`01_text_classification.ipynb`](01_text_classification.ipynb) | Sentiment & spam detection | TF-IDF + Logistic Regression vs **DistilBERT** | Accuracy, Precision, Recall, F1, Macro-F1, confusion matrix |
| 02 | [`02_sequence_labeling.ipynb`](02_sequence_labeling.ipynb) | Named Entity Recognition (French) | **spaCy** `fr_core_news_sm` vs **CamemBERT-NER** | Token accuracy, entity P/R/F1 (seqeval), strict vs partial, F1 per type |
| 03 | [`03_machine_translation.ipynb`](03_machine_translation.ipynb) | Translation French ➜ English | **MarianMT** small vs big | BLEU (step by step), chrF, TER, COMET (optional) |

Every notebook follows the same path:

1. **Settings**: one "control panel" cell, the only one you need to change
2. **How it works**: a simple analogy and a small diagram
3. **Live demo**: type your own sentence and see the result in color
4. **Evaluation**: metrics computed by hand, then with the official library
5. **Traps**: real scenarios where a high score hides a problem
6. **Two models compared**: speed, size and score
7. **Recap**: three numbers to remember, things to try, common problems

---

## What you will see

### 01 · Text classification
- Confidence bars for any English sentence, tested on **negation** (*"not bad at all"*), **sarcasm** and **SMS style**
- A real spam filter trained in one second on **5,574 real SMS** (SMS Spam Collection)
- **The lazy model**: about 87% accuracy while catching **zero** spam
- **The threshold game**: precision vs recall, "paranoid" vs "relaxed" filter
- Reading the mistakes: false alarms and missed spams
- **Old school vs Transformer** on the SST-2 benchmark (speed, size, score)
- A **stress test** per skill: simple, negation, sarcasm, SMS

### 02 · Sequence labeling (French NER)
- Entities highlighted like markers (PER, LOC, ORG, MISC), plus the **BIO** and **POS** labels word by word
- A hand-annotated French test set (16 sentences, Moroccan names, cities, clubs and companies)
- The slide example computed live: 88.9% token accuracy but 66.7% entity F1
- **Trap 1**: the "all O" model with a good token accuracy and 0 entities found
- **Trap 2**: one wrong border = one false positive + one false negative (strict vs partial)
- **Ethics test**: are French, Moroccan, Amazigh and English names detected equally well?

### 03 · Machine translation
- Small vs big model side by side, with idioms (*"Il pleut des cordes"*, *"tomber dans les pommes"*)
- **BLEU step by step**: n-grams, clipping, brevity penalty, matching words in green
- An 18-sentence test set in 5 kinds: daily life, news, idioms, medical, tech
- **The dangerous trap**: *"The patient does have a fever"* gets a **higher** BLEU than the correct translation
- One reference vs three references, and sentence score vs corpus score
- **Ethics test (WinoMT idea)**: does *"The doctor said that she was tired"* become *"le médecin"* or *"la médecin"*?

---

## Setup (one time only)

Requires **Python 3.10+**. On the first run the notebooks download the models (see sizes below), then everything works offline.

```bash
git clone https://github.com/Anaximandr0s/nlp_notebooks.git
cd nlp_notebooks

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

`requirements.txt`:

```text
ipykernel
transformers
torch
sentencepiece
sacremoses
protobuf
datasets
scikit-learn
spacy
seqeval
sacrebleu
matplotlib
pandas
```

Then open the folder in **VS Code**, open a notebook, click **Select Kernel** ➜ **Python Environments** ➜ **.venv**, and run the cells from top to bottom.

> Each notebook also has an install cell (`%pip install ...`) at the top, if you prefer not to use the terminal.

---

## Models and data

| Notebook | Resource | Size (approx.) | Source |
|---|---|---|---|
| 01 | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | 270 MB | Hugging Face |
| 01 | SMS Spam Collection (`ucirvine/sms_spam`) | 0.5 MB | Hugging Face Datasets |
| 01 | SST-2 (`stanfordnlp/sst2`) | 5 MB | Hugging Face Datasets |
| 02 | `Jean-Baptiste/camembert-ner` | 440 MB | Hugging Face |
| 02 | spaCy `fr_core_news_sm` | 15 MB | spaCy |
| 03 | `Helsinki-NLP/opus-mt-fr-en` (small) | 300 MB | Hugging Face |
| 03 | `Helsinki-NLP/opus-mt-tc-big-fr-en` (big) | 900 MB | Hugging Face |
| 03 | `Helsinki-NLP/opus-mt-en-fr` (ethics test) | 300 MB | Hugging Face |
| 03 | `Unbabel/wmt22-comet-da` (optional, `USE_COMET = True`) | 2.3 GB | Hugging Face |

If a dataset cannot be downloaded, notebooks 01 and 02 fall back to a small built-in sample, so the demo never crashes.

---

## Settings you can change

| Notebook | Setting | Effect |
|---|---|---|
| all | `LIVE_DEMO` | `True` = type your own sentences, `False` = skip the interactive cell |
| 01 | `THRESHOLD` | Probability above which an SMS is called spam |
| 01 | `BENCHMARK_SIZE` | Number of SST-2 test sentences (`None` = 872, `200` = faster) |
| 02 | `MIN_SCORE` | Minimum confidence for an entity |
| 03 | `RUN_BIG_MODEL` | `False` on a slow computer |
| 03 | `NUM_BEAMS` | Beam search width (`1` = greedy, faster) |
| 03 | `USE_COMET` | Adds the COMET neural metric (heavy) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `KeyError: The task translation has been removed` | You have **transformers v5**. Notebook 03 already loads MarianMT directly (`Translator` class), so pull the latest version of the notebook |
| `Can't find model 'fr_core_news_sm'` | `python -m spacy download fr_core_news_sm`, then restart the kernel |
| `sentencepiece` / `sacremoses` error | `pip install sentencepiece sacremoses protobuf`, then restart the kernel |
| The `input()` box does not appear | In VS Code it appears at the **top** of the window. Or set `LIVE_DEMO = False` |
| A notebook is slow | Reduce `BENCHMARK_SIZE`, set `RUN_BIG_MODEL = False`, close other apps |
| Different numbers than a classmate | Same models, same `SEED` and `NUM_BEAMS`? Metrics from scikit-learn, seqeval and sacreBLEU are deterministic |

---

## The main lesson

> **A high score can hide a problem.**
> Always look at several metrics, at concrete examples, at the errors, and at the results **per class and per group**.

| Task | The trap shown in the notebook |
|---|---|
| Classification | 87% accuracy without finding a single spam |
| Sequence labeling | Most words are right, but no entity is found |
| Translation | BLEU prefers the translation with the opposite meaning |

---

## References

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.*
- Vaswani et al. (2017). *Attention Is All You Need.*
- Papineni et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.*
- Popović (2015). *chrF: character n-gram F-score for automatic MT evaluation.*
- Snover et al. (2006). *A Study of Translation Edit Rate with Targeted Human Annotation.*
- Rei et al. (2020). *COMET: A Neural Framework for MT Evaluation.*
- Ribeiro et al. (2020). *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList.*
- Stanovsky et al. (2019). *Evaluating Gender Bias in Machine Translation.*
- Mishra et al. (2020). *Assessing Demographic Bias in Named Entity Recognition.*

---

**Author:** Mohamed Doukkani · Master Data & AI, INPT Rabat · 2026/2027