# Deep-Learning-Assignment-3

## EDA and Metric Analysis

Notebook: [`notebooks/01_eda_metric.ipynb`](notebooks/01_eda_metric.ipynb).

| File | What it is | Our use |
|---|---|---|
| `uk_geo_dataset.csv` | ~1M Ukr-Synth sentences with silver location spans | train (`is_valid==0`), early stopping (`is_valid==1`) |
| `labeling_sample.csv` | 26 Telegram posts with gold locations | CV |
| `test.csv` | 477 Telegram posts, no labels | submission |
| `ru_geo_dataset.csv` | ~8M Russian news sentences with silver spans (nerus) | candidate extra data |

### Data insights
1. **Strong domain shift between train and test.**

   | | train | CV (`labeling_sample`) | test |
   |---|---|---|---|
   | median words | 16 | 36.5 | 46 |
   | has emoji | 0% | 69% | 63% |
   | has line break | 0% | 81% | 85% |
   | Russian | 0.1% | 4% | 7% |
   | texts with ≥1 location | 23% | 62% | – |

   Train is short, clean news sentences. Test is long Telegram posts with emoji, line breaks, and some Russian.
2. **The test-like gold follows its own convention.**
   - 14 lines repeat in ≥5 test posts: channel signatures such as `Життя🫶Оболонь` and `Хуйовий Київ | Надіслати новину`.
     They appear in 35% of test posts. Gold never labels a place inside them.
   - Countries are probably not labeled (the only clean case, `мир в Україні`, is not gold). The evidence is weak, so
     this is checked on the LB. In train, the most frequent locations are `України`, `США`, `Росії`.
   - Gold is selective, not exhaustive: stations listed in a train schedule are not labeled.
3. **Different location types.** CV gold has streets, districts, malls, addresses and inflected towns of the Kyiv region.
   28% of the words inside gold locations are missing from a 50k word vocabulary built on train (test words overall:
   15%).
4. **Silver labels are mostly clean.** 4% of train spans are off token boundaries, almost all hyphenated names. Our
   tokenizer keeps inner hyphens and apostrophes in the word (`Івано-Франківськ`, `Слов'янськ`); converting the gold
   spans to BIO and back scores 0.97 on train and 1.00 on CV.
5. **The Russian dataset is a low-priority extra.** It cuts OOV on Russian test posts from 51% to 18%, but those are only
   ~7% of test, the CV can't measure the effect, and the data is very country-centric (`России`, `США`, `РФ`). A shared
   uk + ru vocabulary raises OOV on Ukrainian test posts from 13% to 17%.

### Metric analysis
The metric is **micro F1 over sets of normalized strings**: for each text, the set of predicted locations is compared
with the set of gold locations after `hard_processing`, and TP/FP/FN are summed over all texts.

`hard_processing` lowercases, removes digits and single-letter words, and collapses whitespace. Its punctuation regex is
a raw string, **not an f-string**, so it removes only `( ) . { } _` and the Latin letters `a c e g i n o p r s t u`;
commas, quotes and hyphens survive.

**Strengths:** simple and close to real use (a list of places per post); ignores case, duplicates and house numbers;
`м. Київ` = `Київ` and `у Львові` = `Львові`.

**Weaknesses and edge cases:**
- **All or nothing per string.** A boundary error (`Бандери` for `проспект Бандери`) costs an FP *and* an FN, which is
  worse than predicting nothing. With an uncertain boundary, silence can pay.
- **Inflection matters:** `Києві` ≠ `Київ`. We output the exact surface form and never lemmatize.
- **Punctuation quirk:** `Київ,` ≠ `Київ`. Spans must not include trailing punctuation.
- **Empty strings:** a prediction that normalizes to `""` (`В`, `28`) is still a false positive.
- **Noise:** on 26 CV texts with 32 locations, one location is worth about 3 F1 points.

### Complementary metrics
Reported for every model next to the competition F1:

| Metric | What it isolates |
|---|---|
| precision / recall | the FP vs FN trade-off, for threshold tuning |
| partial-match F1 | boundary errors: the gap to the competition F1 is what boundaries cost |
| F1 on texts with ≥1 location | detection quality without empty texts |
| FP rate on texts without locations | how often the model invents a location |
| token-level loss | what training optimizes; smooth, used for the curves |

Reference predictors on CV:

| Predictor | CV F1 | Precision | Recall | Partial F1 |
|---|---|---|---|---|
| empty | 0.000 | 0.000 | 0.000 | 0.000 |
| gazetteer of train locations (no ML) | 0.427 | 0.333 | 0.594 | 0.472 |
| gold round trip (tokenizer ceiling) | 1.000 | 1.000 | 1.000 | 1.000 |

The gazetteer scores 0.585 on `es` but 0.427 on CV: the domain shift in metric terms. It is the floor any model must
beat.

### Alternative target metric
**Boundary-tolerant (partial-match) entity F1:** a predicted location counts if it shares at least one normalized word
with a gold location of the same text, and recall is counted the same way. It fits the use case better (`Бандери` still
points to the right street, so a geocoder could use it), it doesn't reward silence over a near-miss, and it is less
noisy on a small CV. Its weakness is rewarding over-long spans; a stricter variant requires word-level Jaccard ≥ 0.5.
We still optimize the official metric and use partial F1 as a diagnostic.

## Validation

Notebook: [`notebooks/02_validation_baseline.ipynb`](notebooks/02_validation_baseline.ipynb).

| Set | What | Size | Used for |
|---|---|---|---|
| **train** | `uk_geo_dataset`, `is_valid==0` | 1M sentences | fitting |
| **es** | `uk_geo_dataset`, `is_valid==1` | 10k sentences | early stopping and checkpoint choice |
| **CV** | `labeling_sample` | 26 texts, 32 locations | the reported score, compared with the LB |
| **public LB** | 47% of `test.csv` | ~224 posts | external check, 10 submissions a day |

All notebooks load the splits with the same `load_corpora()` function (section 0 of every notebook), so results are
comparable.

**Why this design:**
- **es is a copy of train, not of test** (same 16-word median, no emoji, no line breaks). It answers "is training still
  improving?", not "how good is the model on Telegram posts?".
- **CV looks like test.** Emoji 69% vs 63%, line breaks 81% vs 85%, and the same 35% of posts carry a channel signature,
  so CV comes from the same channels in the same proportions. It is the only labelled test-like data, so it is never
  trained on. K-fold on the silver train set would measure agreement with Ukr-Synth labels on news sentences, a
  different task.
- **No leakage:** no text is shared between train, CV and test; es shares 52 texts (0.5%) with train.
- **Time and groups:** there are no timestamps, so a temporal split is impossible. The natural group is the Telegram
  channel, and CV and test share channels in the same proportions, so the split behaves like a random split of one pool.
  Ukr-Synth has no document ids; if the Russian set is added, it is split by `doc_id`.
- **Checkpoints are chosen on es, not CV.** Choosing epochs on 26 texts would fit those texts.

**CV is small.** The baseline's 95% bootstrap interval is 0.34–0.66 (±0.16). A CV difference below ~0.05 means nothing
by itself. The public LB has ~9× more posts and is the stronger check, but submissions are limited and it can be
overfit too. Rule: keep a change if it doesn't hurt es and helps CV; confirm big decisions on the LB.

**CV vs LB:** the baseline scored CV 0.488 and LB 0.459, and the best tokenization run CV 0.597 and LB 0.537. Both LB
scores are inside their CV intervals, CV is ~0.04–0.06 optimistic, and the two agree on the ranking. The fastText +
BiGRU + CRF runs (notebook 08) still rank first on both, but the gap grew to 0.13–0.17 (CV 0.765 / LB 0.598, CV 0.734 /
LB 0.60): CV and test share channels and templates (4 near-duplicate CV↔test pairs), so the 26 CV texts are optimistic. The correlation over
all submitted runs is tracked in notebook 02, section 5.

## Models

### Vanilla RNN baseline
Notebook: [`notebooks/02_validation_baseline.ipynb`](notebooks/02_validation_baseline.ipynb).

- **Tokens:** word-level, lowercased, the 50k most frequent training words seen at least twice; the rest are `<unk>`.
  Labels are BIO (`O`, `B-LOC`, `I-LOC`); decoded spans are cut from the original text.
- **Model:** random 128-d embeddings → 1-layer unidirectional `nn.RNN` (tanh), hidden size 256 → linear layer per word.
  6.5M parameters.
- **Training:** cross-entropy, AdamW (lr 2e-3, weight decay 0.01), batch 256, gradient clipping 1.0, dropout 0.3, up to
  6 epochs with early stopping on es F1 (patience 2), seed 42. PyTorch Lightning, logged to W&B.

| Run | CV F1 | 95% CI | Precision | Recall | es F1 | Public LB |
|---|---|---|---|---|---|---|
| `ivan-rnn1-word-random` | 0.488 | 0.34–0.66 | 0.40 | 0.63 | 0.802 | 0.459 |

**Insights:**
- **More training on train hurts on test-like posts.** es F1 rises every epoch (0.78 → 0.80), so early stopping never
  fires and the checkpoint is epoch 6. CV F1 is best after epoch 1 (0.59) and falls to 0.49, and the CV loss rises. The
  model fits Ukr-Synth sentences better and transfers worse to Telegram posts.
- **Optimization is healthy.** Train and es loss fall together, gradient norms are stable over 23k steps (no spikes, no
  collapse), and the total norm stays far below the clipping threshold.
- **Precision is the weak side (0.40).** False positives: places inside channel signatures (`Оболонь`), countries
  (`України`), emoji tagged as locations (`🇱`, `🕑`: they are all `<unk>` in a train vocabulary with no emoji) and
  boundary errors (`ТЦ Аркадія 📍`). 40% of CV texts without a location still get one.
- **Misses:** towns listed without context, unseen names (`Хелма`) and Russian forms (`Фастовском`), which motivates
  subword tokenization and fastText embeddings.

### Tokenization: word-level vs BPE
Notebook: [`notebooks/05_tokenization.ipynb`](notebooks/05_tokenization.ipynb).

**BPE setup.** Hugging Face `tokenizers` BPE trained on the training words (lowercased), so a piece never crosses a word
boundary. Pieces that continue a word get a `##` prefix (`васильковом` → `василь ##ков ##ом`). A word's BIO label goes on
**one** piece, its first or its last; the other pieces get −100 and are ignored by the loss. Predictions are read from
the same piece, so every model outputs word-level probabilities and can be ensembled with any other.

| Vocabulary | ids per word (test) | `<unk>` (test) | ids per gold location word | `<unk>` in gold location words |
|---|---|---|---|---|
| word 50k | 1.00 | 14.6% | 1.00 | 27.7% |
| BPE 8k | 1.63 | 1.8% | 2.26 | 0% |
| BPE 16k | 1.44 | 2.1% | 1.96 | 0% |
| BPE 32k | 1.30 | 2.3% | 1.64 | 0% |

BPE removes `<unk>` from every gold location word; what stays `<unk>` is emoji and symbols. Location words are split
more than ordinary words because they are rare.

**Experiments.** 1-layer GRU, hidden 256, random 128-d embeddings, dropout 0.3, AdamW lr 2e-3, up to 6 epochs with early
stopping on es F1, seed 42. Only the tokenizer, the label alignment and the direction change.

| Run | Tokenizer | Label on | Direction | CV F1 | Precision | Recall | Partial F1 | es F1 | Params | Public LB |
|---|---|---|---|---|---|---|---|---|---|---|
| `ivan-gru1-word-random` | word 50k | – | uni | 0.486 | 0.43 | 0.56 | 0.57 | 0.806 | 6.7M | – |
| `ivan-gru1-bpe16k-random-first` | BPE 16k | first | uni | 0.550 | 0.46 | 0.69 | 0.62 | 0.767 | 2.3M | – |
| `ivan-gru1-bpe16k-random-last` | BPE 16k | last | uni | 0.571 | 0.49 | 0.69 | 0.62 | 0.823 | 2.3M | – |
| `ivan-bigru1-word-random` | word 50k | – | bi | 0.494 | 0.41 | 0.63 | 0.59 | 0.846 | 7.0M | – |
| `ivan-bigru1-bpe16k-random-first` | BPE 16k | first | bi | 0.560 | 0.49 | 0.66 | 0.63 | 0.847 | 2.6M | – |
| **`ivan-bigru1-bpe16k-random-last`** | BPE 16k | last | bi | **0.597** | 0.51 | 0.72 | 0.65 | 0.850 | 2.6M | **0.537** |
| `ivan-bigru1-bpe8k-random-first` | BPE 8k | first | bi | 0.545 | 0.47 | 0.66 | 0.64 | 0.849 | 1.6M | – |
| `ivan-bigru1-bpe32k-random-first` | BPE 32k | first | bi | 0.521 | 0.46 | 0.59 | 0.63 | 0.855 | 4.7M | – |

Only the best run was submitted (one submission instead of eight). **The LB confirms the gain:** 0.537 against the
baseline's 0.459 (+0.078), while CV rose by +0.109. Both runs score about 0.04–0.06 lower on the LB than on CV.

**Insights:**
- **BPE beats word-level in every pairing.** All six BPE runs (CV 0.52–0.60) score above both word-level runs (0.49),
  with 2.6× fewer parameters. One run alone would be inside the CV noise; six out of six is a pattern.
- **Label on the last piece beats the first piece.** For the one-direction GRU, es F1 jumps from 0.767 to 0.823: with
  the first piece, the model must tag `Васильковом` after reading only `василь`. This explains why our first BPE attempt
  on a vanilla RNN failed (CV 0.411). For the BiGRU, es is the same either way, because the backward pass already shows
  the whole word.
- **Bidirectional helps on es (0.806 → 0.846) more than on CV (0.486 → 0.494).** On test-like posts the tokenizer
  matters more than the direction.
- **BPE 16k is the best size on CV** (8k 0.545, 16k 0.560, 32k 0.521), though the differences are within the noise.
- **es and CV disagree again:** es rises until epoch 6, while CV peaks around epochs 2–3. The same domain gap as the
  baseline.
- **What BPE changes:** it finds `Бориспіль`, `ТЦ Аркадія` and `вул. Солом'янська, 28`, loses the Russian forms
  `Фастовском` and `Бучанском`, and makes fewer false positives (precision 0.41 → 0.51).
- **Failed idea:** BPE with first-piece labels on a one-direction model (the first version scored CV 0.411, below
  word-level). The fix was the last-piece label or a bidirectional model.

### Preprocessing and relabelling of the training data
Notebook: [`notebooks/06_preprocessing.ipynb`](notebooks/06_preprocessing.ipynb).

The silver train labels are automatic, and fixing them is allowed (train only; CV and test are never touched). Every
fix is a rule in code, so the file can be rebuilt. The output is one file, `train_v2.parquet`: the whole uk dataset
cleaned and deduplicated, plus a cleaned ru sample of 250k rows (20% of train). `es` keeps the same 10k uk rows,
cleaned the same way.

| Step | What it fixes | Effect |
|---|---|---|
| apostrophes, decomposed letters | `'` `’` `ʼ` `‘` `` ` `` become `'`; `и` + U+0306 becomes `й` (offsets moved) | 777 uk texts with combining marks |
| news-site tails | ` - <rubric> - TCH.` and 10 other site breadcrumbs, whose rubric words were tagged LOC (`Укрaїнa`, `Світ`, `Київ`) | 14,362 tails cut, 5,891 LOC spans dropped |
| hyphenated names | `[Івано][-Франківська]` and `[Ріо]-де-Жанейро` grown to the whole word | 5,565 uk texts relabelled |
| punctuation at span edges | `вулиці.`, `«Київ`, `США (` trimmed to the first and last letter or digit | 2,152 uk + 462 ru texts |
| shifted spans | `[Київщини, зокре]ма`: the whole text is dropped (train rows only) | 1,288 uk + 623 ru texts |
| LOC on names of people | a string tagged PER ≥4× as often as LOC (and ≥20 times) loses its LOC spans (uk only) | 300 strings, 1,205 spans (`Трампа`, `Зеленського`) |
| duplicates and conflicts | exact copies dropped; texts with conflicting labels dropped; train rows also in `es` dropped | uk −2,602 copies, ru −14,079 |
| ru boilerplate | `Об этом сообщает ...` agency credits capped | 20,899 boilerplate rows |

Result: 996k uk + 250k ru train sentences.

### Pretrained fastText + BiGRU + CRF + case feature
Notebook: [`notebooks/08_fasttext_bigru_crf.ipynb`](notebooks/08_fasttext_bigru_crf.ipynb).

One run with every change that should help, instead of an ablation grid:
- **Data:** `train_v2.parquet` (notebook 06), capped at uk 500k + ru 100k sentences for a faster run.
- **Embeddings:** pretrained fastText `cc.uk.300` (300-d, case kept), fine-tuned at 0.1× the learning rate. Its subword
  n-grams give a vector to every word, also unseen towns (`Фастів`, `Баришівка`), so the model can stay word-level.
  The vocabulary is the 200k most frequent train words plus every word of es, CV and test (209k); the extra words only
  get their fastText vector, no label is read.
- **Lookup normalization** (ids only, the output strings are cut from the original text): apostrophe and dash variants
  are unified, and every emoji, bullet and invisible character becomes one `<emoji>` id.
- **Case feature:** 5 classes (`none`, `UPPER`, `Title`, `mixed`, `lower`), embedded (16-d) and concatenated to the word
  vector, since Telegram posts are sometimes all caps.
- **Model:** 2-layer BiGRU (hidden 256) → emission scores → a **self-written CRF** (forward algorithm for the loss,
  Viterbi for decoding, both checked against brute force). 64.7M parameters, 62.6M of them the embedding.
- **Training:** as in notebook 05 (AdamW lr 2e-3, weight decay 0.01, batch 256, clipping 1.0, dropout 0.3, up to 6
  epochs, early stopping on es F1 with patience 2, seed 42); the CRF gets 10× the learning rate. A100, 12–16 minutes.

| Run | CV F1 | Precision | Recall | Partial F1 | es F1 | Best epoch | Public LB |
|---|---|---|---|---|---|---|---|
| `taras-bigru2-word-fasttext-crf-case-data2-uk500k-ru100k` | **0.765** | 0.63 | 0.97 | – | 0.915 | 1 of 4 | 0.598 |
| `taras-bigru2-word-fasttext-crf-case-clean-uk500k-ru100k` | 0.734 | 0.62 | 0.91 | 0.76 | 0.913 | 0 of 3 | **0.60** |

The second run is the same model after the notebook update (lookup normalization, vocabulary rebuilt with the es, CV
and test words). es is cleaned like train, so its F1 is not comparable with notebooks 02–05.

**Insights:**
- **Best LB so far:** 0.60 against 0.537 for the best BPE run (+0.06), and the CV gain is larger (+0.14 to +0.17).
- **The two runs are equal on the LB** (0.598 vs 0.60). The CV drop (0.765 → 0.734) is one location on 26 texts, well
  inside the noise.
- **The CV↔LB gap grew** from 0.04–0.06 (notebooks 02–05) to 0.13–0.17. CV gains are optimistic, so big decisions are
  confirmed on the LB.
- **es and CV disagree again.** es F1 stays at 0.91 every epoch, while CV jumps between 0.66 and 0.74 (and 0.69–0.77 in
  the first run). The best epoch is the first or second, so the model adapts to the news domain quickly.
- **Recall is almost saturated (0.91–0.97); precision is the weak side (0.62).** Of 21 CV errors, 18 are false
  positives:
  - countries: `Україні` in `мир в Україні`, `України` in `міністра оборони України`;
  - bare `Київ` / `Києва` in channel signatures (`Київ — пряма трансляція`) and organisation names (`поліція Києва`);
  - the unlabeled station schedule (`Боярка`, `Тарасівка`, `Київ-Пасажирський`, ...): 8 FPs from one post;
  - `Місяць` (the Moon, capitalised at the start of a sentence).
- **The CRF fixes the tag sequence:** the learned transition `O → I-LOC` is −3.27, and only 1 boundary error is left
  (`Бучанском районах` for `Бучанском`).
- **Next:** the FPs follow the gold convention (a place inside an organisation name or a signature is not labeled), so
  the model-based fixes are an `ORG` tag from `org_markers` and the remaining train label cleanup.

## Ensembling

