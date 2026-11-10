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

**CV vs LB:** the baseline scored CV 0.488 and LB 0.459, inside the CV interval. The correlation over all submitted runs
is tracked in notebook 02, section 5.

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

## Ensembling

