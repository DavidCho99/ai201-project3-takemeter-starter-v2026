# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** for setup, the notebook, the baseline, and troubleshooting.
>
> Once `python test.py` passes, check the expected CSV format with `head -5 data/practice_labels.csv`. Open `takemeter.ipynb` in this folder and select this project's `.venv` kernel.
>
> The notebook runs locally; no accounts or API keys are required.

---

<!-- UNIT 5 — THE BUILD -->

## What This Does

TakeMeter is a text classifier for questions from the Personal Finance & Money Stack Exchange community. It identifies a question's primary purpose: understanding a financial **concept**, seeking financial **advice**, interpreting a **regulation**, or **troubleshooting** an account, payment, or software issue.

The project uses 200 reviewed, labeled questions and fine-tunes `distilbert-base-uncased` for four-class classification. It evaluates not just overall accuracy but also macro F1, performance by label, and prediction confidence.

---

## Label Taxonomy

### `concept`

**Definition:** A question primarily asking for an explanation of financial principles, calculations, or how a financial mechanism works.

**Examples:**
> Why is Market cap higher than equity/book value?
>
> What exactly is the source of the equity risk premium?

### `advice`

**Definition:** A question primarily seeking a recommendation, strategy, or guidance for making a financial choice.

**Examples:**
> How to invest long-term when moving countries frequently?
>
> How should I split my monthly income between savings, expenses, and investments?

### `regulation`

**Definition:** A question primarily asking about the interpretation or application of tax law, financial regulation, insurance rules, or legal obligations.

**Examples:**
> Combining Schedule C and Schedule E business miles for the 50% business-use test
>
> Tax Reporting of options on GLD

### `troubleshooting`

**Definition:** A question primarily trying to diagnose or resolve a particular problem with a financial account, transaction, payment, or application.

**Examples:**
> How can I access my 401k?
>
> Credit card - Transaction was voided but hasn’t reflected

### The hardest boundary

**Which two labels:** `advice` and `troubleshooting`.

**The decision rule:** I use `troubleshooting` when a post describes an existing problem, error, failed transaction, or unexpected account behavior and asks for a fix. I use `advice` when the post asks for a recommendation or strategy, without primarily attempting to fix a malfunction. If both seem relevant, I classify the primary question rather than isolated keywords. If the primary question concerns what a tax, legal, or insurance rule allows or requires, `regulation` takes precedence.

For example, retrieving an inaccessible 401(k) account is `troubleshooting`; deciding how to invest retirement savings is `advice`.

---

## The Dataset

**Where the posts came from:**

The dataset contains 200 questions from **Personal Finance & Money Stack Exchange** (`money.stackexchange.com`). Topics include taxes, investing, retirement, insurance, budgeting, banking, payments, and financial applications. The training CSV contains the full collected question text, its classification, and review metadata in exactly three columns: `text,label,note`.

**How I labelled them:**

The first 23 questions are recorded as `cold; checked`, meaning they were initially labeled separately before the AI-prelabeling batch. An AI assistant provided preliminary classifications for another 177 posts, which are recorded as `pre-labelled, checked`. I subsequently checked the labels using the taxonomy above. These metadata values are preserved in the `note` column. AI assistance is disclosed below.

**Counts per label:**

| Label             |   Count |    Share |
| ----------------- | ------: | -------: |
| `concept`         |      64 |    32.0% |
| `advice`          |      37 |    18.5% |
| `regulation`      |      70 |    35.0% |
| `troubleshooting` |      29 |    14.5% |
| **Total**         | **200** | **100%** |

The largest category is `regulation` at 35%; no label exceeds 70%.

### Three hard cases

**1. How can I access my 401k?**

> **The post:** A former Google employee wants to access deferred vested benefits in an old 401(k).
>
> **Could have been:** `advice` or `troubleshooting`.
>
> **I chose:** `troubleshooting`, because the immediate request is to recover access to an existing account, not to choose a retirement strategy.

**2. Can HealthCare.gov force-change my plan variant mid-year due to an income increase?**

> **The post:** An insurance enrollee reports an unexpected plan-variant change and asks whether it is permitted after an income update.
>
> **Could have been:** `troubleshooting` or `regulation`.
>
> **I chose:** `regulation`, because the central question is what Marketplace eligibility and coverage rules legally permit, even though it started with a practical account problem.

**3. How should an annual insurance payment be recorded in a monthly budget?**

> **The post:** A person asks whether a \$1,200 annual insurance bill should count entirely in January or as \$100 in each month.
>
>
> **Could have been:** `concept` or `advice`.
>
> **I chose:** `advice`, because they want a practical budgeting method to use, rather than primarily asking how a financial product works.

---

## The Training Run

**Base model:** `distilbert-base-uncased` (DistilBERT).

**Settings:**

| Setting              | Value      |
| -------------------- | ---------- |
| Epochs               | 3          |
| Learning rate        | `2e-5`     |
| Batch size           | 16         |
| Maximum input length | 128 tokens |
| Seed                 | 42         |
| Labels               | 4          |

**Anything I changed from the defaults, and why:**

I replaced the starter labels (`analysis`, `hot_take`, and `reaction`) with my dataset's four categories: `concept`, `advice`, `regulation`, and `troubleshooting`. For this initial training run I otherwise kept the default hyperparameters to provide a reproducible starting point.

**Split sizes:** Stratified 70/15/15 split of 200 questions: **140 train, 30 validation, 30 held-out test**.

| Test label        | Examples |
| ----------------- | -------: |
| `concept`         |       10 |
| `advice`          |        6 |
| `regulation`      |       10 |
| `troubleshooting` |        4 |
| **Total**         |   **30** |

The test subsets for `advice` and `troubleshooting` have fewer than eight examples, so per-label scores are especially sensitive to individual mistakes and to the random seed.

**Initial held-out evaluation (seed 42):**

| Metric                          | Result |
| ------------------------------- | -----: |
| Overall accuracy                |  0.500 |
| Macro F1                        |  0.307 |
| F1 — `concept`                  |  0.667 |
| F1 — `advice`                   |  0.000 |
| F1 — `regulation`               |  0.562 |
| F1 — `troubleshooting`          |  0.000 |
| Accuracy, most-confident third  |  0.800 |
| Accuracy, least-confident third |  0.200 |

The model is substantially stronger on `concept` and `regulation` than on `advice` or `troubleshooting`. A zero F1 does not by itself establish whether a class was never predicted or was predicted only incorrectly; the confusion matrix is needed to determine which. Class imbalance, small sample sizes, overlapping intents, and truncation at 128 tokens are possible explanations, not confirmed causes. More-confident predictions were more accurate in this single split, although 30 examples are not enough to establish reliable calibration.

The notebook wrote `results.json` and `test_split.csv` as evidence of this run; these files should be committed.

---

## How I Used AI

**Moment 1 — Dataset preparation and prelabeling**

- **What I asked for:** Help formatting Personal Finance & Money Stack Exchange questions into an exact three-column CSV (`text,label,note`) and obtaining initial classifications for the additional posts.
- **What came back:** A consistent CSV with 200 questions, including 23 separately labeled examples and 177 AI-prelabeled examples.
- **What I changed:** I requested removal of source/author strings from the `note` field, asked for the initial 23 examples to be added, and standardized the review metadata (`cold; checked` / `pre-labelled, checked`).

**Moment 2 — Fixing a data-loading error**

- **What I asked for:** Help understanding a `KeyError: 'label'` raised while performing the stratified split.
- **What came back:** A diagnosis that the DataFrame did not contain a `label` column, with instructions to check the CSV header and reload the data.
- **What I changed:** I checked the expected `text,label,note` structure and used the corrected CSV format for the notebook workflow.

**Pre-labelling disclosure:** AI suggested labels for 177 questions; the other 23 questions are marked `cold`. The 177 AI-prelabeled entries are marked `pre-labelled, checked`. AI also assisted with CSV formatting, debugging, and documentation drafting. The trained classifier's reported results come from the local notebook evaluation, not AI-generated estimates.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->


## Unit 6 — Testing

### Baseline vs. Trained

The zero-shot baseline and the fine-tuned model were compared on the same 30 held-out financial questions.

| Measure                | Baseline | Trained | Difference (trained − baseline) |
| ---------------------- | -------: | ------: | ------------------------------: |
| Overall accuracy       |    0.467 |   0.500 |                          +0.033 |
| Macro F1               |    0.474 |   0.307 |                          −0.166 |
| F1 — `advice`          |    0.444 |   0.000 |                          −0.444 |
| F1 — `concept`         |    0.462 |   0.667 |                          +0.205 |
| F1 — `regulation`      |    0.625 |   0.562 |                          −0.062 |
| F1 — `troubleshooting` |    0.364 |   0.000 |                          −0.364 |

*The reported differences are copied from the baseline comparison; minor discrepancies from subtraction reflect rounding.*

**What I predicted before I looked:** I did not preserve a timestamped prediction before inspecting the baseline results, so I cannot honestly claim a pre-result prediction.

**What the gap actually means:** Fine-tuning increased overall accuracy by 3.3 percentage points (46.7% to 50.0%) in this held-out split, but macro F1 decreased from 0.474 to 0.307. Fine-tuning helped on `concept` while weakening `advice`, `regulation`, and `troubleshooting`. The model does not demonstrate a balanced improvement over the zero-shot baseline.

### Run Log — Before

The three-seed runs used seeds **42, 7, and 2024**. The evaluation device and its exact name are recorded in `results_three_seeds_before.json` and should be reported from that file, rather than inferred from the computer used.

| Measure                | Seed 42 | Seed 7 | Seed 2024 | Spread (max − min) |
| ---------------------- | ------: | -----: | --------: | -----------------: |
| Overall accuracy       |   0.500 |  0.333 |     0.467 |              0.167 |
| Macro F1               |   0.307 |  0.125 |     0.282 |              0.182 |
| F1 — `concept`         |   0.667 |  0.000 |     0.571 |              0.667 |
| F1 — `advice`          |   0.000 |  0.000 |     0.000 |              0.000 |
| F1 — `regulation`      |   0.562 |  0.500 |     0.556 |              0.062 |
| F1 — `troubleshooting` |   0.000 |  0.000 |     0.000 |              0.000 |

**Original acceptance criteria across three runs:**

| Criterion                                         | Original target      |      Seed 42 |       Seed 7 |    Seed 2024 | Verdict                                        |
| ------------------------------------------------- | -------------------- | -----------: | -----------: | -----------: | ---------------------------------------------- |
| 1. Overall accuracy                               | ≥ 0.60 in every seed |        0.500 |        0.333 |        0.467 | **MISSED**                                     |
| 2. Macro F1                                       | ≥ 0.50 in every seed |        0.307 |        0.125 |        0.282 | **MISSED**                                     |
| 3. Minimum F1 across all four labels              | ≥ 0.30 in every seed |        0.000 |        0.000 |        0.000 | **MISSED**                                     |
| 4. Largest dataset class share                    | ≤ 40%                |        35.0% |       35.0%* |       35.0%* | **MET**                                        |
| 5. Model predictions vs. independent human labels | ≥ 70% exact match    | Not measured | Not measured | Not measured | **NOT MEASURABLE with the supplied staff set** |

*Criterion 4 is a property of the original 200-post dataset, not of the seed. The same 35% balance applies to all runs using that unchanged dataset. Criterion 5 likewise is not inherently a per-seed metric; it requires predictions on a suitable independently labeled financial-question set.*

#### Confusion matrix — before improvement (seed 42)

| True \ predicted  | `concept` | `advice` | `regulation` | `troubleshooting` |
| ----------------- | --------: | -------: | -----------: | ----------------: |
| `concept`         |         6 |        0 |            4 |                 0 |
| `advice`          |         1 |        0 |            5 |                 0 |
| `regulation`      |         1 |        0 |            9 |                 0 |
| `troubleshooting` |         0 |        0 |            4 |                 0 |

**My biggest off-diagonal number, and what it means:** The largest single error is **five true `advice` questions classified as `regulation`**. In the reverse direction, only **one** true `regulation` question was labeled `concept`, and none was predicted `advice`. All four `troubleshooting` posts were also misclassified as `regulation`. In this split, the model never predicted `advice` or `troubleshooting` at all, suggesting it has not learned usable decision boundaries for these classes under this training setup. The confusion matrix is a **seed-42** result, not an aggregate matrix for all seeds.

### Verdicts and Diagnoses

| #   | Criterion             | Original target                                              | Verdict                                    | How I decided                                                                                                                                                                         |
| --- | --------------------- | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Overall accuracy      | ≥ 0.60 in each of three seeds                                | **MISSED**                                 | The measured accuracies were 0.500, 0.333, and 0.467; none reached 0.60.                                                                                                              |
| 2   | Macro F1              | ≥ 0.50 in each of three seeds                                | **MISSED**                                 | The macro-F1 values were 0.307, 0.125, and 0.282, below 0.50 in every run.                                                                                                            |
| 3   | Minimum per-label F1  | ≥ 0.30 for every label in each seed                          | **MISSED**                                 | Both `advice` and `troubleshooting` had F1 = 0.000 in all three runs.                                                                                                                 |
| 4   | Dataset class balance | Largest class ≤ 40%                                          | **MET**                                    | The largest category is `regulation`: 70 of 200 = 35%.                                                                                                                                |
| 5   | Model/human agreement | ≥ 70% of model predictions matching independent human labels | **NOT TESTABLE AS WRITTEN WITH STAFF SET** | Staff posts use `analysis` / `hot_take` / `reaction`, whereas my model predicts four different financial categories; the 93.3% staff agreement measures my annotations, not my model. |

**Diagnoses:**

**Criterion 1 — Overall accuracy:** The 16.7-percentage-point spread between seeds (0.333–0.500) exceeds ten percentage points, showing substantial sensitivity to seed-dependent training and held-out samples. Each held-out set has only 30 questions, so even a few changed predictions strongly affect accuracy. Small test size and unstable learning are plausible explanations; the data do not isolate one cause.

**Criterion 2 — Macro F1:** The zero-score minority classes are the immediate reason the model's macro F1 remained low. The model performed relatively better on `regulation` and sometimes `concept`, but not across all four purposes. The original baseline's macro F1 (0.474) also exceeded the trained model's (0.307), so training has not demonstrably improved balanced classification.

**Criterion 3 — Every-label F1:** In the seed-42 confusion matrix, the model made **zero predictions** of either `advice` or `troubleshooting`, despite six and four true examples respectively. It classified five of six `advice` examples and all four `troubleshooting` examples as `regulation`. The dataset has only 37 `advice` and 29 `troubleshooting` posts versus 70 `regulation` and 64 `concept`, which may contribute to learning difficulty, but imbalance alone has not been proven causal. Borderlines between legal questions, requests for financial choices, and requests to fix an account may also be difficult for the model.

**Criterion 4 — Class balance:** The criterion is met on the original dataset, although being under 40% does not guarantee equal model performance.

**Criterion 5 — Measurement mismatch:** The original text in `criteria.md` requires **the trained model** to agree with independent humans on a compatible set. The Unit 6 staff exercise measures **me versus staff annotators** under a different taxonomy, and its 28/30 result therefore cannot satisfy or fail this criterion. I retain the original criterion unchanged and report it as not testable with the provided staff data; a future valid test would require independent human labels for held-out **financial questions using the same four labels** and predictions from this classifier on those exact questions.

**What the staff agreement adds to the diagnosis:** The two staff disagreements show that even with well-defined rules, I sometimes overemphasized the mere existence of a number or undervalued a small but relevant observation. This is **indirect evidence** of a possible annotation-process weakness; it does not prove that my original 200 financial posts contain incorrect labels. Reviewing specific model errors against the original four-label decision rules would be necessary to test that explanation.

### Agreement Report

**Agreement rate:** **28 / 30 = 93.3%** (shown as 93% in `agreement.py`). The 30 running-forum posts were classified using the **staff taxonomy**, not TakeMeter's finance taxonomy. The original independent labels in `my_staff_labels.csv` remain unchanged.

**Disagreement 1 — `sp_010`**

> **The post:** “Three of us ran the same session on the same day, different shoes, within 10 seconds of each other over 5 miles. I'm not saying shoes do nothing, but whatever they do was smaller than the noise between three people.”
>
> **Staff said:** `analysis`  
> **I said:** `hot_take`

**My call, and why:** I now think **`analysis` is the better-supported reading** under Rule 1. I originally treated the conclusion about running shoes as too broad for a comparison among only three runners. But the staff rule asks whether a checkable fact **carries the argument**, not whether the evidence proves a universal claim. The ten-second observation is the basis for the writer's qualified conclusion. The small sample limits confidence in the claim, but it does not make the post an unsupported assertion.

**Which pile:** **(a) Rule covered it; I applied it loosely.** I confused the strength of the evidence with whether the post provides relevant evidence at all.

**Disagreement 2 — `sp_014`**

> **The post:** “Times are down about 3% since the shoe rules changed, so obviously the shoes are the whole story and the sport has a problem.”
>
> **Staff said:** `hot_take`  
> **I said:** `analysis`

**My call, and why:** I favor **`hot_take`** after applying Rule 1 more carefully. My initial `analysis` label focused on the 3% statistic and the mention of the shoe rule. However, a change happening after a rule change does not show that shoes are the *whole* explanation. The writer jumps to an absolute judgment (“obviously the shoes are the whole story”), using the number mainly as emphasis rather than as a sufficient argument. This case closely matches the staff taxonomy's explicit example of a decorative statistic.

**Which pile:** **(a) Rule covered it; I applied it loosely.** I mistakenly treated the presence of a statistic as proof that the post was analytical.

**What the pattern in my disagreements tells me:** Both errors occurred at the **`analysis` vs. `hot_take`** boundary, and neither was caused by a missing decision rule. In one case I discounted relevant evidence because it came from a small sample; in the other I overvalued a number that did not do the argumentative work. I need to judge **how evidence is used**, not merely whether numbers appear. The high agreement rate suggests I could usually follow this staff taxonomy, but this separate running-forum task does not directly validate my own four-label financial dataset.

### The Improvement

**What I changed:** **Not yet performed.** The assignment permits one targeted change; do not report an unperformed experiment as completed.

**Which diagnosis pointed at it:** The strongest candidate is poor recognition of `troubleshooting` (29 examples and F1 = 0 across all three seeds). A potential single change would be **adding 30 genuinely new, independently reviewed `troubleshooting` examples**, leaving the taxonomy and training hyperparameters unchanged. This is a proposal only; confirm it against individual prediction errors before implementing it. Alternatively, one targeted correction of labeling inconsistency at a clearly defined boundary could be chosen, but do **not** combine improvements.

---

### Run Log — After

| Criterion                            | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict      |
| ------------------------------------ | ------ | ------- | ------ | --------- | ------------ |
| 1. Overall accuracy                  | ≥ 0.60 | 0.571   | 0.343  | 0.486     | MISSED       |
| 2. Macro F1                          | ≥ 0.50 | 0.446   | 0.172  | 0.362     | MISSED       |
| 3. Minimum per-label F1              | ≥ 0.30 | 0.000   | 0.000  | 0.000     | MISSED       |
| 4. Dataset class balance             | ≤ 40%  | 30.4%   | 30.4%  | 30.4%     | MET          |
| 5. Independent model-human agreement | ≥ 70%  | N/A     | N/A    | N/A       | NOT MEASURED |
**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->
The improvement produced better observed results across all three seeds.

Overall accuracy increased from 0.500 to 0.571 for seed 42, from 0.333 to 0.343 for seed 7, and from 0.467 to 0.486 for seed 2024.

Macro F1 increased from 0.307 to 0.446, from 0.125 to 0.172, and from 0.282 to 0.362.

Most importantly, troubleshooting F1 increased from 0.000 across all three seeds to 0.500, 0.200, and 0.714.

These results support the hypothesis that additional troubleshooting examples were useful in this experiment.

However, advice F1 remained at 0.000 across all three seeds. The improvement did not solve the classifier’s inability to recognize advice-seeking questions.

The accuracy spread also increased from 0.167 before the improvement to 0.229 afterward, suggesting that the model remained sensitive to the random seed and data split.

The original dataset contained 200 questions, while the improved dataset contained 230. As a result, the held-out test sets changed from 30 to 35 examples. These before-and-after comparisons are therefore descriptive rather than a controlled evaluation on identical test posts.

The improvement helped the targeted category in the observed runs, but none of the original model-performance targets were met consistently across all three seeds.


---

### What’s Still Broken

### 1. Overall Accuracy — MISSED (Target: ≥ 0.60)

After adding 30 troubleshooting examples, accuracy improved across all three seeds, reaching 0.571, 0.343, and 0.486. However, none met the original 0.60 target. The accuracy spread also increased from 0.167 to 0.229, indicating that performance remains sensitive to the random seed and dataset split.

In future work, I would examine individual misclassifications and evaluate the model on a fixed held-out test set to separate the effects of additional training data from changes in test composition.

### 2. Macro F1 — MISSED (Target: ≥ 0.50)

Macro F1 improved from 0.307, 0.125, and 0.282 to 0.446, 0.172, and 0.362. Nevertheless, every seed remained below the 0.50 target.

The main problem is uneven performance across labels. Although troubleshooting improved, advice remained at an F1 score of 0.000 across all three seeds. I would investigate whether the model confuses advice-seeking questions with regulatory questions and review the relevant training examples for labeling consistency.

### 3. Minimum Per-Label F1 — MISSED (Target: ≥ 0.30)

The model still failed to achieve an F1 score of at least 0.30 for every category.

Before the improvement, both advice and troubleshooting had F1 scores of 0.000 across all three seeds. After adding 30 troubleshooting examples, troubleshooting F1 increased to 0.500, 0.200, and 0.714. However, advice remained at 0.000 in every run.

A future improvement would focus on reviewing the advice training examples, particularly the boundary between advice and regulation, and collecting additional examples if necessary.

### 4. Dataset Class Balance — MET (Target: ≤ 40%)

The largest category, regulation, represents 70 of the 230 examples, or approximately 30.4%. This satisfies the original class-balance criterion.

However, improved class balance alone did not guarantee strong performance across all four categories.

### 5. Independent Model-Human Agreement — NOT MEASURED (Target: ≥ 70%)

The original criterion required the model’s predictions to agree with independently assigned human labels.

The provided staff exercise instead compared my manually assigned labels against staff labels using a different taxonomy. I achieved 28 agreements out of 30 posts (93.3%), but this measures human annotation agreement rather than the financial classifier’s accuracy against an independently labeled dataset.

To evaluate the original criterion, I would need a separate set of financial questions labeled independently under my four-category taxonomy.

Why I Stopped

I stopped after one targeted improvement because the assignment required a single change followed by a measured before-and-after comparison.

Adding 30 troubleshooting examples improved the observed troubleshooting F1 scores, but further changes to the dataset, taxonomy, or training hyperparameters would introduce additional experimental variables.

The before-and-after runs also used different held-out test sets because the dataset increased from 200 to 230 examples. Therefore, the observed improvements cannot be attributed exclusively to the additional examples without a controlled evaluation.

The gap between what I meant my labels to capture and what the model learned:

I intended TakeMeter to distinguish financial explanations, recommendations, regulatory interpretations, and solutions to account or transaction problems, but the original confusion matrix showed that five of six advice questions and all four troubleshooting questions were misclassified as regulation or another category.

After adding 30 troubleshooting examples, the model began recognizing troubleshooting questions, but advice F1 remained at 0.000 across all three seeds, showing that the model still had not learned all four intended distinctions reliably.



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**
