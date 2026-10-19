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

> **The post:** A person asks whether a $1,200 annual insurance bill should count entirely in January or as $100 in each month.
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
     Leave unmeasured fields blank until those steps are actually completed.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Run python baseline.py --trained results.json; paste observed results. -->

| Measure                | Baseline | Trained | Difference |
| ---------------------- | -------- | ------: | ---------- |
| Overall accuracy       | TBD      |   0.500 | TBD        |
| Macro F1               | TBD      |   0.307 | TBD        |
| F1 — `concept`         | TBD      |   0.667 | TBD        |
| F1 — `troubleshooting` | TBD      |   0.000 | TBD        |

**What I predicted before I looked:** [Write down your prediction BEFORE running the baseline comparison.]

**What the gap actually means:** [Fill in after baseline evaluation.]

---

## Run Log — Before

<!-- Copy exact results of three-seed evaluation; do not invent seed 7/2024 results. -->

| Criterion                           | Target | Seed 42 | Seed 7       | Seed 2024    | Verdict                  |
| ----------------------------------- | ------ | ------- | ------------ | ------------ | ------------------------ |
| 1. Overall accuracy                 | ≥ 0.60 | 0.500*  | TBD          | TBD          | Pending all seeds        |
| 2. Macro F1                         | ≥ 0.50 | 0.307*  | TBD          | TBD          | Pending all seeds        |
| 3. Minimum F1 for every label       | ≥ 0.30 | 0.000*  | TBD          | TBD          | Pending all seeds        |
| 4. Largest class share              | ≤ 40%  | 35.0%   | Same dataset | Same dataset | MET (dataset)            |
| 5. Independent comparison agreement | ≥ 70%  | TBD     | TBD          | TBD          | Pending agreement report |

*Single-seed initial run shown for context; use the Unit 6 evaluation outputs for the official three-seed report.

### Confusion matrix

<!-- REQUIRED: paste the notebook's actual four-by-four markdown confusion matrix here. -->

| True \\ predicted | `concept` | `advice` | `regulation` | `troubleshooting` |
| ----------------- | --------: | -------: | -----------: | ----------------: |
| `concept`         |       TBD |      TBD |          TBD |               TBD |
| `advice`          |       TBD |      TBD |          TBD |               TBD |
| `regulation`      |       TBD |      TBD |          TBD |               TBD |
| `troubleshooting` |       TBD |      TBD |          TBD |               TBD |

**My biggest off-diagonal number, and what it means:** [Requires real confusion matrix.]

---

## Verdicts and Diagnoses

| #   | Criterion             | Target      | Verdict | How I decided                                  |
| --- | --------------------- | ----------- | ------- | ---------------------------------------------- |
| 1   | Overall accuracy      | ≥ 0.60      | Pending | Compare held-out accuracy for all three seeds. |
| 2   | Macro F1              | ≥ 0.50      | Pending | Compare macro F1 for all three seeds.          |
| 3   | Per-label F1          | ≥ 0.30 each | Pending | All four labels must pass under each seed.     |
| 4   | Class balance         | ≤ 40%       | MET     | Largest label is 70/200 = 35%.                 |
| 5   | Independent agreement | ≥ 70%       | Pending | Requires the Unit 6 comparison set.            |

**Diagnoses:** [Use the actual confusion matrix and agreement disagreements as evidence. The initial seed suggests failure on `advice` and `troubleshooting`, but does not by itself distinguish model limitations from label inconsistency.]

---

## Agreement Report

**Agreement rate:** TBD / 30 = TBD%

<!-- Apply data/staff_taxonomy.md, not the custom TakeMeter taxonomy. Document every disagreement. -->

**Disagreements:** [Paste every staff-vs-personal disagreement, your adjudication, and category (a), (b), or (c).]

**What the pattern in my disagreements tells me:** [Complete after the comparison.]

---

## The Improvement

**What I changed:** [Complete after one measured change.]

**Which diagnosis pointed at it:** [Cite confusion matrix / agreement evidence.]

### Run Log — After

| Criterion                           | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
| ----------------------------------- | ------ | ------- | ------ | --------- | ------- |
| 1. Overall accuracy                 | ≥ 0.60 | TBD     | TBD    | TBD       | TBD     |
| 2. Macro F1                         | ≥ 0.50 | TBD     | TBD    | TBD       | TBD     |
| 3. Minimum F1 for every label       | ≥ 0.30 | TBD     | TBD    | TBD       | TBD     |
| 4. Largest class share              | ≤ 40%  | TBD     | TBD    | TBD       | TBD     |
| 5. Independent comparison agreement | ≥ 70%  | TBD     | TBD    | TBD       | TBD     |

**Did it help, and how do I know:** [Report measured before/after values.]

---

## What's Still Broken

[After improvement, name each still-missed criterion and explain next steps and why work stopped.]

**The gap between what I meant my labels to capture and what the model learned:** [Two sentences grounded in the actual confusion matrix.]

---
---

## How I Used AI

**Moment 1**

- **What I asked for:**
  I asked ChatGPT to help classify financial questions collected from
  Personal Finance & Money Stack Exchange using my four labels:
  `concept`, `advice`, `regulation`, and `troubleshooting`.

- **What came back:**
  ChatGPT generated preliminary labels for 177 financial questions
  and organized the dataset into a CSV file containing three columns:
  `text`, `label`, and `note`.

- **What I changed:**
  I reviewed the classifications using my label taxonomy and marked
  the reviewed examples as `pre-labelled, checked`. I also removed
  unnecessary source metadata from the CSV and standardized the
  column structure.

**Moment 2**

- **What I asked for:**
  While preparing the dataset for training, I encountered a
  `KeyError: 'label'` when running the stratified train/validation/test
  split. I asked ChatGPT to identify the cause and explain how to fix it.

- **What came back:**
  ChatGPT explained that the CSV was missing its header, causing
  pandas to interpret the first financial question as a column name
  instead of recognizing the expected `label` column.

- **What I changed:**
  I corrected the CSV format by adding the required header:

  `text,label,note`

  I then reloaded the dataset and successfully continued the
  DistilBERT training and evaluation workflow.

**Pre-labelling disclosure:**

I initially created 23 cold-labelled examples and used ChatGPT
to pre-label the remaining 177 examples.

The AI-generated labels were subsequently reviewed against my
four-category taxonomy. The dataset records these examples as
`pre-labelled, checked`, while the original 23 examples are marked
as `cold; checked`.

AI assistance was also used for CSV formatting, debugging,
and documentation. The final dataset contains 200 labelled
financial questions.
**Pre-labelling disclosure:**

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure          | Baseline | Trained | Difference |
| ---------------- | -------- | ------- | ---------- |
| Overall accuracy |          |         |            |
| Macro F1         |          |         |            |
| F1 — `label_one` |          |         |            |
| F1 — `label_two` |          |         |            |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
| --------- | ------ | ------- | ------ | --------- | ------- |
| 1.        |        |         |        |           |         |
| 2.        |        |         |        |           |         |
| 3.        |        |         |        |           |         |
| 4.        |        |         |        |           |         |
| 5.        |        |         |        |           |         |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |     |     |     |
| ---------------- | --- | --- | --- |
| ****             |     |     |     |
| ****             |     |     |     |
| ****             |     |     |     |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| #   | Criterion | Target | Verdict | How I decided |
| --- | --------- | ------ | ------- | ------------- |
| 1   |           |        |         |               |
| 2   |           |        |         |               |
| 3   |           |        |         |               |
| 4   |           |        |         |               |
| 5   |           |        |         |               |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
| --------- | ------ | ------- | ------ | --------- | ------- |
| 1.        |        |         |        |           |         |
| 2.        |        |         |        |           |         |
| 3.        |        |         |        |           |         |
| 4.        |        |         |        |           |         |
| 5.        |        |         |        |           |         |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



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
