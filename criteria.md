# Acceptance Criteria — TakeMeter

These five acceptance criteria define success for TakeMeter **before model training**. TakeMeter classifies 200 Personal Finance & Money Stack Exchange questions as `concept`, `advice`, `regulation`, or `troubleshooting`.

## 1. Overall accuracy

The model must achieve **at least 60% accuracy** on the held-out test set **in each of the three training seeds**.

**Why this target:** With four labels, a uniform random guess would average 25% accuracy. A 60% target would indicate that the model learned useful differences in question intent rather than merely guessing, while allowing for ambiguity in a test set of approximately 30 posts.

---

## 2. Macro F1 score

The model must achieve a **macro F1 score of at least 0.50** on the held-out test set **in each of the three seeds**.

**Why this target:** My dataset contains 70 `regulation` posts and 64 `concept` posts, but only 37 `advice` posts and 29 `troubleshooting` posts. Macro F1 weights the four labels equally, so the classifier cannot meet this goal simply by performing well on the two larger categories.

---

## 3. Minimum per-label F1

**Every one of the four labels must have an F1 score of at least 0.30** on the held-out test set **in each of the three seeds**.

**Why this target:** `troubleshooting` has only 29 examples, so a 15% test split will typically contain about four or five such posts. One incorrect prediction can substantially change that label's F1. This minimum is deliberately lower than the macro-F1 target, but still requires the model to identify examples from every category.

---

## 4. Dataset class balance

**No label may account for more than 40%** of the 200 examples in `labels.csv`, measured before training.

**Why this target:** The current counts are 70 `regulation` (35%), 64 `concept` (32%), 37 `advice` (18.5%), and 29 `troubleshooting` (14.5%). A 40% cap accommodates this naturally uneven distribution while preventing a single category from dominating the training data.

---

## 5. Agreement with independent human labels

On the independently labeled comparison set supplied in Unit 6, **at least 70% of the model's predictions must match the human labels**, measured as the number of exact label matches divided by the number of compared posts.

**Why this target:** The boundary between `advice` and `troubleshooting` can be subjective: a question describing a problem may ask either for a recommendation or for a fix. Requiring 70% exact agreement tests whether the taxonomy is understandable beyond my own labeling, without assuming perfect agreement on borderline questions.
