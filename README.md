# SMS Spam Classification using SVM

Classifies SMS messages as **spam** or **ham** (legitimate) using TF-IDF features and a linear Support Vector Machine (`LinearSVC`).

- **Notebook:** `SMS_Spam_SVM.ipynb`
- **Dataset:** [SMS Spam Collection on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,572 messages, about 13% spam)

## Pipeline

1. **Load data**: reads the Kaggle CSV (`v1` = label, `v2` = message). Works in Google Colab (upload prompt) and locally (`DATA_PATH`).
2. **Data quality**: checks missing values and removes the 403 duplicate messages *before* splitting, so identical texts cannot leak between train and test.
3. **EDA**: class distribution, message length statistics and spam/ham word clouds.
4. **Preprocessing**: lowercasing, URL / e-mail / number / currency replaced by placeholder tokens (kept as spam signals), punctuation removal, stopword removal, lemmatization.
5. **Split**: stratified 80/20 train/test split (`random_state=42`).
6. **Baseline**: TF-IDF (unigrams + bigrams, 5,000 features) + `LinearSVC`.
7. **Tuning**: `GridSearchCV` with stratified 5-fold CV, optimising **spam-class F1** over n-gram range, vocabulary size, sublinear TF, `C` and `class_weight`.
8. **Evaluation**: accuracy, precision, recall, F1, ROC AUC, confusion matrix and ROC curve on the untouched test set.
9. **Cross-validation** on the training set only, plus **error analysis** (false positives/negatives) and the most influential spam/ham words.
10. **Inference**: the model is saved to `spam_svm_model.joblib` and `predict_sms(text)` classifies new messages.

## What changed from the original notebook

| Area | Before | Now |
|---|---|---|
| Data loading | Colab-only `files.upload()` | Works in Colab and locally via `DATA_PATH` |
| Duplicates | Dropped after EDA, inside the cleaning cell | Dropped right after loading, so EDA matches the modelling data |
| Cleaning | URLs, numbers and currency deleted | Replaced with placeholder tokens that preserve spam signals |
| Tuning | `C` only, optimised accuracy | Several parameters, optimised spam-class F1, with `class_weight` option |
| Evaluation | Imported confusion matrix / ROC but never used them | Confusion matrix, ROC curve, ROC AUC added |
| Cross-validation | Run on the full dataset, which includes the test rows | Run on the training set only; the test set stays held out |
| Extras | none | Error analysis, top spam/ham words, saved model |
| Code hygiene | Unused imports, duplicate import, no seed | Cleaned imports, fixed random seed, `.reset_index` after dedupe |

## Original baseline results (for reference)

From the original notebook run on the Kaggle data: accuracy 0.9845, spam precision 0.983, spam recall 0.893, spam F1 0.936. The main weakness was spam recall, which the updated tuning targets. Re-run the notebook to get the updated numbers.

## Running it

```bash
pip install wordcloud nltk scikit-learn seaborn pandas matplotlib joblib
jupyter notebook SMS_Spam_SVM.ipynb
```

Put the Kaggle CSV next to the notebook and set `DATA_PATH` in the "Load Dataset" cell (default `SMS.csv`; the Kaggle file is named `spam.csv`).

## Using the saved model

```python
import joblib
model = joblib.load("spam_svm_model.joblib")
# input must be cleaned with the notebook's clean_text() first
model.predict([clean_text("Congratulations! You have won a free iPhone.")])  # 1 = spam
```

## Possible next steps

- Compare against Logistic Regression, Multinomial Naive Bayes and a calibrated SVM.
- Add handcrafted features (message length, share of uppercase letters, digit count).
- Tune the decision threshold on `decision_function` to trade precision for recall.

Author **Nair Nazmim Roja**
