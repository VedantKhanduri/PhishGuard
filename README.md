# PhishGuard — Final Project Package

An explainable, local machine-learning prototype for estimating phishing risk in messages and analysing batches from CSV files.

## Features

- Single-message analysis
- Batch CSV analysis (up to 5,000 rows in this version)
- Calibrated Linear SVM + TF-IDF
- Estimated phishing risk and complementary score
- Rule-based indicators and primary threat-type labels
- Risk distribution chart
- Downloadable CSV results
- Local inference; no external AI API required

## Requirements

- Python 3.10 or newer (tested design target: Python 3.11)
- Dataset file used to train the model

## Setup on Windows

Open PowerShell in the PhishGuard folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation in the current terminal, you can run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## Dataset

Place the dataset at one of these paths:

- `data/phishing_email_clean.csv` (preferred)
- `data/phishing_email.csv`

Required columns:

- `text_combined`: message body/text
- `label`: `0` for legitimate and `1` for phishing

The training script performs basic cleaning and removes duplicate text/label rows. Only use a dataset whose labels and licence you have checked.

## Train the model

```powershell
python src/train_model.py
```

This creates:

- `models/phishguard_calibrated.pkl`
- `models/phishguard_tfidf.pkl`
- `models/calibrated_metrics.json`

## Run the app

```powershell
streamlit run app.py
```

## Batch CSV input

Upload a CSV with a message column named one of:

`text`, `message`, `email`, `email_text`, `text_combined`, `content`, `body`, `sms`

If the column name is different, select the correct column in the UI.

## Important limitations

- Risk scores are model estimates, not guarantees or proof of maliciousness/safety.
- The complementary trust score is calculated as `100 - estimated risk`; it is not an independently measured probability of safety.
- Threat categories and indicators are transparent keyword/pattern heuristics, not a separate trained threat-type classifier.
- A low-risk prediction must not be treated as a safety guarantee.
- The model's training corpus may differ from real SMS, chat messages, newer phishing campaigns, languages, and URL behaviour. External validation is recommended before any real-world security use.
- This project does not fetch or visit links and does not analyse attachments.
