# 🛡️ PhishGuard

**A Machine Learning-Based Phishing and Scam Risk Assessment Tool**

PhishGuard is a Python-based application that helps assess suspicious messages and URLs using machine learning and heuristic checks.

## ✨ Features

- **Single Message Analysis:** Assess suspicious email or message text.
- **Batch CSV Analysis:** Analyse multiple messages from a CSV file.
- **URL Scanner:** Inspect URLs using heuristic-based checks.
- **Screenshot OCR:** Extract text from screenshots using EasyOCR and assess the extracted content.
- **Model Analytics:** View model evaluation metrics and a confusion matrix.

## 🧰 Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- EasyOCR
- Pandas

## ⚙️ Run Locally

1. Clone this repository:

   ```bash
   git clone https://github.com/VedantKhanduri/PhishGuard.git
   cd PhishGuard
   ```

2. Create and activate a virtual environment.

   Windows PowerShell:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   streamlit run app.py
   ```

## 🔒 Privacy and Security

- Model inference runs locally when the application is run on your own machine.
- Risk scores are estimates, not proof that a message or URL is safe.
- URL checks may use heuristics and should not be treated as a guarantee of safety.

## 📊 Model Evaluation

The repository includes saved evaluation metrics. These reflect a held-out dataset split and do not guarantee performance on new, real-world phishing messages.

## ⚠️ Disclaimer

PhishGuard is an educational project intended for security awareness and experimentation. Always verify suspicious messages and links independently.

## 👨‍💻 Author

**Vedant Khanduri**

GitHub: [@VedantKhanduri](https://github.com/VedantKhanduri)
