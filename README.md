🛡️ PhishGuard

A Machine Learning-Based Phishing and Scam Risk Assessment Tool

PhishGuard is a Python-based application designed to assess suspicious messages and URLs using machine learning and heuristic-based checks.

✨ Features

- Single Message Analysis: Assess suspicious emails and messages.
- Batch CSV Analysis: Analyse multiple messages from a CSV file.
- URL Scanner: Inspect URLs using heuristic-based checks.
- Screenshot OCR: Extract text from screenshots using EasyOCR and assess the extracted content.
- Model Analytics: View model evaluation metrics and a confusion matrix.

🧰 Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- EasyOCR
- Pandas

⚙️ Run Locally

1. Get the Project

Download the project files from this repository.

2. Create a Virtual Environment

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

3. Install Dependencies

pip install -r requirements.txt

4. Start the Application

streamlit run app.py

🔒 Privacy and Security

- Model inference runs locally when the application is run on your own machine.
- Risk scores are estimates, not proof that a message or URL is safe.
- URL checks use heuristic-based analysis and cannot guarantee safety.

📊 Model Evaluation

The project includes saved model evaluation metrics based on a held-out dataset split. These results do not guarantee the same performance on new, real-world phishing messages.

⚠️ Disclaimer

PhishGuard is an educational project intended for security awareness and experimentation. Always verify suspicious messages and links independently.
