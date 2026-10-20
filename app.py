from pathlib import Path
import hashlib
import ipaddress
import json
import math
import re
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# EasyOCR is optional. Other app features remain available without it.
try:
    from PIL import Image
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    Image = None
    easyocr = None
    OCR_AVAILABLE = False


# -------------------- CONFIGURATION --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MAX_ANALYSIS_ROWS = 5000
MAX_UPLOAD_MB = 10

MODEL_FILES = [
    MODEL_DIR / "phishguard_calibrated.pkl",
    MODEL_DIR / "phishguard_svm.pkl",
    MODEL_DIR / "phishing_model.pkl",
]
VECTORIZER_FILES = [
    MODEL_DIR / "phishguard_tfidf.pkl",
    MODEL_DIR / "tfidf_vectorizer.pkl",
]
METRIC_FILES = [
    MODEL_DIR / "calibrated_metrics.json",
    MODEL_DIR / "model_metrics.json",
]

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
.stApp {background:#0b1020;color:#edf2ff;}
[data-testid="stSidebar"] {background:#10182b;}
.hero {
    padding:22px; border-radius:16px;
    background:linear-gradient(120deg,#16213e,#103544);
    border:1px solid #334155; margin-bottom:20px;
}
.hero h1 {margin:0;}
.hero p {color:#cbd5e1;margin-bottom:0;}
div[data-testid="stMetric"] {
    background:#16213e; padding:12px;
    border-radius:12px; border:1px solid #334155;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD SAVED FILES --------------------

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)


@st.cache_resource
def load_assets():
    model_path = first_existing(MODEL_FILES)
    vectorizer_path = first_existing(VECTORIZER_FILES)

    if not model_path or not vectorizer_path:
        return None, None, (
            "Model/vectorizer files not found. Check your models folder."
        )

    try:
        # Only load model files you trust. Pickle/joblib files can execute code.
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None
    except Exception as exc:
        return None, None, f"Could not load model assets: {exc}"


@st.cache_data
def load_metrics():
    path = first_existing(METRIC_FILES)
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@st.cache_resource
def get_ocr_reader():
    """Create the EasyOCR reader once per Streamlit process."""
    if not OCR_AVAILABLE:
        return None
    return easyocr.Reader(["en"], gpu=False, verbose=False)


# -------------------- DETECTION INDICATORS --------------------

INDICATORS = {
    "Urgency or threats": [
        "urgent", "immediately", "act now", "suspended",
        "suspend", "final warning", "limited time",
        "within 24 hours", "action required",
    ],
    "Credentials or account access": [
        "password", "login", "log in", "sign in",
        "verify your account", "username", "otp",
        "one-time password", "account verification",
    ],
    "Financial language": [
        "bank account", "credit card", "debit card",
        "payment", "refund", "invoice", "upi",
        "transaction", "billing",
    ],
    "Prize or reward": [
        "you won", "winner", "prize", "claim reward",
        "gift card", "lottery", "free gift",
    ],
    "Possible impersonation": [
        "customer support", "security team", "your bank",
        "government", "tax department", "delivery failed",
        "official notice",
    ],
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "cutt.ly",
    "rb.gy", "shorturl.at",
}


def extract_urls(text):
    pattern = r"(?i)\b(?:https?://|www\.)[^\s<>'\"()\]]+"
    urls = re.findall(pattern, str(text))
    return [u.rstrip(".,!?;:}") for u in urls]


def detect_indicators(text):
    text = str(text)
    lowered = text.lower()
    found = []

    for category, phrases in INDICATORS.items():
        matches = [p for p in phrases if p in lowered]
        if matches:
            found.append(f"{category}: {', '.join(matches[:4])}")

    urls = extract_urls(text)
    if urls:
        found.append(f"Contains {len(urls)} URL(s)")

    for raw_url in urls[:5]:
        candidate = raw_url
        if candidate.lower().startswith("www."):
            candidate = "https://" + candidate

        try:
            parsed = urlparse(candidate)
            host = (parsed.hostname or "").lower()

            if host in SHORTENERS:
                found.append(f"URL shortener: {host}")

            try:
                ipaddress.ip_address(host)
                found.append(f"URL uses a raw IP address: {host}")
            except ValueError:
                pass

            if host.startswith("xn--") or ".xn--" in host:
                found.append("URL contains a punycode domain")

            if "@" in parsed.netloc:
                found.append("URL contains @ in its authority section")
        except (ValueError, TypeError):
            pass

    if (
        re.search(r"(?i)\b(password|otp|pin|cvv)\b", text)
        and re.search(
            r"(?i)\b(send|share|reply|provide|enter|confirm|submit)\b",
            text,
        )
    ):
        found.append("Possible request for sensitive credentials")

    return list(dict.fromkeys(found))


def classify_threat(text):
    t = str(text).lower()

    if any(x in t for x in [
        "password", "otp", "login", "log in",
        "sign in", "verify your account",
    ]):
        return "Credential harvesting"

    if any(x in t for x in [
        "bank", "credit card", "payment", "refund",
        "upi", "invoice", "transaction",
    ]):
        return "Financial deception"

    if any(x in t for x in [
        "winner", "prize", "lottery", "claim reward",
        "gift card",
    ]):
        return "Prize / reward lure"

    if extract_urls(t):
        return "Suspicious-link lure"

    if any(x in t for x in [
        "support team", "security team", "government",
        "tax department", "your bank",
    ]):
        return "Possible impersonation"

    return "Potential phishing / social engineering"


def risk_level(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MODERATE"
    return "LOW"


def recommendation(score):
    if score >= 75:
        return (
            "Do not interact",
            "Avoid clicking links, replying, or sharing information. "
            "Verify through an independently obtained official contact.",
        )
    if score >= 50:
        return (
            "Verify independently",
            "Do not follow the message's instructions until the sender "
            "and request have been checked through an official channel.",
        )
    if score >= 25:
        return (
            "Use caution",
            "Check the sender and destination carefully. Never share "
            "passwords, OTPs, PINs, or payment details through message links.",
        )
    return (
        "Stay alert",
        "No strong risk pattern was detected. This is not proof that "
        "the message or link is safe.",
    )


# -------------------- MODEL LABELS / PREDICTIONS --------------------

def is_phishing_label(label):
    """Recognize common label formats; verify this against training labels."""
    if isinstance(label, (int, np.integer)):
        return int(label) == 1
    if isinstance(label, (float, np.floating)) and float(label).is_integer():
        return int(label) == 1

    value = str(label).strip().lower()
    phishing_names = {
        "phishing", "phish", "spam", "malicious", "fraud", "1", "true"
    }
    legitimate_names = {
        "legitimate", "ham", "safe", "benign", "not phishing",
        "not_phishing", "0", "false"
    }
    if value in phishing_names:
        return True
    if value in legitimate_names:
        return False
    return None


def get_phishing_class_index(model):
    classes = list(getattr(model, "classes_", []))
    if not classes:
        return None, None

    matches = [
        i for i, label in enumerate(classes)
        if is_phishing_label(label) is True
    ]
    if len(matches) == 1:
        return matches[0], classes[matches[0]]

    # Common binary convention, only when class 1 is explicitly present.
    for i, label in enumerate(classes):
        if str(label).strip() == "1":
            return i, label

    return None, None


def predict_phishing(model, features):
    """
    Return prediction and risk estimate separately.
    Probability values are meaningful only if the model's class mapping
    and calibration are appropriate for the training task.
    """
    raw_prediction = model.predict(features)[0]
    prediction_is_phishing = is_phishing_label(raw_prediction)
    classes = list(getattr(model, "classes_", []))
    phishing_index, phishing_label = get_phishing_class_index(model)

    if prediction_is_phishing is None:
        raise ValueError(
            f"Unrecognized model label {raw_prediction!r}. "
            "Check how phishing labels were encoded during training."
        )

    probability = None
    note = "Model probability unavailable; risk score is a fallback."

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(features))
        if probs.ndim == 2 and probs.shape[0] == 1 and phishing_index is not None:
            probability = float(probs[0, phishing_index])
            note = (
                "Model-estimated phishing probability; not a safety guarantee."
            )
        elif probs.ndim == 2 and probs.shape[0] == 1 and len(classes) == 2:
            # If labels cannot be mapped, do not guess which class is phishing.
            note = "Probability available, but phishing class mapping is unknown."

    if probability is None and hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(features)).reshape(-1)
        if len(decision) == 1 and len(classes) == 2 and phishing_index is not None:
            # sklearn binary decision_function is positive toward classes_[1].
            oriented_score = float(decision[0])
            if phishing_index == 0:
                oriented_score *= -1
            oriented_score = max(-30.0, min(30.0, oriented_score))
            probability = 1 / (1 + math.exp(-oriented_score))
            note = (
                "Approximate score derived from a decision function; "
                "not a calibrated probability."
            )

    if probability is None:
        probability = 1.0 if prediction_is_phishing else 0.0
        note = (
            "No usable probability was available. Risk is a 0/100 fallback "
            "based on the model's predicted label, not calibrated confidence."
        )

    probability = max(0.0, min(1.0, probability))
    return prediction_is_phishing, probability, note, raw_prediction


def analyse_message(text, model, vectorizer):
    text = str(text or "").strip()
    if len(text) < 10:
        raise ValueError("Enter at least 10 characters of message text.")

    features = vectorizer.transform([text])
    prediction_is_phishing, probability, probability_note, raw_prediction = (
        predict_phishing(model, features)
    )

    score = round(probability * 100, 2)
    action, explanation = recommendation(score)

    return {
        "prediction": 1 if prediction_is_phishing else 0,
        "raw_prediction": str(raw_prediction),
        "risk": score,
        "trust": round(100 - score, 2),
        "level": risk_level(score),
        "threat": classify_threat(text),
        "indicators": detect_indicators(text),
        "action": action,
        "explanation": explanation,
        "probability_note": probability_note,
    }


def analyse_batch(df, column, model, vectorizer):
    result = df.copy()
    texts = result[column].fillna("").astype(str).str.strip()

    result["phishguard_prediction"] = "NOT ANALYSED"
    result["phishing_risk_percent"] = pd.NA
    result["complementary_trust_percent"] = pd.NA
    result["risk_level"] = "NOT ANALYSED"
    result["primary_threat_type"] = "Not analysed"
    result["risk_indicators"] = ""
    result["recommended_action"] = "Provide at least 10 characters."

    valid = texts.str.len() >= 10
    if not valid.any():
        return result

    # Process rows in small groups to avoid excessive memory use on larger CSVs.
    for start in range(0, int(valid.sum()), 250):
        valid_indices = result.index[valid].tolist()[start:start + 250]
        valid_texts = texts.loc[valid_indices].tolist()

        for idx, text in zip(valid_indices, valid_texts):
            single_features = vectorizer.transform([text])
            phishing, probability, _note, _raw = predict_phishing(
                model, single_features
            )
            score = round(probability * 100, 2)
            action, _ = recommendation(score)

            result.at[idx, "phishguard_prediction"] = (
                "PHISHING" if phishing else "NO STRONG PHISHING PATTERN"
            )
            result.at[idx, "phishing_risk_percent"] = score
            result.at[idx, "complementary_trust_percent"] = round(100 - score, 2)
            result.at[idx, "risk_level"] = risk_level(score)
            result.at[idx, "primary_threat_type"] = classify_threat(text)
            result.at[idx, "risk_indicators"] = " | ".join(detect_indicators(text))
            result.at[idx, "recommended_action"] = action

    return result


# -------------------- URL SCANNER: NO NETWORK ACCESS --------------------

def scan_url(raw_url):
    raw_url = (raw_url or "").strip()

    if not raw_url:
        raise ValueError("Enter a URL to scan.")

    if any(ch.isspace() for ch in raw_url):
        raise ValueError("URL must not contain spaces.")

    candidate = raw_url

    if not re.match(r"(?i)^https?://", candidate):
        candidate = "https://" + candidate

    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
        _ = parsed.port
    except Exception:
        raise ValueError("Malformed URL.")

    if not host:
        raise ValueError("Unable to detect hostname.")

    score = 0
    checks = []

    # --------------------------------------------------
    # TLD CHECKS
    # --------------------------------------------------

    suspicious_tlds = {
        ".xyz",
        ".top",
        ".click",
        ".biz",
        ".ru",
        ".info",
        ".gq",
        ".cf",
        ".ml",
        ".tk",
        ".work",
        ".loan",
    }

    if any(host.endswith(tld) for tld in suspicious_tlds):
        score += 20
        checks.append("Uses a commonly abused phishing domain suffix.")

    # --------------------------------------------------
    # HTTPS
    # --------------------------------------------------

    if parsed.scheme.lower() != "https":
        score += 15
        checks.append("Uses HTTP instead of HTTPS.")

    # --------------------------------------------------
    # SHORTENER
    # --------------------------------------------------

    if host in SHORTENERS:
        score += 25
        checks.append("Uses URL shortening service.")

    # --------------------------------------------------
    # IP ADDRESS
    # --------------------------------------------------

    try:
        ipaddress.ip_address(host)
        score += 30
        checks.append("Uses a raw IP address.")
    except ValueError:
        pass

    # --------------------------------------------------
    # @ SIGN
    # --------------------------------------------------

    if "@" in parsed.netloc:
        score += 25
        checks.append("@ used in URL authority.")

    # --------------------------------------------------
    # PUNYCODE
    # --------------------------------------------------

    if host.startswith("xn--") or ".xn--" in host:
        score += 20
        checks.append("Contains punycode domain.")

    # --------------------------------------------------
    # HOST LENGTH
    # --------------------------------------------------

    if len(host) > 45:
        score += 15
        checks.append("Hostname is unusually long.")

    # --------------------------------------------------
    # URL LENGTH
    # --------------------------------------------------

    if len(candidate) > 100:
        score += 10
        checks.append("URL is unusually long.")

    # --------------------------------------------------
    # HYPHENS
    # --------------------------------------------------

    hyphen_count = host.count("-")

    if hyphen_count >= 3:
        score += 15
        checks.append("Contains many hyphens.")

    # --------------------------------------------------
    # LONG QUERY
    # --------------------------------------------------

    if len(parsed.query) > 100:
        score += 10
        checks.append("Long query string detected.")

    # --------------------------------------------------
    # SUSPICIOUS KEYWORDS
    # --------------------------------------------------

    text = (host + parsed.path + parsed.query).lower()

    phishing_keywords = [
        "login",
        "signin",
        "verify",
        "verification",
        "secure",
        "update",
        "account",
        "password",
        "wallet",
        "payment",
        "bank",
        "otp",
        "reward",
        "gift",
        "claim",
        "activate",
        "unlock",
        "recover",
        "suspended",
        "confirm",
        "paypal",
        "microsoft",
        "google",
        "amazon",
    ]

    matched = [
        word
        for word in phishing_keywords
        if word in text
    ]

    if matched:
        score += min(len(matched) * 5, 30)

        checks.append(
            "Contains suspicious keywords: "
            + ", ".join(sorted(set(matched)))
        )

    # --------------------------------------------------
    # SUBDOMAIN TRICK
    # --------------------------------------------------

    brands = [
        "google",
        "microsoft",
        "paypal",
        "amazon",
        "facebook",
        "apple",
        "netflix",
        "instagram",
    ]

    parts = host.split(".")

    if len(parts) >= 3:
        for brand in brands:
            if brand in host and not host.endswith(f"{brand}.com"):
                score += 20
                checks.append(
                    f"Brand name '{brand}' appears in a non-official domain."
                )
                break

    # --------------------------------------------------
    # TOO MANY SUBDOMAINS
    # --------------------------------------------------

    if host.count(".") >= 4:
        score += 10
        checks.append("Excessive subdomains detected.")

    # --------------------------------------------------
    # ENCODED URL
    # --------------------------------------------------

    if "%" in candidate:
        score += 10
        checks.append("Encoded characters found in URL.")

    # --------------------------------------------------
    # NO DOT HOST
    # --------------------------------------------------

    if "." not in host and host != "localhost":
        score += 25
        checks.append("Hostname looks unusual.")

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    score = min(score, 100)

    if score >= 75:
        level = "HIGH"
    elif score >= 35:
        level = "MODERATE"
    else:
        level = "LOW"

    if not checks:
        checks.append(
            "No basic phishing indicators detected. This does not guarantee safety."
        )

    action, explanation = recommendation(score)

    return {
        "url": raw_url,
        "host": host,
        "risk": score,
        "level": level,
        "checks": checks,
        "action": action,
        "explanation": explanation,
    }

# -------------------- HEADER / SIDEBAR --------------------

st.markdown("""
<div class="hero">
  <h1>🛡️ PHISHGUARD</h1>
  <p>Phishing and scam risk assessment · Local machine learning · Explainable checks</p>
</div>
""", unsafe_allow_html=True)

model, vectorizer, model_error = load_assets()
metrics = load_metrics()

with st.sidebar:
    st.header("System status")
    if model is not None and vectorizer is not None:
        st.success("Model loaded")
        st.caption(f"Model file: {first_existing(MODEL_FILES).name}")
        st.caption(f"Vectorizer file: {first_existing(VECTORIZER_FILES).name}")
    else:
        st.error("Model unavailable")
        st.caption(model_error)

    st.markdown("---")
    st.caption("Model inference runs locally. Messages are not sent to an external AI API.")
    st.caption("Risk estimates are not proof that a message or link is safe.")

    if st.button("Reload model files"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

if model is None or vectorizer is None:
    st.warning(
        "Analysis is disabled until a compatible model and its matching "
        "TF-IDF vectorizer are available in the models folder."
    )

tabs = st.tabs([
    "🔎 Single Message",
    "📁 Batch CSV",
    "🔗 URL Scanner",
    "🖼️ Screenshot OCR",
    "📊 Model Analytics",
])


# -------------------- TAB 1: SINGLE MESSAGE --------------------

with tabs[0]:
    st.subheader("Single-message analysis")
    message = st.text_area(
        "Paste an email, SMS, chat message, or notification",
        height=180,
        placeholder="Paste the message here...",
    )

    if st.button(
        "Analyse message",
        type="primary",
        disabled=(model is None or vectorizer is None),
        key="analyse_message_button",
    ):
        try:
            result = analyse_message(message, model, vectorizer)

            c1, c2, c3 = st.columns(3)
            c1.metric("Estimated phishing risk", f"{result['risk']:.1f}%")
            c2.metric("Complementary score", f"{result['trust']:.1f}%")
            c3.metric("Risk level", result["level"])
            st.progress(int(result["risk"]))

            if result["prediction"] == 1:
                st.error("Model prediction: PHISHING")
            else:
                st.success("Model prediction: NO STRONG PHISHING PATTERN")

            st.markdown(f"**Primary threat type:** {result['threat']}")
            st.markdown(f"**Recommended action:** {result['action']}")
            st.write(result["explanation"])
            st.caption(result["probability_note"])

            with st.expander("Explainable indicators", expanded=True):
                if result["indicators"]:
                    for item in result["indicators"]:
                        st.markdown(f"- {item}")
                else:
                    st.write("No listed keyword or URL-pattern indicators found.")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption("Check that the saved model and vectorizer match.")

def detect_column(df):
    """Automatically suggest the most likely message-text column."""
    preferred_names = [
        "text",
        "message",
        "email",
        "email_text",
        "text_combined",
        "content",
        "body",
        "sms",
    ]

    columns_by_lowercase = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in preferred_names:
        if name in columns_by_lowercase:
            return columns_by_lowercase[name]

    # If no familiar column name exists, use the first column.
    return df.columns[0] if len(df.columns) else None


def uploaded_signature(raw_csv, column):
    """Identify the uploaded CSV and selected text column."""
    digest = hashlib.sha256()
    digest.update(raw_csv)
    digest.update(str(column).encode("utf-8"))
    return digest.hexdigest()


def safe_csv_for_spreadsheet(df):
    """Neutralize formula-like text before exporting a CSV."""
    safe_df = df.copy()

    def neutralize(value):
        if not isinstance(value, str):
            return value

        stripped = value.lstrip(" \t\r\n")

        if stripped.startswith(("=", "+", "-", "@")):
            return "'" + value

        if value.startswith(("\t", "\r")):
            return "'" + value

        return value

    for column in safe_df.columns:
        safe_df[column] = safe_df[column].map(neutralize)

    return safe_df
# -------------------- TAB 2: BATCH CSV --------------------

with tabs[1]:
    st.subheader("Batch CSV analysis")
    st.write(
        f"Upload a CSV file. Up to {MAX_ANALYSIS_ROWS:,} rows are analysed per run. "
        f"Maximum upload size: {MAX_UPLOAD_MB} MB."
    )

    uploaded = st.file_uploader(
        "Choose CSV file",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:
        raw_csv = uploaded.getvalue()
        if len(raw_csv) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"CSV exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        else:
            try:
                # Read at most one extra row so we can detect truncation without
                # loading an arbitrarily large CSV into memory.
                try:
                    from io import BytesIO
                    df = pd.read_csv(
                        BytesIO(raw_csv), encoding="utf-8-sig",
                        nrows=MAX_ANALYSIS_ROWS + 1,
                    )
                except UnicodeDecodeError:
                    df = pd.read_csv(
                        BytesIO(raw_csv), encoding="latin-1",
                        nrows=MAX_ANALYSIS_ROWS + 1,
                    )

                if df.empty:
                    st.warning("This CSV has no data rows.")
                else:
                    has_more_rows = len(df) > MAX_ANALYSIS_ROWS
                    if has_more_rows:
                        st.warning(
                            f"CSV has more than {MAX_ANALYSIS_ROWS:,} rows. "
                            f"Only the first {MAX_ANALYSIS_ROWS:,} rows will be analysed."
                        )
                        df = df.head(MAX_ANALYSIS_ROWS).copy()

                    st.caption(
                        f"{len(df):,} rows loaded for analysis · "
                        f"{len(df.columns):,} columns"
                    )

                    suggested = detect_column(df)
                    columns = list(df.columns)
                    default_index = (
                        columns.index(suggested)
                        if suggested in columns else 0
                    )
                    column = st.selectbox(
                        "Choose the message-text column",
                        columns,
                        index=default_index,
                        key="batch_text_column",
                    )

                    signature = uploaded_signature(raw_csv, column)
                    stored_signature = st.session_state.get("batch_signature")
                    if stored_signature != signature:
                        st.session_state.pop("batch_results", None)
                        st.session_state["batch_signature"] = signature

                    if st.button(
                        "Run batch analysis",
                        type="primary",
                        disabled=(model is None or vectorizer is None),
                        key="run_batch_button",
                    ):
                        try:
                            with st.spinner("Analysing CSV locally..."):
                                st.session_state["batch_results"] = analyse_batch(
                                    df, column, model, vectorizer
                                )
                                st.session_state["batch_signature"] = signature
                        except Exception as exc:
                            st.error(f"Batch analysis failed: {exc}")

                    batch = st.session_state.get("batch_results")
                    if (
                        isinstance(batch, pd.DataFrame)
                        and st.session_state.get("batch_signature") == signature
                    ):
                        total = len(batch)
                        analysed = int(
                            (batch["risk_level"] != "NOT ANALYSED").sum()
                        )
                        phishing = int(
                            (batch["phishguard_prediction"] == "PHISHING").sum()
                        )

                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Rows in report", total)
                        c2.metric("Analysed", analysed)
                        c3.metric("Phishing predictions", phishing)
                        c4.metric("Not analysed", total - analysed)

                        st.markdown("### Risk distribution")
                        st.bar_chart(batch["risk_level"].value_counts())

                        options = [
                            "ALL", "CRITICAL", "HIGH",
                            "MODERATE", "LOW", "NOT ANALYSED",
                        ]
                        selected = st.selectbox(
                            "Filter by risk level", options,
                            key="batch_risk_filter",
                        )
                        filtered = (
                            batch if selected == "ALL"
                            else batch[batch["risk_level"] == selected]
                        )
                        st.dataframe(
                            filtered,
                            use_container_width=True,
                            hide_index=True,
                        )

                        export_df = safe_csv_for_spreadsheet(batch)
                        csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "Download full results CSV",
                            data=csv_data,
                            file_name="phishguard_batch_results.csv",
                            mime="text/csv",
                            type="primary",
                            key="download_batch_results",
                        )
                        st.caption(
                            "The downloaded report contains all analysed rows, "
                            "not just the visible filter. Formula-like text fields "
                            "are neutralized for safer spreadsheet opening."
                        )
            except Exception as exc:
                st.error(f"Could not process CSV: {exc}")


# -------------------- TAB 3: URL SCANNER --------------------

with tabs[2]:
    st.subheader("URL risk scanner")
    st.warning(
        "Local pattern checks only. This scanner does not open or visit URLs "
        "and does not check live reputation."
    )
    url = st.text_input(
        "URL to inspect", placeholder="https://example.com/login",
        key="url_input",
    )

    if st.button("Scan URL", type="primary", key="scan_url_button"):
        try:
            result = scan_url(url)
            c1, c2 = st.columns(2)
            c1.metric("Heuristic risk score", f"{result['risk']}/100")
            c2.metric("Risk level", result["level"])
            st.markdown(f"**Parsed host:** `{result['host']}`")
            st.markdown(f"**Recommended action:** {result['action']}")
            st.write(result["explanation"])
            st.markdown("**Checks triggered**")
            for item in result["checks"]:
                st.markdown(f"- {item}")
            st.caption(
                "A low score is not proof of safety. HTTPS does not mean a "
                "site is trustworthy."
            )
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"URL scan failed: {exc}")


# -------------------- TAB 4: SCREENSHOT OCR (EASYOCR) --------------------

with tabs[3]:
    st.subheader("Screenshot OCR scanner")
    st.write(
        "Upload a screenshot of an email, SMS, or notification. EasyOCR "
        "extracts the text so you can review it and run the phishing model."
    )
    st.caption(
        "Requires EasyOCR and Pillow. The first OCR run may download language "
        "model files and can take longer."
    )

    screenshot = st.file_uploader(
        "Upload screenshot",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="screenshot_upload",
    )

    if screenshot is not None:
        if not OCR_AVAILABLE:
            st.error(
                "EasyOCR is not installed. Run this in your project terminal: "
                "`pip install easyocr pillow`"
            )
        else:
            try:
                image = Image.open(screenshot).convert("RGB")
                st.image(
                    image,
                    caption="Uploaded screenshot",
                    use_container_width=True,
                )

                if st.button(
                    "Extract screenshot text",
                    type="primary",
                    key="extract_ocr_button",
                ):
                    try:
                        with st.spinner(
                            "Loading EasyOCR and extracting text... "
                            "The first run may take longer."
                        ):
                            reader = get_ocr_reader()
                            results = reader.readtext(
                                np.array(image), detail=0, paragraph=True
                            )
                            extracted_text = "\n".join(
                                str(item).strip()
                                for item in results
                                if str(item).strip()
                            ).strip()
                            st.session_state["ocr_text"] = extracted_text
                            st.session_state["ocr_edit"] = extracted_text
                            st.session_state["ocr_image_signature"] = hashlib.sha256(
                                screenshot.getvalue()
                            ).hexdigest()
                    except Exception as exc:
                        st.error(f"OCR failed: {exc}")
                        st.caption(
                            "Check your internet connection for the first model "
                            "download, and ensure enough disk space is available."
                        )

                if "ocr_text" in st.session_state:
                    st.text_area(
                        "Extracted text (editable)",
                        key="ocr_edit",
                        height=180,
                        help="Review OCR mistakes and edit the text before analysis.",
                    )

                    if st.button(
                        "Analyse extracted text",
                        type="primary",
                        disabled=(model is None or vectorizer is None),
                        key="analyse_ocr_button",
                    ):
                        try:
                            extracted = st.session_state.get("ocr_edit", "")
                            result = analyse_message(
                                extracted, model, vectorizer
                            )
                            c1, c2, c3 = st.columns(3)
                            c1.metric(
                                "Estimated phishing risk",
                                f"{result['risk']:.1f}%",
                            )
                            c2.metric("Risk level", result["level"])
                            c3.metric(
                                "Prediction",
                                "PHISHING" if result["prediction"] == 1
                                else "NO STRONG PATTERN",
                            )
                            st.markdown(f"**Threat type:** {result['threat']}")
                            st.markdown(
                                f"**Recommended action:** {result['action']}"
                            )
                            st.write(result["explanation"])
                            st.caption(result["probability_note"])
                            with st.expander("Explainable indicators", expanded=True):
                                if result["indicators"]:
                                    for item in result["indicators"]:
                                        st.markdown(f"- {item}")
                                else:
                                    st.write(
                                        "No listed keyword or URL-pattern indicators found."
                                    )
                        except ValueError as exc:
                            st.warning(str(exc))
                        except Exception as exc:
                            st.error(f"Could not analyse extracted text: {exc}")
            except Exception as exc:
                st.error(f"Could not process image: {exc}")


# -------------------- TAB 5: MODEL ANALYTICS --------------------

with tabs[4]:
    st.subheader("Model evaluation analytics")
    st.caption(
        "Evaluation results saved during model training. "
        "Performance on new, unseen data may differ."
    )

    if metrics:
        # -------------------- METRIC CARDS --------------------

        st.markdown("### Evaluation summary")

        metric_definitions = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1 Score": "f1_score",
        }

        values = {}

        for label, key in metric_definitions.items():
            value = metrics.get(key)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0 <= float(value) <= 1
            ):
                values[label] = float(value)

        cols = st.columns(4)

        for i, label in enumerate(metric_definitions):
            with cols[i]:
                if label in values:
                    st.metric(
                        label,
                        f"{values[label] * 100:.2f}%"
                    )
                else:
                    st.metric(label, "N/A")

        # -------------------- PERFORMANCE CHART --------------------

        if values:
            st.markdown("### Performance comparison")

            chart_data = pd.DataFrame({
                "Metric": list(values.keys()),
                "Score": list(values.values()),
            })

            chart_spec = {
                "mark": {
                    "type": "bar",
                    "cornerRadiusEnd": 5,
                    "tooltip": True,
                },
                "encoding": {
                    "x": {
                        "field": "Metric",
                        "type": "nominal",
                        "sort": None,
                        "axis": {
                            "title": None,
                            "labelAngle": 0,
                        },
                    },
                    "y": {
                        "field": "Score",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, 1],
                            "nice": False,
                        },
                        "axis": {
                            "title": "Score",
                            "format": ".0%",
                            "values": [
                                0, 0.2, 0.4, 0.6, 0.8, 1
                            ],
                        },
                    },
                },
                "width": "container",
                "height": 300,
            }

            st.vega_lite_chart(
                chart_data,
                spec=chart_spec,
                use_container_width=True,
            )
        else:
            st.warning(
                "No valid metric scores found. Expected numeric "
                "values between 0 and 1."
            )

        # -------------------- CONFUSION MATRIX --------------------

        st.markdown("### Confusion matrix")

        confusion = metrics.get("confusion_matrix")

        valid_confusion = (
            isinstance(confusion, (list, tuple))
            and len(confusion) == 2
            and all(
                isinstance(row, (list, tuple))
                and len(row) == 2
                and all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(float(x))
                    and float(x).is_integer()
                    and x >= 0
                    for x in row
                )
                for row in confusion
            )
        )

        if valid_confusion:
            # Training labels: 0 = legitimate, 1 = phishing.
            tn = int(confusion[0][0])
            fp = int(confusion[0][1])
            fn = int(confusion[1][0])
            tp = int(confusion[1][1])

            matrix_df = pd.DataFrame(
                [
                    [tn, fp],
                    [fn, tp],
                ],
                index=[
                    "Actual: Legitimate (0)",
                    "Actual: Phishing (1)",
                ],
                columns=[
                    "Predicted: Legitimate (0)",
                    "Predicted: Phishing (1)",
                ],
            )

            st.caption(
                "Rows are actual classes; columns are predicted classes. "
                "Class 0 = legitimate, class 1 = phishing."
            )

            st.dataframe(
                matrix_df,
                use_container_width=True,
            )

            st.markdown("#### Classification breakdown")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Correct legitimate detections (TN)",
                    f"{tn:,}",
                )
                st.metric(
                    "Missed phishing messages (FN)",
                    f"{fn:,}",
                )

            with col2:
                st.metric(
                    "False phishing alarms (FP)",
                    f"{fp:,}",
                )
                st.metric(
                    "Correct phishing detections (TP)",
                    f"{tp:,}",
                )

        else:
            st.warning(
                "A valid 2×2 confusion matrix was not found in the "
                "loaded metrics. Check models/calibrated_metrics.json."
            )

        # -------------------- SAVED METADATA --------------------

        st.markdown("### Saved metadata")

        metadata_rows = [
            ("Dataset", metrics.get("dataset", "N/A")),
            (
                "Total cleaned rows",
                metrics.get("total_rows_after_cleaning", "N/A"),
            ),
            ("Training rows", metrics.get("train_rows", "N/A")),
            ("Testing rows", metrics.get("test_rows", "N/A")),
        ]

        # Show only validated numeric scores.
        for label, key in metric_definitions.items():
            if label in values:
                metadata_rows.append(
                    (label, f"{values[label] * 100:.2f}%")
                )
            else:
                metadata_rows.append((label, "N/A"))

        note = metrics.get("note")
        if isinstance(note, str) and note.strip():
            metadata_rows.append(("Evaluation note", note.strip()))

        metadata_df = pd.DataFrame(
            metadata_rows,
            columns=["Property", "Value"],
        )

        # Table display avoids raw JSON formatting.
        st.dataframe(
            metadata_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No saved metrics JSON found. Check that "
            "models/calibrated_metrics.json exists."
        )

    # -------------------- METRIC DEFINITIONS --------------------

    st.markdown("### Metric definitions")

    st.markdown("""
    - **Accuracy:** proportion of all examples classified correctly.
    - **Precision:** proportion of predicted phishing examples that
      were actually phishing.
    - **Recall:** proportion of actual phishing examples detected.
    - **F1 Score:** harmonic mean of precision and recall.
    - **TN:** legitimate messages correctly identified.
    - **FP:** legitimate messages incorrectly flagged as phishing.
    - **FN:** phishing messages incorrectly classified as legitimate.
    - **TP:** phishing messages correctly identified.
    """)

    st.caption(
        "These scores are from the saved held-out test evaluation. "
        "They do not guarantee real-world performance. External "
        "validation on separate data is recommended."
    )
# -------------------- FOOTER --------------------

st.markdown("---")
st.caption(
    "PhishGuard is an educational risk-assessment prototype. Do not use it "
    "as the sole basis for security decisions. A low score is not a safety guarantee."
)
