import streamlit as st
import spacy
import pandas as pd
import joblib
from PyPDF2 import PdfReader
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ---------- Load Models ----------
nlp = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization", model="t5-small")
try:
    clf, vectorizer = joblib.load("models/clause_model.pkl")
except:
    # Quick toy model if not trained
    texts = ["The buyer shall pay within 30 days",
             "The seller is not liable for damages",
             "Both parties agree to confidentiality"]
    labels = ["Payment", "Liability", "Confidentiality"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = RandomForestClassifier().fit(X, labels)
    joblib.dump((clf, vectorizer), "models/clause_model.pkl")

# ---------- Helper Functions ----------
def read_pdf(file):
    reader = PdfReader(file)
    return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])

def extract_ner(text):
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    return pd.DataFrame(ents, columns=["Entity", "Label"])

def summarize_text(text):
    return summarizer(text[:1000], max_length=80, min_length=30, do_sample=False)[0]['summary_text']

def classify_clauses(text):
    clauses = text.split(". ")
    X = vectorizer.transform(clauses)
    preds = clf.predict(X)
    return pd.DataFrame({"Clause": clauses, "Category": preds})

# ---------- Streamlit App ----------
st.title("📑 Legal Document Summarizer & Clause Extractor")

uploaded = st.file_uploader("Upload a PDF legal document", type=["pdf"])
if uploaded:
    text = read_pdf(uploaded)
    st.subheader("Extracted Text")
    st.write(text[:1000] + "...")

    st.subheader("Named Entities")
    st.dataframe(extract_ner(text))

    st.subheader("Summary")
    st.write(summarize_text(text))

    st.subheader("Clause Classification")
    st.dataframe(classify_clauses(text))

