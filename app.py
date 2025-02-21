import streamlit as st
import mysql.connector
from transformers import BartTokenizer, BartForConditionalGeneration, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
import pandas as pd
import nltk
from docx import Document
import PyPDF2
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load models (cached)
@st.cache_resource
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_keybert_model():
    return KeyBERT(model="distilbert-base-nli-mean-tokens")

summ_tokenizer, summ_model = load_summarization_model()
sent_tokenizer, sent_model = load_sentiment_model()
keybert_model = load_keybert_model()

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="book_summarizer"
    )

# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith('.docx'):
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        st.error("Unsupported file format. Please upload a .docx or .pdf file.")
        return None

# Function to process input
def process_input(title, input_text, summary_length):
    with st.spinner("Analyzing the text..."):
        # Preprocess input: remove title and metadata if provided
        if title and title in input_text:
            # Remove the title and any following line (assuming metadata follows)
            lines = input_text.split('\n')
            filtered_lines = [line for line in lines if line.strip() and line.strip() != title]
            # Skip first few lines if they look like metadata (e.g., authorship)
            content_start = 0
            for i, line in enumerate(filtered_lines):
                if "by " in line.lower() or "institute" in line.lower() or len(line.split()) < 5:
                    content_start = i + 1
                else:
                    break
            input_for_summary = "\n".join(filtered_lines[content_start:])
        else:
            input_for_summary = input_text

        if not input_for_summary.strip():
            st.error("No content found after removing title/metadata.")
            return

        # Summarization with BART
        inputs = summ_tokenizer(input_for_summary, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = summ_model.generate(
            inputs["input_ids"],
            max_length=summary_length,
            min_length=max(60, summary_length // 3),  # Increase min_length for more detail
            length_penalty=1.0,  # Reduce penalty to encourage longer output
            num_beams=6,  # Increase beams for better quality
            early_stopping=False  # Allow full generation
        )
        summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Key Aspects with KeyBERT (use original input for broader context)
        key_aspects = keybert_model.extract_keywords(input_text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
        key_aspects = [kw[0] for kw in key_aspects]

        # Important Points with TF-IDF (use original input)
        sentences = sent_tokenize(input_text)
        if len(sentences) > 1:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1
            scored_sentences = list(zip(sentences, sentence_scores))
            important_points = [sent for sent, score in sorted(scored_sentences, key=lambda x: x[1], reverse=True)][:5]
        else:
            important_points = sentences[:5]

        # Sentiment Analysis with DistilBERT
        positive, neutral, negative = 0, 0, 0
        for sentence in sentences:
            inputs = sent_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            outputs = sent_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).detach().numpy()[0]
            if probs[1] > 0.6:
                positive += 1
            elif probs[0] > 0.6:
                negative += 1
            else:
                neutral += 1
        total = len(sentences)
        positive_pct = (positive / total) * 100 if total > 0 else 0
        neutral_pct = (neutral / total) * 100 if total > 0 else 0
        negative_pct = (negative / total) * 100 if total > 0 else 0

        # Dashboard
        st.header("Summary")
        st.write(summary)

        st.header("Key Aspects")
        st.write(", ".join(key_aspects))

        st.header("Important Points")
        for point in important_points:
            st.write(f"- {point}")

        st.header("Sentiment Analysis")
        fig = go.Figure(go.Bar(
            x=[positive_pct, neutral_pct, negative_pct],
            y=["Positive", "Neutral", "Negative"],
            orientation="h",
            marker_color=["#00cc96", "#636efa", "#ef553b"]
        ))
        st.plotly_chart(fig)

        # Save to database
        try:
            db = get_db_connection()
            cursor = db.cursor()
            sql = """INSERT INTO books (title, text, summary, key_aspects, important_points, 
                     sentiment_positive, sentiment_neutral, sentiment_negative)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
            values = (
                title or "Untitled",
                input_text,
                summary,
                ", ".join(key_aspects),
                "; ".join(important_points),
                positive_pct,
                neutral_pct,
                negative_pct
            )
            cursor.execute(sql, values)
            db.commit()
            st.success("Data saved to database!")
        except Exception as e:
            st.error(f"Database error: {e}")
        finally:
            cursor.close()
            db.close()

# Streamlit app
st.title("Book and Text Summarization App")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Analyze Text", "View Saved Data"])

if page == "Analyze Text":
    st.write("Paste your text or upload a .docx/.pdf file for analysis.")
    
    # Tabs for input method
    tab1, tab2 = st.tabs(["Paste Text", "Upload File"])
    
    with tab1:
        title = st.text_input("Book Title (optional)", key="text_title")
        input_text = st.text_area("Paste your text here", height=200, key="text_input")
        summary_length = st.slider("Select Summary Length (tokens)", min_value=50, max_value=500, value=150, step=10, key="text_slider")
        analyze_text_button = st.button("Analyze Text", key="text_button")

    with tab2:
        file_title = st.text_input("Book Title (optional)", key="file_title")
        uploaded_file = st.file_uploader("Upload a .docx or .pdf file", type=["docx", "pdf"], key="file_upload")
        file_summary_length = st.slider("Select Summary Length (tokens)", min_value=50, max_value=500, value=150, step=10, key="file_slider")
        analyze_file_button = st.button("Analyze File", key="file_button")

    # Process text input
    if analyze_text_button and input_text:
        process_input(title, input_text, summary_length)
    
    # Process file input
    if analyze_file_button and uploaded_file:
        file_text = extract_text_from_file(uploaded_file)
        if file_text:
            process_input(file_title, file_text, file_summary_length)

elif page == "View Saved Data":
    st.header("Saved Books Data")
    try:
        db = get_db_connection()
        cursor = db.cursor()
        query = """
            SELECT id, title, summary, key_aspects, important_points, 
                   sentiment_positive, sentiment_neutral, sentiment_negative, date_added 
            FROM books
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        if rows:
            columns = ["ID", "Title", "Summary", "Key Aspects", "Important Points", 
                       "Positive (%)", "Neutral (%)", "Negative (%)", "Date Added"]
            df = pd.DataFrame(rows, columns=columns)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No data found in the database.")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
    finally:
        cursor.close()
        db.close()