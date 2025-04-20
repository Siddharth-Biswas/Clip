import streamlit as st
import pandas as pd
import numpy as np
import torch
import requests
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import re

# -------------------- Setup --------------------
st.set_page_config(page_title="Smart Product Classifier", layout="wide")
st.title("🧠 Product Classifier: NLP + CLIP")
st.markdown("Upload your product list — we'll discover labels from titles and classify images accordingly.")

devices = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
@st.cache_resource
def load_models():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(devices)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return text_model, clip_model, clip_processor

text_model, clip_model, clip_processor = load_models()

# -------------------- Helpers --------------------
def preprocess_title(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

def extract_keywords(texts, n_terms=3):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(texts)
    top_keywords = []
    terms = np.array(tfidf.get_feature_names_out())
    for row in tfidf_matrix.toarray():
        top = row.argsort()[-n_terms:][::-1]
        top_keywords.append(", ".join(terms[top]))
    return top_keywords

def cluster_titles(titles, n_clusters=15):
    embeddings = text_model.encode(titles, show_progress_bar=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

def name_clusters(titles, labels, n_keywords=3):
    grouped = pd.DataFrame({'title': titles, 'cluster': labels})
    cluster_names = {}
    for label in grouped['cluster'].unique():
        subset = grouped[grouped['cluster'] == label]['title']
        keywords = extract_keywords(subset.tolist(), n_keywords)
        cluster_names[label] = pd.Series(keywords).mode()[0]  # most common keyword string
    return cluster_names

def classify_with_clip(image_url, label_texts):
    try:
        image = Image.open(BytesIO(requests.get(image_url, timeout=5).content)).convert("RGB")
        inputs = clip_processor(text=label_texts, images=image, return_tensors="pt", padding=True).to(devices)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # (1, labels)
        probs = logits_per_image.softmax(dim=1).cpu().numpy().flatten()
        return label_texts[np.argmax(probs)]
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------- Streamlit UI --------------------
uploaded_file = st.file_uploader("📦 Upload your product file (CSV/Excel)", type=["csv", "xlsx"])
num_clusters = st.slider("🔢 Number of clusters to discover", 5, 50, 15)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('xlsx') else pd.read_csv(uploaded_file)
        if 'TITLE' not in df.columns or 'IMAGE_URL' not in df.columns:
            st.error("Your file must include 'TITLE' and 'IMAGE_URL' columns.")
            st.stop()

        df['clean_title'] = df['TITLE'].apply(preprocess_title)

        st.subheader("📊 Clustering titles into labels")
        labels, centers = cluster_titles(df['clean_title'].tolist(), n_clusters=num_clusters)
        label_names = name_clusters(df['clean_title'].tolist(), labels)

        df['cluster_id'] = labels
        df['generated_label'] = df['cluster_id'].map(label_names)

        st.success("✅ Titles clustered and labeled.")

        st.subheader("🖼️ Classifying images with CLIP")
        label_set = sorted(df['generated_label'].unique().tolist())

        results = []
        progress = st.progress(0)
        for idx, row in enumerate(df.itertuples(index=False)):
            image_url = getattr(row, 'IMAGE_URL')
            predicted_label = classify_with_clip(image_url, label_set) if pd.notna(image_url) else "No Image"
            results.append(predicted_label)
            progress.progress((idx + 1) / len(df))

        df['clip_image_label'] = results
        st.success("🏁 Image classification complete!")

        st.subheader("🔍 Preview")
        st.dataframe(df[['TITLE', 'generated_label', 'clip_image_label']], use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results CSV", csv, "classified_products.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

else:
    st.info("👆 Upload a file to begin.")
