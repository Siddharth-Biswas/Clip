import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import os
import tempfile

st.set_page_config(layout="wide")
st.title("Product Classifier using CLIP and Title Clustering")

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, clip_model, preprocess

sentence_model, clip_model, preprocess = load_models()

def cluster_titles(titles, n_clusters=10):
    embeddings = sentence_model.encode(titles, show_progress_bar=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

def classify_images_with_clip(df, clip_model, preprocess, label_names):
    device = "cpu"
    clip_model.eval()
    results = []

    with st.spinner("Classifying images with CLIP..."):
        for i, row in df.iterrows():
            try:
                image = preprocess(Image.open(row["image_path"]).convert("RGB")).unsqueeze(0).to(device)
                text = clip.tokenize(label_names).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    text_features = clip_model.encode_text(text)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).detach().numpy()
                label = label_names[similarity.argmax()]
            except Exception as e:
                label = f"Error: {e}"
            results.append(label)

    return results

uploaded_file = st.file_uploader("Upload product data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "title" not in df.columns or "image_path" not in df.columns:
        st.error("CSV/XLSX must contain 'title' and 'image_path' columns.")
    else:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=10)
        if st.button("Run Clustering"):
            with st.spinner("Clustering product titles..."):
                labels, kmeans = cluster_titles(df["title"].tolist(), n_clusters)
                df["cluster"] = labels

            # Ask for mapping file
            mapping_file = st.file_uploader("Upload rules file to rename clusters (CSV with columns: cluster,label)", type=["csv"])
            if mapping_file:
                rules_df = pd.read_csv(mapping_file)
                cluster_to_label = dict(zip(rules_df["cluster"], rules_df["label"]))
                df["label"] = df["cluster"].map(cluster_to_label)
                st.success("Applied label mapping from rules file.")
            else:
                df["label"] = df["cluster"]
                st.info("No rules file uploaded. Using cluster numbers as labels.")

            # Now classify with CLIP
            if st.button("Classify Images"):
                label_names = df["label"].unique().tolist()
                df["image_label"] = classify_images_with_clip(df, clip_model, preprocess, label_names)
                st.success("Image classification complete.")

                # Download button
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                df.to_csv(tmp_file.name, index=False)
                st.download_button("Download Classified CSV", tmp_file.name, file_name="classified_products.csv")

                st.dataframe(df.head(20))
