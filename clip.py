import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm

st.set_page_config(layout="wide")
st.title("🧠 Product Title Clustering + 🖼 Image Classification with CLIP")

# Load models
@st.cache_resource
def load_models():
    st.text("Loading models...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return sentence_model, clip_model, clip_processor

sentence_model, clip_model, clip_processor = load_models()

# File upload
uploaded_csv = st.file_uploader("Upload CSV with 'title' and 'image_path' columns", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    
    if 'title' not in df.columns or 'image_path' not in df.columns:
        st.error("CSV must have 'title' and 'image_path' columns")
    else:
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=5)
        if st.button("Cluster Titles"):
            with st.spinner("Clustering..."):
                embeddings = sentence_model.encode(df['title'].tolist(), show_progress_bar=True)
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(embeddings)
                df['cluster'] = kmeans.labels_
                st.success("✅ Clustering complete")

            st.write(df[['title', 'cluster']])

            # Upload rules
            rules_file = st.file_uploader("Upload Rule Mapping CSV (cluster,label)", type=["csv"], key="rules_upload")
            if rules_file:
                rules_df = pd.read_csv(rules_file)
                mapping = dict(zip(rules_df['cluster'], rules_df['label']))
                df['label'] = df['cluster'].map(mapping)
                st.success("✅ Rule mapping applied")

                # Image classification
                def classify_image(image_path, labels):
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as e:
                        return "Error"

                    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)
                    label_idx = probs.argmax().item()
                    return labels[label_idx]

                st.info("Classifying images...")
                df['predicted_label'] = [
                    classify_image(row['image_path'], df[df['cluster'] == row['cluster']]['label'].unique().tolist())
                    for _, row in tqdm(df.iterrows(), total=len(df))
                ]
                st.success("✅ Image classification complete")

                st.dataframe(df[['title', 'label', 'predicted_label']])
                st.download_button("📥 Download Results", df.to_csv(index=False), "classified_products.csv", "text/csv")

