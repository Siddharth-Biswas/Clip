import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
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

    # Pre-tokenize once
    text_tokens = clip.tokenize(label_names).to(device)

    with st.spinner("Classifying images with CLIP..."):
        for _, row in df.iterrows():
            try:
                image = Image.open(row["image_path"]).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image_input)
                    text_features = clip_model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).squeeze(0).cpu().numpy()
                predicted_label = label_names[similarity.argmax()]
            except Exception as e:
                predicted_label = f"Error: {e}"
            results.append(predicted_label)

    return results

uploaded_file = st.file_uploader("Upload product data (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load DataFrame
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

    if "title" not in df.columns or "image_path" not in df.columns:
        st.error("CSV/XLSX must contain 'title' and 'image_path' columns.")
    else:
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=10)

       if "df_clustered" not in st.session_state and st.button("Run Clustering"):
    with st.spinner("Clustering product titles..."):
        labels, _ = cluster_titles(df["title"].tolist(), n_clusters)
        df["cluster"] = labels
        st.session_state.df_clustered = df  # Save to session

# If clustering was done
if "df_clustered" in st.session_state:
    df = st.session_state.df_clustered

    mapping_file = st.file_uploader("Upload rules file to rename clusters (CSV with columns: cluster,label)", type=["csv"])
    
    if mapping_file:
        try:
            rules_df = pd.read_csv(mapping_file)
            cluster_to_label = dict(zip(rules_df["cluster"], rules_df["label"]))
            df["label"] = df["cluster"].map(cluster_to_label).fillna(df["cluster"])
            st.session_state.df_labeled = df
            st.success("Applied label mapping from rules file.")
        except Exception as e:
            st.warning(f"Error processing rules file: {e}")
            df["label"] = df["cluster"]
    else:
        df["label"] = df["cluster"]
        st.info("No rules file uploaded. Using cluster numbers as labels.")

    if st.button("Classify Images"):
        label_names = sorted(df["label"].astype(str).unique().tolist())
        df["image_label"] = classify_images_with_clip(df, clip_model, preprocess, label_names)
        st.success("Image classification complete.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            st.download_button("Download Classified CSV", tmp_file.name, file_name="classified_products.csv")

        st.dataframe(df.head(20))
