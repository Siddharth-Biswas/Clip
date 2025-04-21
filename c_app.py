import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
import io
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import requests
import tempfile
import numpy as np
import gc
import time
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Product Classifier")


@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, clip_model, preprocess


sentence_model, clip_model, preprocess = load_models()


def cluster_titles(titles, n_clusters):
    embeddings = sentence_model.encode(titles)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def calculate_wcss(titles, max_clusters=20):
    embeddings = sentence_model.encode(titles)
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        wcss.append(kmeans.inertia_)
    return wcss


def load_image(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        st.warning(f"Could not load image from URL '{url}': {e}")
        return None


def classify_images_with_clip(df, clip_model, preprocess, label_names, image_url_col):
    device = "cpu"
    clip_model.eval()
    results = []
    num_images = len(df)
    start_time = time.time()
    progress_bar = st.progress(0.0, f"Classifying images with CLIP... Time: 0s | Estimated Time Remaining: Calculating...")
    text_tokens = clip.tokenize(label_names).to(device)

    for index, row in df.iterrows():
        url = row.get(image_url_col)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            results.append("Error: Invalid or missing URL")
            elapsed_time = int(time.time() - start_time)
            progress = (index + 1) / num_images
            remaining_time = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
            progress_bar.progress(progress, f"Classifying images with CLIP... Time: {elapsed_time}s | Estimated Time Remaining: {int(remaining_time)}s")
            continue

        try:
            image = load_image(url)
            if image is None:
                results.append("Error: Could not load image")
                elapsed_time = int(time.time() - start_time)
                progress = (index + 1) / num_images
                remaining_time = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
                progress_bar.progress(progress, f"Classifying images with CLIP... Time: {elapsed_time}s | Estimated Time Remaining: {int(remaining_time)}s")
                continue

            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (
                100.0 * image_features @ text_features.T
            ).squeeze(0).cpu().numpy()

            predicted_label = label_names[similarity.argmax()]
            confidence = round(similarity.max(), 2)
            results.append(f"{predicted_label} ({confidence}%)")

        except Exception as e:
            results.append(f"Error: {e}")

        elapsed_time = int(time.time() - start_time)
        progress = (index + 1) / num_images
        remaining_time = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
        progress_bar.progress(progress, f"Classifying images with CLIP... Time: {elapsed_time}s | Estimated Time Remaining: {int(remaining_time)}s")

    progress_bar.empty()
    gc.collect()
    return results


def load_rules(rules_file):
    rules_df = pd.read_csv(rules_file)
    required_columns = ["Rule", "Node", "Include", "Exclude"]
    missing_cols = [col for col in required_columns if col not in rules_df.columns]
    if missing_cols:
        st.error(f"Rules file is missing columns: {', '.join(missing_cols)}")
        return None
    return rules_df


def apply_rule(title, rule, rules_df):
    if not isinstance(rule, str) or rule == "Unclassified":
        return "Unclassified"

    if rules_df is None or rule not in rules_df["Rule"].values:
        return "Unclassified"

    rule_row = rules_df[rules_df["Rule"] == rule].iloc[0]

    include_text = str(rule_row["Include"]).lower() if pd.notna(rule_row["Include"]) else ""
    exclude_text = str(rule_row["Exclude"]).lower() if pd.notna(rule_row["Exclude"]) else ""

    title_lower = title.lower()

    # Include logic: handles "A and B or C"
    include_clauses = [clause.strip() for clause in include_text.split(" or ") if clause.strip()]
    include_match = False
    for clause in include_clauses:
        and_keywords = [kw.strip() for kw in clause.split(" and ") if kw.strip()]
        if all(kw in title_lower for kw in and_keywords):
            include_match = True
            break

    # Exclude logic: treat all keywords (even with 'and') as OR
    exclude_keywords = [kw.strip() for kw in exclude_text.replace(" and ", " or ").split(" or ") if kw.strip()]
    exclude_match = any(kw in title_lower for kw in exclude_keywords)

    if include_match and not exclude_match:
        return str(rule_row["Rule"])
    else:
        return "Unclassified"

# Streamlit App
uploaded_file = st.file_uploader("Upload product data (.csv or .xlsx)", type=["csv", "xlsx"])
rules_file = st.file_uploader("Upload rules file (CSV)", type=["csv"])

if uploaded_file and rules_file:
    try:
        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        rules_df = load_rules(rules_file)

        if rules_df is not None:
            title_col = "TITLE"
            image_url_columns = [col for col in df.columns if "IMAGE_URL" in col.upper()]

            if not title_col in df.columns or not image_url_columns:
                st.error("Data must contain a TITLE column and a column containing 'IMAGE_URL' in its name.")
            else:
                image_url_col = image_url_columns[0]

                n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=10)

                if st.button("Generate Elbow Plot"):
                    with st.spinner("Calculating WCSS for Elbow Method..."):
                        titles = df[title_col].tolist()
                        max_k = 20
                        wcss = calculate_wcss(titles, max_k)
                        fig, ax = plt.subplots()
                        ax.plot(range(1, max_k + 1), wcss, marker='o')
                        ax.set_title('Elbow Method')
                        ax.set_xlabel('Number of Clusters')
                        ax.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
                        st.pyplot(fig)
                    st.info("Look for the 'elbow' in the plot. Adjust the 'Number of clusters' slider based on the elbow and then run title clustering.")

                if "df_clustered" not in st.session_state and st.button("Run Title Clustering"):
                    with st.spinner("Clustering product titles..."):
                        labels, kmeans_model = cluster_titles(df[title_col].tolist(), n_clusters)
                        df["cluster"] = labels
                        st.session_state.df_clustered = df
                        st.session_state.kmeans_model = kmeans_model
                    st.success("Product title clustering complete!")

                if "df_clustered" in st.session_state:
                    df = st.session_state.df_clustered
                    unique_clusters = sorted(df["cluster"].unique())

                    st.subheader("Review Clusters and Assign Rules")
                    cluster_rules = {}
                    for cluster_id in unique_clusters:
                        st.write(f"**Cluster {cluster_id}:**")
                        sample_data = df[df["cluster"] == cluster_id].head(2)
                        for _, row in sample_data.iterrows():
                            try:
                                col1, col2 = st.columns([1, 2])
                                image = load_image(row[image_url_col])
                                if image:
                                    col1.image(image, width=150)
                                col2.write(f"**Title:** {row[title_col]}")
                            except Exception as e:
                                st.warning(f"Could not display image/title: {e}")

                        rule_options = ["Unclassified"] + rules_df["Rule"].tolist()
                        selected_rule = st.selectbox(
                            f"Rule for Cluster {cluster_id}",
                            rule_options,
                            key=f"rule_{cluster_id}",
                        )
                        cluster_rules[cluster_id] = selected_rule

                    if st.button("Apply Rules to Products"):
                        with st.spinner("Applying rules..."):
                            df["manual_label"] = df.apply(
                                lambda row: apply_rule(
                                    row[title_col], cluster_rules.get(row["cluster"]), rules_df
                                ),
                                axis=1,
                            )
                            st.session_state.df_labeled = df
                        st.success("Rules applied to products!")

                if "df_labeled" in st.session_state:
                    df = st.session_state.df_labeled

                    if st.checkbox("Run CLIP Classification"):
                        label_names = sorted(df["manual_label"].unique().tolist())
                        df_valid = df[df[image_url_col].notna() & df[image_url_col].str.startswith(("http://", "https://"))].copy()
                        df_valid["clip_label"] = classify_images_with_clip(
                            df_valid, clip_model, preprocess, label_names, image_url_col
                        )
                        df = pd.merge(df, df_valid[["clip_label"]], left_index=True, right_index=True, how="left")
                        st.session_state.df_classified = df
                        st.success("CLIP image classification complete!")

                    st.subheader("Classification Results")
                    st.dataframe(df.head(20))

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        st.download_button(
                            "Download Classified CSV",
                            tmp_file.name,
                            file_name="classified_products.csv",
                        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
