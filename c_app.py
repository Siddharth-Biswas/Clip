import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
import io
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="wide")
st.title("Product Classifier")


@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, clip_model, preprocess


sentence_model, clip_model, preprocess = load_models()


def cluster_titles(titles, n_clusters=10):
    embeddings = sentence_model.encode(titles, show_progress_bar=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans


def classify_images_with_clip(df, clip_model, preprocess, label_names):
    device = "cpu"
    clip_model.eval()
    results = []

    # Pre-tokenize once
    text_tokens = clip.tokenize(label_names).to(device)

    with st.spinner("Classifying images with CLIP..."):
        for index, row in df.iterrows():
            try:
                image_url = row["IMAGE_URL"]
                try:
                    response = st.session_state.requests.get(image_url, stream=True)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                except Exception as e:
                    st.warning(f"Error loading image from URL '{image_url}': {e}")
                    results.append(f"Error: Could not load image from URL")
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
            except Exception as e:
                predicted_label = f"Error: {e}"
            results.append(predicted_label)
    return results


def load_rules(rules_file):
    rules_df = pd.read_csv(rules_file)
    if "Rule" not in rules_df.columns or "Node" not in rules_df.columns:
        st.error("Rules file must contain 'Rule' and 'Node' columns.")
        return None
    return rules_df


def apply_rule(title, rule, rules_df):
    if not isinstance(rule, str) or rule == "Unclassified":
        return "Unclassified"

    if rules_df is None or rule not in rules_df["Rule"].values:
        return "Unclassified"

    rule_row = rules_df[rules_df["Rule"] == rule].iloc[0]  # Get the first row

    include_keywords = (
        str(rule_row["Include"]).lower().split(" or ")
        if pd.notna(rule_row["Include"])
        else []
    )
    exclude_keywords = (
        str(rule_row["Exclude"]).lower().split(" and ")
        if pd.notna(rule_row["Exclude"])
        else []
    )

    title_lower = title.lower()
    include_match = all(
        keyword.strip() in title_lower for keyword in include_keywords if keyword.strip()
    )
    exclude_match = not any(
        keyword.strip() in title_lower for keyword in exclude_keywords if keyword.strip()
    )

    if include_match and exclude_match:
        return str(rule_row["Node"])
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

        if rules_df is not None:  # Proceed only if rules file is loaded correctly
            if "TITLE" not in df.columns or "IMAGE_URL" not in df.columns:
                st.error("Data must contain TITLE and IMAGE_URL columns.")
            else:
                # Initialize requests in session state
                if "requests" not in st.session_state:
                    import requests

                    st.session_state.requests = requests

                n_clusters = st.slider("Number of clusters", min_value=2, max_value=20, value=10)

                if "df_clustered" not in st.session_state and st.button(
                    "Run Title Clustering"
                ):
                    with st.spinner("Clustering product titles..."):
                        labels, kmeans_model = cluster_titles(
                            df["TITLE"].tolist(), n_clusters
                        )
                        df["cluster"] = labels
                        st.session_state.df_clustered = df
                        st.session_state.kmeans_model = kmeans_model

                if "df_clustered" in st.session_state:
                    df = st.session_state.df_clustered
                    unique_clusters = sorted(df["cluster"].unique())

                    st.subheader("Review Clusters and Assign Rules")
                    cluster_rules = {}
                    for cluster_id in unique_clusters:
                        st.write(f"**Cluster {cluster_id}:**")
                        # Display sample images and titles
                        sample_data = df[df["cluster"] == cluster_id].head(2)
                        for _, row in sample_data.iterrows():
                            try:
                                col1, col2 = st.columns([1, 2])  # Adjust proportions as needed
                                image = Image.open(
                                    io.BytesIO(
                                        st.session_state.requests.get(
                                            row["IMAGE_URL"], stream=True
                                        ).content
                                    )
                                )
                                col1.image(image, width=150)
                                col2.write(f"**Title:** {row['TITLE']}")
                            except Exception as e:
                                st.warning(f"Could not display image/title: {e}")

                        # Rule selection
                        rule_options = ["Unclassified"] + rules_df["Rule"].tolist()
                        selected_rule = st.selectbox(
                            f"Rule for Cluster {cluster_id}",
                            rule_options,
                            key=f"rule_{cluster_id}",
                        )
                        cluster_rules[cluster_id] = selected_rule

                    if st.button("Apply Rules to Products"):
                        df["manual_label"] = df.apply(
                            lambda row: apply_rule(
                                row["TITLE"],
                                cluster_rules.get(row["cluster"]),
                                rules_df,
                            ),
                            axis=1,
                        )
                        st.session_state.df_labeled = df

                if "df_labeled" in st.session_state:
                    df = st.session_state.df_labeled

                    if st.checkbox("Run CLIP Classification"):
                        label_names = sorted(df["manual_label"].unique().tolist())
                        df["clip_label"] = classify_images_with_clip(
                            df, clip_model, preprocess, label_names
                        )
                        st.session_state.df_classified = df

                    st.subheader("Classification Results")
                    st.dataframe(df.head(20))

                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".csv"
                    ) as tmp_file:
                        df.to_csv(tmp_file.name, index=False)
                        st.download_button(
                            "Download Classified CSV",
                            tmp_file.name,
                            file_name="classified_products.csv",
                        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
