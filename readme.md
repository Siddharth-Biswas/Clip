# 🧠 Product Classifier App with Auto-Generated Labels + CLIP

This Streamlit app lets you upload a product dataset with titles and image URLs, and automatically:

1. Clusters product titles using NLP (SentenceTransformer + KMeans)
2. Generates meaningful category labels using top keywords from each cluster
3. Uses OpenAI's CLIP model to classify images into the discovered labels
4. Allows you to download the final dataset with text + image classifications

---

## 🚀 How It Works

### 🔠 Step 1: Upload a product file
Your file must contain:
- `TITLE` — product name
- `IMAGE_URL` — link to product image

### 🧠 Step 2: Title Clustering
- Uses [SentenceTransformer](https://www.sbert.net/) to embed product titles
- Applies **KMeans clustering** to group similar products
- Extracts top **TF-IDF keywords** from each cluster to create labels

### 🖼️ Step 3: Image Classification
- Uses [CLIP (Contrastive Language-Image Pre-training)](https://github.com/openai/CLIP)
- Compares each image against the discovered labels to find the best fit

---

## 📥 Example Input Format

| TITLE               | IMAGE_URL                         |
|---------------------|-----------------------------------|
| Men's running shoes | https://example.com/shoe1.jpg     |
| Women's handbag     | https://example.com/bag1.jpg      |
| Coffee mug          | https://example.com/mug1.jpg      |

---

## 📤 Output Columns

- `TITLE` — original product title  
- `text_cluster` — cluster ID from KMeans  
- `generated_label` — label based on top keywords in that cluster  
- `clip_image_label` — best label match for the image from CLIP

---

## 📦 Installation

```bash
git clone https://github.com/your-username/product-classifier-app.git
cd product-classifier-app
pip install -r requirements.txt
streamlit run app.py
