# 🛍️ Product Classifier App

This Streamlit app helps classify products using both **text (title) clustering** and **image-based classification using CLIP**. It's designed for product ops, catalog, or analytics teams to manually cluster titles and classify product images based on a custom rules file.

---

## 📦 Features

- Upload product data with titles and image URLs.
- Cluster titles using SentenceTransformers (`MiniLM-L6-v2`).
- Manually assign category rules to clusters.
- Automatically classify product images using OpenAI's CLIP model.
- Apply rules for final classification.
- Export results as a CSV.

---

## 📁 File Uploads

- **Product Data:** `.csv` or `.xlsx` with at least `TITLE` and `IMAGE_URL` columns.
- **Rules File:** `.csv` containing columns:
  - `Rule`: Name of the classification rule.
  - `Node`: The final category/tag to be assigned.
  - `Include`: (optional) Keywords (separated by `or`) to match in title.
  - `Exclude`: (optional) Keywords (separated by `and`) to avoid in title.

---

## 🧰 Requirements

- Python 3.10

Install dependencies:

```bash
pip install -r requirements.txt


streamlit run app.py
