# 🎬 Movie Recommendation System using Python

This project is a simple **Content-Based Movie Recommendation System** built in Python. It recommends similar movies based on the movie's description, genre, and director using NLP and cosine similarity.

---

## 📌 Features

- Text cleaning (stopword removal, punctuation filtering, lemmatization)
- Uses **TF Vectorization** to convert movie content to vectors
- Computes **cosine similarity** to find related movies
- Returns top 10 similar movies based on user input

---

## 🧠 Tech Stack

- Python
- Libraries used:
  - `pandas`, `numpy`
  - `nltk`
  - `scikit-learn`
  - `matplotlib`, `seaborn` (for optional visualizations)
  - `ydata-profiling` (for quick EDA)

---

## 📂 Dataset Used

- `netflix_titles.csv`  
- Make sure this file is correctly placed or update its path inside the code:
