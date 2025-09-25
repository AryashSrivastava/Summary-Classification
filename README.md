# Session Summaries Clustering & Retrieval

This project reconstructs the session-wise organization of a set of jumbled student summaries from a Data Science course, where original labels were lost. Using advanced NLP preprocessing, feature extraction, unsupervised clustering, and visualization, the pipeline groups similar summaries, recovers likely session associations, and provides a lightweight app for summary search and retrieval based on keyword input.

## Features

- **Unsupervised clustering** using Bag of Words, TF-IDF, Word2Vec, and hybrid embeddings.
- **Advanced text preprocessing:** cleaning, tokenization, normalization, lemmatization, and stopword removal.
- **Visualization:** UMAP dimensional reduction and word clouds to reveal cluster structure and frequent topics.
- **Internal evaluation** with Silhouette Score and manual validation for semantic grouping.
- **Ranking & search app:** Retrieve and rank the most relevant session summaries for given topics.

## Project Workflow

1. **Data Preprocessing:**
   - Clean and normalize text
   - Tokenize, lemmatize, remove stopwords and non-alphabetic characters

2. **Feature Engineering:**
   - Generate BoW, TF-IDF, and Word2Vec embeddings
   - Hybrid embedding: TF-IDF weighted Word2Vec averaging

3. **Clustering & Visualization:**
   - K-means/UMAP for cluster assignment and 2D visualization
   - Evaluate clusters using Silhouette Score and qualitative review

4. **Summary Retrieval App:**
   - Python-based keyword search returns top summaries by cluster proximity
   - Minimal UI for clarity

## Tools & Libraries

- Python 3.x
- NumPy, Pandas
- scikit-learn (KMeans, TF-IDF)
- Gensim (Word2Vec)
- NLTK (preprocessing)
- spaCy (optional for advanced NLP steps)
- UMAP-learn (visualization)
- Flask/Streamlit (for retrieval app, if implemented)
- matplotlib, wordcloud (visualizations)

## Installation

1. Clone the repository.
2. Install dependencies:
