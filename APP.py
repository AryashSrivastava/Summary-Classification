"""Import all required libraries."""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from openpyxl import load_workbook
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from umap import UMAP
from collections import Counter
from scipy.spatial import ConvexHull
from wordcloud import WordCloud

import streamlit as st
import pandas as pd
import io
import ds_203_project_revision_2_2 as backend

st.set_page_config(layout="wide", page_title="Data Analysis App")

# #Backend
# backend.backend_import_libraries()


st.title("Data Analysis Application")

## Function 1: File Upload and Processing

st.header("1. File Upload and Processing")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xls') or uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            df = None

        # Process the dataframe ...
        df = backend.backend_process_data(df)
        backend.backend_summary_stats(df)
        df, vocab_size, token_to_idx = backend.backend_tokenize_selfvocab_v1(df)
        df, vocab_size, token_to_idx = backend.backend_tokenize_selfvocab_v2(df)
        df = backend.backend_average_onehot_vectors(df,vocab_size)
        df, X, kmeans = backend.backend_kmeans_clustering_basic(df, k=20)
        df, embedding_2d = backend.backend_visualize_umap_clusters(df, X)
        backend.backend_save_clusters_to_excel(df, sheet_name="Basic_Euclidean_avg")        
        df, X_normalized, kmeans = backend.backend_cosine_similarity_clustering(df)
        backend.backend_save_cosine_clusters_to_excel(df)
        df = backend.internal_clean_dataframe_(df)
        w2v_model, sentences = backend.backend_train_word2vec_model(df)
        df = backend.backend_create_sentence_vectors(df, w2v_model)
        df, C, kmeans = backend.backend_cluster_word2vec_embeddings(df)
        backend.backend_save_word2vec_clusters_to_excel(df)
        df, X, kmeans = backend.backend_cluster_normalized_word2vec(df)
        df, embedding_2d = backend.backend_visualize_clusters_with_hulls(df,X)
        backend.backend_save_cosine_word2vec_to_excel(df)
        df, tfidf_matrix, index_to_word = backend.backend_create_tfidf_hybrid_embeddings(df,w2v_model)
        df, X, kmeans = backend.backend_cluster_hybrid_embeddings(df)
        backend.backend_save_hybrid_clusters_to_excel(df)
        df, embedding_2d = backend.backend_visualize_hybrid_embeddings(df, X)
        df = backend.backend_visualize_hybrid_clusters_with_hulls(df, X)
        cluster_wordclouds = backend.backend_create_wordclouds(df)
        df = backend.backend_analyze_summary_lengths(df)
        df, top_summaries = backend.backend_tokenize_tfidf(df)
        # # Display images from fn_out1
        # images = fn_out1()
        # for i, img in enumerate(images):
        #     st.image(img, caption=f"Image {i+1}")
        
        # Display CSV or Excel from fn_out1
        output_df = df
        st.write("Output Data:")
        st.dataframe(output_df)
        
        # Provide download button for the output file
        output_buffer = io.BytesIO()
        if uploaded_file.name.endswith('.csv'):
            output_df.to_csv(output_buffer, index=False)
            file_type = "csv"
        else:
            output_df.to_excel(output_buffer, index=False)
            file_type = "xlsx"
        
        st.download_button(
            label="Download processed file",
            data=output_buffer.getvalue(),
            file_name=f"processed_output.{file_type}",
            mime=f"application/{file_type}"
        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

## Function 2: Phrase Input and Matrix Display

st.header("2. Phrase Analysis")

phrase = st.text_area("Enter a phrase (long string):")

if st.button("Analyze Phrase"):
    if phrase:
        # Process the phrase with fn_in2
        keyword_list = backend.tokenize_and_lemmatize(phrase)
        # Get matrix from fn_out2
        matrix = backend.backend_find_most_relevant_session(keyword_list, df, model = w2v_model, cluster_col='cluster_tf-idf_wv', top_k=3)()
        
        st.write("Output Matrix:")
        st.dataframe(matrix)
    else:
        st.warning("Please enter a phrase to analyze.")

## Function 3: File Upload, Row Selection, and Image Display

st.header("3. Row Selection and Image Generation")

uploaded_file_2 = st.file_uploader("Choose a CSV or Excel file for row selection", type=["csv", "xlsx"])

if uploaded_file_2 is not None:
    try:
        if uploaded_file_2.name.endswith('.csv'):
            df2 = pd.read_csv(uploaded_file_2)
        else:
            df2 = pd.read_excel(uploaded_file_2)
        
        # Create dropdown for row selection
        row_options = [str(row) for row in df2.to_dict('records')]
        selected_row = st.selectbox("Select a row:", row_options)
        
        if st.button("Generate Image"):
            # Convert selected_row back to dictionary
            selected_row_dict = eval(selected_row)
            
            # Process the selected row with fn_in3
            fn_in3(selected_row_dict)
            
            # Get image from fn_out3
            output_image = fn_out3()
            
            st.image(output_image, caption="Generated Image")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.sidebar.info("This app demonstrates file processing, phrase analysis, and row-based image generation.")
