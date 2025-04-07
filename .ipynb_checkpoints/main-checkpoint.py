from fastapi import FastAPI, Query
from typing import List
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(title="API Rekomendasi Tempat Wisata")

# ========== LOAD MODEL ==========
with open("woofi_model.pkl", "rb") as f:
    tfidf, tfidf_matrix, wisata = pickle.load(f)

# ========== FUNGSI UTAMA ==========
def ambil_kategori_provinsi(row):
    provinsi = [col.replace("Provinsi_", "") for col in wisata.columns if col.startswith("Provinsi_") and row.get(col, 0) == 1]
    kategori = [col.replace("Kategori_", "") for col in wisata.columns if col.startswith("Kategori_") and row.get(col, 0) == 1]
    return pd.Series([
        provinsi[0] if provinsi else "", 
        kategori[0] if kategori else ""
    ])

def rekomendasi_konten(query: str, top_n: int = 5):
    input_tfidf = tfidf.transform([query])
    cosine_sim = cosine_similarity(input_tfidf, tfidf_matrix)
    top_indices = cosine_sim.argsort()[0][-top_n:][::-1]
    
    hasil = wisata.iloc[top_indices].copy()
    hasil[["Provinsi", "Kategori"]] = hasil.apply(ambil_kategori_provinsi, axis=1)
    
    return hasil[["NameLocation", "Provinsi", "Kategori", "Penjelasan", "Rating", "LinkGmaps"]]


@app.get("/rekomendasi/")
def get_rekomendasi(q: str = Query(..., description="Deskripsi tempat wisata yang diinginkan"), top_n: int = 5):
    hasil = rekomendasi_konten(q, top_n)
    return {"hasil": hasil.to_dict(orient="records")}
