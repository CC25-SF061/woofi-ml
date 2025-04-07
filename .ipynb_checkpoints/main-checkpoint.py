from fastapi import FastAPI, Query
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging
import pickle

# Setup logging
logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Inisialisasi FastAPI
app = FastAPI()

# Load model dan data
with open("woofi_model.pkl", "rb") as f:
    tfidf, cosine_sim, df = pickle.load(f)


# Fungsi Rekomendasi
def rekomendasi_konten(query: str, kategori: Optional[str] = None, provinsi: Optional[str] = None, top_k: int = 5):
    tfidf_query = tfidf.transform([query])
    similarity_scores = cosine_sim.dot(tfidf_query.T).toarray().ravel()
    df["score"] = similarity_scores

    # Filter berdasarkan kategori
    filtered_df = df.copy()
    if kategori:
        kolom_kategori = [col for col in df.columns if kategori.lower().strip() in col.lower() and "Kategori_" in col]
        if kolom_kategori:
            filtered_df = filtered_df[filtered_df[kolom_kategori[0]] == 1]

    # Filter berdasarkan provinsi
    if provinsi:
        kolom_provinsi = [col for col in df.columns if provinsi.lower().strip() in col.lower() and "Provinsi_" in col]
        if kolom_provinsi:
            filtered_df = filtered_df[filtered_df[kolom_provinsi[0]] == 1]

    hasil = filtered_df.sort_values(by="score", ascending=False).head(top_k)
    return hasil[["NameLocation", "Penjelasan", "Rating", "LinkGmaps", "Foto"]].to_dict(orient="records")



# Endpoint rekomendasi
@app.get("/rekomendasi")
def get_rekomendasi(
    q: str = Query(..., description="Masukkan kata kunci pencarian"),
    kategori: Optional[str] = Query(None, description="Nama kategori (misal: Gunung, Pantai)"),
    provinsi: Optional[str] = Query(None, description="Nama provinsi (misal: Bali, Sumatera Selatan)")
):
    logging.info(f"Rekomendasi request | q='{q}' | kategori='{kategori}' | provinsi='{provinsi}'")
    hasil = rekomendasi_konten(q, kategori, provinsi)
    logging.info(f"Rekomendasi ditemukan: {len(hasil)} item")
    return {"rekomendasi": hasil}


