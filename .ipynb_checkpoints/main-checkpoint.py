from fastapi import FastAPI, Query
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging
import pickle

# Tambahkan import untuk translasi multibahasa
from googletrans import Translator

# Inisialisasi FastAPI dan Translator
app = FastAPI()
translator = Translator()

# Setup logging
logging.basicConfig(
    filename="api_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load model dan data
with open("woofi_model.pkl", "rb") as f:
    tfidf, cosine_sim, df = pickle.load(f)

# Fungsi Rekomendasi
def rekomendasi_konten(query: str, kategori: Optional[str] = None, provinsi: Optional[str] = None, top_k: Optional[int] = None):
    try:
        translated_query = translator.translate(query, dest='en').text
        logging.info(f"Query diterjemahkan: '{query}' -> '{translated_query}'")
    except Exception as e:
        logging.warning(f"Gagal menerjemahkan query '{query}', error: {e}")
        translated_query = query  # fallback pakai query asli

    # Transform ke bentuk vektor
    tfidf_query = tfidf.transform([translated_query])
    similarity_scores = cosine_sim.dot(tfidf_query.T).toarray().ravel()
    df["score"] = similarity_scores

    # Filter berdasarkan kategori jika ada
    filtered_df = df.copy()
    if kategori:
        kolom_kategori = [col for col in df.columns if kategori.lower().strip() in col.lower() and "Kategori_" in col]
        if kolom_kategori:
            filtered_df = filtered_df[filtered_df[kolom_kategori[0]] == 1]

    # Filter berdasarkan provinsi jika ada
    if provinsi:
        provinsi_cols = [col for col in df.columns if col.startswith("Provinsi_") and provinsi.lower() in col.lower()]
        if provinsi_cols:
            mask = df[provinsi_cols].sum(axis=1) > 0
            filtered_df = filtered_df[mask]


    # Ambil top hasil rekomendasi
    hasil = filtered_df.sort_values(by="score", ascending=False).head(top_k)
    return hasil[["NameLocation", "Penjelasan_English", "Rating", "LinkGmaps", "Foto"]].to_dict(orient="records")

# Endpoint rekomendasi
@app.get("/rekomendasi")
def get_rekomendasi(
    q: str = Query(..., description="Masukkan kata kunci pencarian (dalam bahasa apa saja)"),
    kategori: Optional[str] = Query(None, description="Nama kategori (misal: Gunung, Pantai)"),
    provinsi: Optional[str] = Query(None, description="Nama provinsi (misal: Bali, Sumatera Selatan)")
):
    logging.info(f"Rekomendasi request | q='{q}' | kategori='{kategori}' | provinsi='{provinsi}'")
    hasil = rekomendasi_konten(q, kategori, provinsi)
    logging.info(f"Rekomendasi ditemukan: {len(hasil)} item")
    return {"rekomendasi": hasil}
