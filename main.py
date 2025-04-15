from fastapi import FastAPI, Query
from typing import Optional, Dict
import pandas as pd
import numpy as np
import logging
import pickle
import joblib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

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
    tfidf, cosine_sim, wisata_df = pickle.load(f)

model = joblib.load("rekomendasi_wisata_model.pkl")
ohe = joblib.load("rekomendasi_encoder_kategori.pkl")
mlb = joblib.load("rekomendasi_encoder_interest.pkl")

# Ekstrak kategori dari wisata_df
kategori_columns = [col for col in wisata_df.columns if col.startswith("Kategori_")]
kategori_series = wisata_df[kategori_columns].idxmax(axis=1).str.replace("Kategori_", "", regex=False).str.strip()
wifi_df = pd.concat([wisata_df[["NameLocation"]], kategori_series.rename("Kategori")], axis=1)
wifi_df["Kategori"] = kategori_series

# Fungsi bantu

def calculate_age(birth_date_str):
    birth_date = datetime.fromisoformat(birth_date_str.replace('Z', '+00:00'))
    today = datetime.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

def recommend_for_new_user(gender, age, interest_list, top_n=5):
    interest_data = mlb.transform([interest_list])
    interest_df = pd.DataFrame(interest_data, columns=[f"interest_{c}" for c in mlb.classes_])
    dummy_kategori = pd.DataFrame(ohe.transform([["Taman"]]), columns=ohe.get_feature_names_out(["kategori"]))

    input_df = pd.concat([
        interest_df,
        dummy_kategori,
        pd.DataFrame([[gender, age, 1]], columns=["gender", "age", "count"])
    ], axis=1)

    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    proba = model.predict_proba(input_df)[0]
    top_indices = proba.argsort()[::-1][:top_n]
    labels = model.classes_[top_indices]

    rekomendasi = wisata_df[wisata_df["NameLocation"].isin(labels)].copy()
    rekomendasi["skor"] = rekomendasi["NameLocation"].apply(lambda x: proba[labels.tolist().index(x)] if x in labels else 0)
    rekomendasi = rekomendasi.sort_values(by="skor", ascending=False)

    return rekomendasi[["NameLocation", "Kategori", "skor"]].head(top_n).to_dict(orient="records")

def recommend_for_existing_user(user_data, wisata_df, top_n=5):
    user_interest = user_data.get("interest", [])
    searchs = user_data.get("searchs", [])
    searched_places = [s["name"] for s in searchs if isinstance(s, dict) and "name" in s]

    rekomendasi = wisata_df[
        wisata_df["Kategori"].isin(user_interest) &
        ~wisata_df["NameLocation"].isin(searched_places)
    ]

    return rekomendasi[["NameLocation", "Kategori"]].drop_duplicates().head(top_n).to_dict(orient="records")

# ========== Endpoint: Rekomendasi berdasarkan query ==========
@app.get("/rekomendasi")
def get_rekomendasi(
    q: str = Query(..., description="Masukkan kata kunci pencarian (dalam bahasa apa saja)"),
    kategori: Optional[str] = Query(None, description="Nama kategori (misal: Gunung, Pantai)"),
    provinsi: Optional[str] = Query(None, description="Nama provinsi (misal: Bali, Sumatera Selatan)"),
    top_k: Optional[int] = 5
):
    logging.info(f"Rekomendasi request | q='{q}' | kategori='{kategori}' | provinsi='{provinsi}'")
    try:
        translated_query = translator.translate(q, dest='en').text
    except Exception as e:
        logging.warning(f"Gagal menerjemahkan query '{q}', error: {e}")
        translated_query = q

    tfidf_query = tfidf.transform([translated_query])
    similarity_scores = cosine_sim.dot(tfidf_query.T).toarray().ravel()
    df = wisata_df.copy()
    df["score"] = similarity_scores

    if kategori:
        kolom_kategori = [col for col in df.columns if kategori.lower().strip() in col.lower() and "Kategori_" in col]
        if kolom_kategori:
            df = df[df[kolom_kategori[0]] == 1]

    if provinsi:
        provinsi_cols = [col for col in df.columns if col.startswith("Provinsi_") and provinsi.lower() in col.lower()]
        if provinsi_cols:
            mask = df[provinsi_cols].sum(axis=1) > 0
            df = df[mask]

    hasil = df.sort_values(by="score", ascending=False).head(top_k)
    logging.info(f"Rekomendasi ditemukan: {len(hasil)} item")
    return {"rekomendasi": hasil[["NameLocation", "Penjelasan_English", "Rating", "LinkGmaps", "Foto"]].to_dict(orient="records")}

# ========== Endpoint: Rekomendasi berdasarkan data user ==========
@app.post("/rekomendasi-user")
def get_rekomendasi_user(data: Dict):
    if data.get("is_new_user", True):
        hasil = recommend_for_new_user(
            gender=data["gender"],
            age=data["age"],
            interest_list=data["interest"],
            top_n=data.get("top_n", 5)
        )
        return {"tipe": "baru", "rekomendasi": hasil}
    else:
        hasil = recommend_for_existing_user(data, wisata_df, top_n=data.get("top_n", 5))
        return {"tipe": "lama", "rekomendasi": hasil}
