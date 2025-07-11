import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

from config import (
    metabolites_selected_onco, metabolites_selected_CVD,
    metabolites_selected_Liv, metabolites_selected_PULM,
    metabolites_selected_RA
)

# Загрузка моделей один раз
@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Преобразование вероятности в 10-балльную шкалу
def probability_to_score(prob, threshold):
    prob = min(max(prob, 0), 1)
    if prob < threshold:
        score = 5 * prob / threshold
    else:
        score = 5 + 5 * (prob - threshold) / (1 - threshold)
    return round(score, 1)

# Конфигурация моделей
models_info = {
    "Onco": {
        "model_path": "Onco_healthy_RF_0907.pkl",
        "features": metabolites_selected_onco
    },
    "CVD": {
        "model_path": "CVD_healthy_RF_0907.pkl",
        "features": metabolites_selected_CVD
    },
    "Liver": {
        "model_path": "Liver_healthy_RF_0907.pkl",
        "features": metabolites_selected_Liv
    },
    "Pulmo": {
        "model_path": "Pulmo_healthy_RF_0907.pkl",
        "features": metabolites_selected_PULM
    },
    "RA": {
        "model_path": "RA_healthy_RF_0907.pkl",
        "features": metabolites_selected_RA
    }
}

# --- Streamlit UI ---
st.title("🎯 Метаболомный предсказатель по 5 патологиям")

uploaded_file = st.file_uploader("Загрузите файл с метаболомным профилем (Excel)", type=["xlsx"])

# Настройка порогов
st.sidebar.title("🔧 Пороги классификации")
thresholds = {}
for disease in models_info:
    thresholds[disease] = st.sidebar.slider(
        f"{disease} порог",
        min_value=0.0, max_value=1.0, value=0.6, step=0.01
    )

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    df_input = df_input.set_index(df_input.columns[0]) if df_input.columns[0] != "Название образца" else df_input

    results = []

    for idx, row in df_input.iterrows():
        row_result = {"ID": idx}
        for disease, info in models_info.items():
            try:
                model = load_model(info["model_path"])
                features = info["features"]
                row_data = pd.DataFrame([row[features].fillna(0)], columns=features)
                proba = model.predict_proba(row_data)[0][1]
                threshold = thresholds[disease]
                label = disease if proba >= threshold else "Здоровый"
                score = probability_to_score(proba, threshold)

                row_result.update({
                    f"{disease} — вероятность": round(proba, 2),
                    f"{disease} — исход": label,
                    f"{disease} — скор (0–10)": 10-score
                })

            except Exception as e:
                row_result[f"{disease} — ошибка"] = str(e)
        results.append(row_result)

    df_results = pd.DataFrame(results)
    st.success("✅ Расчёты завершены!")
    st.dataframe(df_results)
    excel_data = convert_df_to_excel(df_results)

    st.download_button(
        label="📥 Скачать результаты в Excel",
        data=excel_data,
        file_name="результаты_по_всем_моделям.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

