import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import joblib
import os
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Инференс моделей",
    layout="wide"
)

st.title("Обнаружение мошенничества")

FEATURE_COLS = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order"
]

@st.cache_resource
def load_models():
    base = "models/"
    models = {}
    try:
        models["ML1 Классическая"]   = joblib.load(base + "ml1_classic.joblib")
        models["ML2 Бустинг"]        = joblib.load(base + "ml2_boost.joblib")
        
        if os.path.exists(base + "ml3_catboost.cbm"):
            cb = CatBoostClassifier()
            cb.load_model(base + "ml3_catboost.cbm")
            models["ML3 CatBoost"] = cb
        elif os.path.exists(base + "ml3_xgboost.json"):
            models["ML3 XGBoost"] = XGBClassifier().load_model(base + "ml3_xgboost.json")
        else:
            models["ML3 Продвинутый"] = joblib.load(base + "ml3_adv.joblib")
            
        models["ML4 Бэггинг"]        = joblib.load(base + "ml4_bagging.joblib")
        
        with open(base + "ml5_stacking.pkl", "rb") as f:
            models["ML5 Stacking"] = pickle.load(f)
            
        models["ML6 Нейросеть"]      = load_model(base + "ml6_neuralnet.h5")
        
        return models
    except Exception as e:
        st.error(f"Ошибка загрузки моделей: {e}")
        st.stop()

models_dict = load_models()

model_name = st.selectbox("Выберите модель", list(models_dict.keys()))
model = models_dict[model_name]

tab1, tab2 = st.tabs(["Ручной ввод", "Загрузка CSV"])

def prepare_input_data(data):
    df = data.copy()
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) != len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available)
        st.warning(f"Отсутствуют признаки: {missing}. Предсказание может быть некорректным.")
    df = df[available] if available else df
    return df[FEATURE_COLS] if set(FEATURE_COLS).issubset(df.columns) else df

# Вкладка 1: Ручной ввод
with tab1:
    st.subheader("Введите данные одной транзакции")
    
    input_data = {}
    
    for col in FEATURE_COLS:
        if col == "distance_from_home":
            label = "Расстояние от дома до места транзакции (км)"
            val = st.number_input(label, min_value=0.0, value=10.0, step=0.1, key=f"num_{col}")
            input_data[col] = float(val)
            
        elif col == "distance_from_last_transaction":
            label = "Расстояние от предыдущей транзакции (км)"
            val = st.number_input(label, min_value=0.0, value=5.0, step=0.1, key=f"num_{col}")
            input_data[col] = float(val)
            
        elif col == "ratio_to_median_purchase_price":
            label = "Отношение суммы покупки к медиане по карте"
            val = st.number_input(label, min_value=0.0, value=1.0, step=0.01, key=f"num_{col}")
            input_data[col] = float(val)
            
        else:
            label = {
                "repeat_retailer": "Повторный продавец?",
                "used_chip": "Использован чип карты?",
                "used_pin_number": "Введён PIN-код?",
                "online_order": "Онлайн-заказ?"
            }[col]
            
            val = st.radio(
                label=label,
                options=["Нет", "Да"],
                horizontal=True,
                key=f"radio_{col}",
                index=0
            )
            input_data[col] = 1.0 if val == "Да" else 0.0
    
    if st.button("Выполнить предсказание", type="primary", use_container_width=True):
        try:
            df_input = pd.DataFrame([input_data])[FEATURE_COLS]
            
            is_nn = "neuralnet" in model_name.lower() or isinstance(model, tf.keras.Model)
            
            if is_nn:
                pred_prob = float(model.predict(df_input.values, verbose=0)[0][0])
                pred = 1 if pred_prob > 0.5 else 0
            else:
                pred = int(model.predict(df_input)[0])
                pred_prob = float(model.predict_proba(df_input)[0][1]) if hasattr(model, "predict_proba") else None
            
            if pred == 1:
                st.error(f"МОШЕННИЧЕСКАЯ ТРАНЗАКЦИЯ\nВероятность: {pred_prob:.2%}" if pred_prob else "МОШЕННИЧЕСКАЯ ТРАНЗАКЦИЯ")
            else:
                st.success(f"НОРМАЛЬНАЯ ТРАНЗАКЦИЯ\nВероятность мошенничества: {pred_prob:.2%}" if pred_prob else "НОРМАЛЬНАЯ ТРАНЗАКЦИЯ")
                
        except Exception as e:
            st.error(f"Ошибка предсказания: {str(e)}")

# Вкладка 2: Загрузка CSV
with tab2:
    st.subheader("Загрузите файл .csv")
    st.info(f"Ожидаемые признаки: {', '.join(FEATURE_COLS)}")
    
    uploaded = st.file_uploader("Выберите CSV-файл", type=["csv"])
    
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.write("Первые строки файла:")
            st.dataframe(df_upload.head())
            
            if st.button("Предсказать по файлу"):
                df_processed = prepare_input_data(df_upload)
                
                is_nn = "neuralnet" in model_name.lower() or isinstance(model, tf.keras.Model)
                
                if is_nn:
                    preds_prob = model.predict(df_processed.values, verbose=0).ravel()
                    preds = (preds_prob > 0.5).astype(int)
                else:
                    preds = model.predict(df_processed)
                    preds_prob = model.predict_proba(df_processed)[:, 1] if hasattr(model, "predict_proba") else None
                
                result_df = df_upload.copy()
                result_df["prediction"] = preds
                result_df["is_fraud"] = ["Да" if p == 1 else "Нет" for p in preds]
                
                if preds_prob is not None:
                    result_df["prob_fraud"] = preds_prob.round(4)
                
                st.write("Результаты:")
                st.dataframe(result_df.head())
                
                st.download_button(
                    label="Скачать результаты (CSV)",
                    data=result_df.to_csv(index=False).encode('utf-8'),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Ошибка обработки файла: {str(e)}")