import streamlit as st
import os

st.set_page_config(
    page_title="Обо мне",
    layout="wide"
)

st.title("О разработчике")

# Используем колонки для размещения фото и текста рядом
col_photo, col_info = st.columns([1, 2])

with col_photo:
    image_path = "photo.png"
    if os.path.exists(image_path):
        st.image(image_path, width=220)
    else:
        st.warning("Файл photo.png не найден")

with col_info:
    st.markdown("### Репин Артур Александрович")
    st.write("Студент 3 курса, группа ФИТ-231, ОмГТУ")
    st.markdown("---")
    
    st.write("**Дисциплина:** Машинное обучение и анализ данных (MLDA)")
    st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")
    st.markdown("---")
    st.caption("ОмГТУ, февраль 2026")