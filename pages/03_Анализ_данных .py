import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Анализ данных",
    layout="wide"
)

st.title("Анализ данных (EDA)")

# Загрузка данных
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("updated_card_transdata.csv")
        return df
    except FileNotFoundError:
        st.error("Файл updated_card_transdata.csv не найден в корне проекта. Проверьте путь.")
        st.stop()

df = load_data()


st.subheader("Общая информация о датасете")

col_info1, col_info2 = st.columns([3, 2])

with col_info1:
    st.write("#### Первые 5 строк:")
    st.dataframe(df.head())

with col_info2:
    st.write("#### Краткая сводка")
    
    stats_data = {
        "Метрика": ["Размер (строки, столбцы)", "Пропущенные значения", "Полные дубликаты"],
        "Значение": [
            f"{df.shape[0]} × {df.shape[1]}",
            int(df.isna().sum().sum()),
            int(df.duplicated().sum())
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

st.markdown("#### Основная статистика по числовым признакам")
st.dataframe(df.describe().round(2))

target = "fraud"
st.markdown(f"#### Распределение целевой переменной `{target}` (в процентах)")

value_counts = df[target].value_counts(normalize=True) * 100
value_counts = value_counts.round(2)

figsize_val = (4.5, 3.0)

fig_bar, ax_bar = plt.subplots(figsize=figsize_val)
bars = ax_bar.bar(
    value_counts.index.astype(str),
    value_counts.values,
    color=['#66BB6A', '#EF5350'],
    width=0.5,
    edgecolor='black',
    linewidth=0.8
)

for bar in bars:
    height = bar.get_height()
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1.5,
        f'{height}%',
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

ax_bar.set_ylim(0, max(value_counts) + 15)
ax_bar.set_ylabel("Доля, %")
ax_bar.set_xlabel(target)
ax_bar.set_title("Соотношение классов")
ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
ax_bar.tick_params(labelsize=9)

st.pyplot(fig_bar, width="content")

st.subheader("Ключевые визуализации")

num_cols = df.select_dtypes(include=[np.number]).columns[:4].tolist()
bin_cols = [col for col in df.columns if df[col].nunique() <= 3 and col != target][:4]

cols = st.columns(2)

with cols[0]:
    if num_cols:
        fig1, ax1 = plt.subplots(figsize=figsize_val)
        sns.histplot(data=df, x=num_cols[0], kde=True, hue=target,
                     ax=ax1, bins=20, palette="muted", legend=False)
        ax1.set_title(f"Распределение: {num_cols[0]}")
        ax1.tick_params(labelsize=9)
        st.pyplot(fig1, width="content")

with cols[1]:
    y_col = num_cols[0] if num_cols else df.select_dtypes(include=[np.number]).columns[0]
    fig2, ax2 = plt.subplots(figsize=figsize_val)
    sns.boxplot(x=target, y=y_col, data=df, ax=ax2, hue=target, palette="Set2", legend=False)
    ax2.set_title(f"{y_col} по классам")
    ax2.tick_params(labelsize=9)
    st.pyplot(fig2, width="content")

cols2 = st.columns(2)

with cols2[0]:
    fig3, ax3 = plt.subplots(figsize=figsize_val)
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3,
                annot_kws={"size": 7}, linewidths=0.3, cbar_kws={'shrink': 0.8})
    ax3.set_title("Матрица корреляций")
    ax3.tick_params(labelsize=8)
    st.pyplot(fig3, width="content")

with cols2[1]:
    if bin_cols:
        fig4, ax4 = plt.subplots(figsize=figsize_val)
        sns.barplot(x=bin_cols[0], y=target, data=df, estimator=np.mean, ax=ax4, hue=bin_cols[0], palette="viridis", legend=False)
        ax4.set_ylabel(f"Доля {target}=1")
        ax4.set_title(f"Доля мошенничества по {bin_cols[0]}")
        ax4.tick_params(labelsize=9)
        st.pyplot(fig4, width="content")

if len(bin_cols) > 1:
    st.markdown("---")
    st.subheader("Другие бинарные признаки")
    cols3 = st.columns(min(3, len(bin_cols) - 1))
    for i, col in enumerate(bin_cols[1:]):
        with cols3[i % len(cols3)]:
            fig, ax = plt.subplots(figsize=figsize_val)
            sns.barplot(x=col, y=target, data=df, estimator=np.mean, ax=ax, hue=col, palette="muted", legend=False)
            ax.set_ylabel(f"Доля {target}=1")
            ax.set_title(col)
            ax.tick_params(labelsize=9)
            st.pyplot(fig, width="content")

st.success("Визуальный анализ завершён.")