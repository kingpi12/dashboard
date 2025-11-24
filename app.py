import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Анализ техногенных пожаров",
    page_icon="🔥",
    layout="wide"
)

def main():
    st.title("🔥 Анализ техногенных пожаров")
    st.markdown("Загрузите Excel файл для автоматического анализа")
    
    # Загрузка файла
    uploaded_file = st.file_uploader("📁 Загрузите Excel файл", type=['xlsx', 'xls'])
    
    if not uploaded_file:
        show_requirements()
        return
    
    try:
        # Загрузка данных
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Файл загружен! Записей: {len(df)}")
        
        # Автоматический анализ
        auto_analyze(df)
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")

def show_requirements():
    """Показ требований к данным"""
    st.info("""
    ### 📋 Требования к данным:
    
    **Основные колонки:**
    - Дата возникновения
    - Муниципальный район  
    - Причина пожара
    - Погибло людей: Всего
    - Получили травмы: Всего
    
    **Приложение автоматически определит структуру данных!**
    """)

def auto_analyze(df):
    """Автоматический анализ данных"""
    
    # Основные метрики
    st.header("📊 Ключевые показатели")
    
    total_fires = len(df)
    
    # Автоматический поиск колонок
    deaths_col = find_column(df, ['погибло', 'death', 'умер'])
    injured_col = find_column(df, ['травм', 'injury', 'пострадал'])
    district_col = find_column(df, ['район', 'district', 'муниципальный'])
    date_col = find_column(df, ['дата', 'date'])
    cause_col = find_column(df, ['причина', 'cause'])
    
    total_deaths = df[deaths_col].sum() if deaths_col else 0
    total_injured = df[injured_col].sum() if injured_col else 0
    total_districts = df[district_col].nunique() if district_col else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего пожаров", total_fires)
    
    with col2:
        st.metric("Погибло людей", int(total_deaths))
    
    with col3:
        st.metric("Травмировано", int(total_injured))
    
    with col4:
        st.metric("Районов", total_districts)
    
    # Создание вкладок
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Динамика", 
        "🏢 Районы", 
        "🔍 Причины",
        "📅 Сезонность",
        "📊 Данные"
    ])
    
    with tab1:
        show_dynamics(df, date_col)
    
    with tab2:
        show_districts(df, district_col)
    
    with tab3:
        show_causes(df, cause_col)
    
    with tab4:
        show_seasonality(df, date_col)
    
    with tab5:
        show_data_preview(df)

def find_column(df, keywords):
    """Поиск колонки по ключевым словам"""
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in keywords:
            if keyword in col_lower:
                return col
    return None

def show_dynamics(df, date_col):
    """1. Динамика пожаров по годам"""
    st.header("1. Динамика количества пожаров по годам")
    
    if date_col:
        try:
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed'])
            df_temp['year'] = df_temp['date_parsed'].dt.year
            
            yearly_data = df_temp['year'].value_counts().sort_index()
            
            fig = px.line(
                x=yearly_data.index,
                y=yearly_data.values,
                title='Динамика количества пожаров по годам',
                labels={'x': 'Год', 'y': 'Количество пожаров'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка анализа дат: {str(e)}")
    else:
        st.warning("Для анализа динамики нужна колонка с датами")

def show_districts(df, district_col):
    """2. Распределение пожаров по районам"""
    st.header("2. Распределение пожаров по районам")
    
    if district_col:
        district_data = df[district_col].value_counts().head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                district_data,
                orientation='h',
                title='Рейтинг районов по количеству пожаров',
                labels={'index': 'Район', 'value': 'Количество пожаров'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                district_data.head(8),
                values=district_data.head(8).values,
                names=district_data.head(8).index,
                title='Доля пожаров по районам (Топ-8)'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Для анализа по районам нужна соответствующая колонка")

def show_causes(df, cause_col):
    """3. Основные причины пожаров"""
    st.header("3. Основные причины возникновения пожаров")
    
    if cause_col:
        cause_data = df[cause_col].value_counts().head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                cause_data,
                orientation='h',
                title='Основные причины пожаров',
                labels={'index': 'Причина', 'value': 'Количество'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                cause_data.head(8),
                values=cause_data.head(8).values,
                names=cause_data.head(8).index,
                title='Распределение причин (Топ-8)'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Для анализа причин нужна соответствующая колонка")

def show_seasonality(df, date_col):
    """5. Сезонность пожаров"""
    st.header("5. Сезонность (распределение по месяцам)")
    
    if date_col:
        try:
            df_temp = df.copy()
            df_temp['date_parsed'] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['date_parsed'])
            df_temp['month'] = df_temp['date_parsed'].dt.month
            
            monthly_data = df_temp['month'].value_counts().sort_index()
            month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 
                          'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            
            fig = px.line(
                x=month_names,
                y=monthly_data.values,
                title='Распределение пожаров по месяцам',
                labels={'x': 'Месяц', 'y': 'Количество пожаров'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Ошибка анализа сезонности: {str(e)}")
    else:
        st.warning("Для анализа сезонности нужна колонка с датами")

def show_data_preview(df):
    """Просмотр данных"""
    st.header("📋 Предпросмотр данных")
    
    st.subheader("Структура данных:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Колонки:**")
        for i, col in enumerate(df.columns):
            st.write(f"{i+1}. {col}")
    
    with col2:
        st.write("**Типы данных:**")
        st.write(df.dtypes.astype(str))
    
    st.subheader("Первые 10 записей:")
    st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
