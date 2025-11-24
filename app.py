import streamlit as st
import pandas as pd
import plotly.express as px

# Простая версия без сложных зависимостей
st.set_page_config(page_title="Анализ пожаров", page_icon="🔥")

def main():
    st.title("🔥 Простой анализ данных о пожарах")
    
    uploaded_file = st.file_uploader("Загрузите Excel файл", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Чтение данных
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Успешно загружено {len(df)} записей")
            
            # Показ данных
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head())
            
            # Базовая статистика
            st.subheader("Основная статистика")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего записей", len(df))
            with col2:
                st.metric("Колонок", len(df.columns))
            with col3:
                st.metric("Пропущенных значений", df.isnull().sum().sum())
            
            # Простые графики
            if len(df.columns) >= 2:
                st.subheader("Простой анализ")
                
                # Автоматический поиск числовых колонок
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    # Гистограмма для первой числовой колонки
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Распределение {numeric_cols[0]}")
                    st.plotly_chart(fig)
                
                # Если есть колонка с датами
                date_cols = df.select_dtypes(include=['datetime']).columns
                if len(date_cols) > 0:
                    df['Год'] = df[date_cols[0]].dt.year
                    yearly_counts = df['Год'].value_counts().sort_index()
                    fig = px.line(x=yearly_counts.index, y=yearly_counts.values, 
                                title="Пожары по годам")
                    st.plotly_chart(fig)
                    
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
            st.info("Проверьте формат файла и попробуйте снова")

if __name__ == "__main__":
    main()
