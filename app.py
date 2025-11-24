import streamlit as st
import pandas as pd
import plotly.express as px
import chardet

st.set_page_config(page_title="Анализ пожаров", page_icon="🔥", layout="wide")

def detect_encoding(file_content):
    """Определяем кодировку файла"""
    result = chardet.detect(file_content)
    return result['encoding']

def load_data_correctly(uploaded_file):
    """Правильная загрузка данных с обработкой разных форматов"""
    try:
        # Пробуем разные способы загрузки
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            # Для CSV определяем кодировку
            file_content = uploaded_file.getvalue()
            encoding = detect_encoding(file_content)
            uploaded_file.seek(0)  # Сбрасываем позицию чтения
            df = pd.read_csv(uploaded_file, encoding=encoding)
        else:
            st.error("Неподдерживаемый формат файла")
            return None
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {str(e)}")
        return None

def clean_data(df):
    """Очистка и подготовка данных"""
    # Удаляем полностью пустые колонки
    df = df.dropna(axis=1, how='all')
    
    # Заменяем странные значения на NaN
    df = df.replace(['', ' ', 'NULL', 'null', 'None', 'none'], pd.NA)
    
    # Попытка определить настоящие названия колонок
    if df.shape[1] >= 3:  # Если есть хотя бы 3 колонки
        # Предполагаем, что первая колонка - № п/п
        if df.columns[0] != '№ п/п':
            st.info(f"Первая колонка интерпретирована как '№ п/п': {df.columns[0]}")
        
    return df

def main():
    st.title("🔥 Корректный анализ данных о пожарах")
    
    uploaded_file = st.file_uploader("📁 Загрузите Excel или CSV файл", 
                                   type=['xlsx', 'xls', 'csv'])
    
    if not uploaded_file:
        st.info("""
        ### 📋 Требования к данным:
        **Обязательные колонки:**
        - № п/п (порядковый номер)
        - Муниципальный район
        - Дата возникновения
        - Причина пожара
        
        **Рекомендуемые:**
        - Погибло людей: Всего
        - Получили травмы: Всего
        - Адрес
        - Объект пожара
        """)
        return
    
    # Загрузка данных
    df = load_data_correctly(uploaded_file)
    if df is None:
        return
    
    # Очистка данных
    df = clean_data(df)
    
    st.success(f"✅ Загружено {len(df)} строк, {len(df.columns)} колонок")
    
    # Показываем реальную структуру данных
    st.header("🔍 Структура данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Названия колонок:")
        for i, col in enumerate(df.columns):
            st.write(f"{i+1}. `{col}`")
    
    with col2:
        st.subheader("Типы данных:")
        st.write(df.dtypes)
    
    # Показываем данные как есть
    st.header("📊 Предпросмотр данных")
    st.dataframe(df.head(10))
    
    # Автоматический анализ
    analyze_data(df)

def analyze_data(df):
    """Анализ данных с правильной интерпретацией колонок"""
    
    st.header("📈 Автоматический анализ")
    
    # Определяем возможные колонки по шаблонам
    date_columns = [col for col in df.columns if any(word in str(col).lower() 
                   for word in ['дата', 'date', 'время'])]
    
    district_columns = [col for col in df.columns if any(word in str(col).lower() 
                       for word in ['район', 'district', 'муниципальный'])]
    
    cause_columns = [col for col in df.columns if any(word in str(col).lower() 
                     for word in ['причина', 'cause', 'reason'])]
    
    death_columns = [col for col in df.columns if any(word in str(col).lower() 
                     for word in ['погибло', 'погиб', 'death', 'умер'])]
    
    injury_columns = [col for col in df.columns if any(word in str(col).lower() 
                      for word in ['травм', 'ранен', 'injury', 'пострадал'])]
    
    # Показываем обнаруженные колонки
    st.subheader("Обнаруженные колонки:")
    
    info_cols = st.columns(4)
    with info_cols[0]:
        if date_columns:
            st.metric("Даты", date_columns[0])
        else:
            st.metric("Даты", "Не найдено")
    
    with info_cols[1]:
        if district_columns:
            st.metric("Районы", district_columns[0])
        else:
            st.metric("Районы", "Не найдено")
    
    with info_cols[2]:
        if cause_columns:
            st.metric("Причины", cause_columns[0])
        else:
            st.metric("Причины", "Не найдено")
    
    with info_cols[3]:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Числовые колонки", len(numeric_cols))
    
    # Анализ по датам
    if date_columns:
        analyze_dates(df, date_columns[0])
    
    # Анализ по районам
    if district_columns:
        analyze_districts(df, district_columns[0])
    
    # Анализ причин
    if cause_columns:
        analyze_causes(df, cause_columns[0])
    
    # Анализ последствий
    if death_columns or injury_columns:
        analyze_consequences(df, death_columns, injury_columns)

def analyze_dates(df, date_col):
    """Анализ временных данных"""
    st.subheader("📅 Анализ по датам")
    
    try:
        # Пробуем разные форматы дат
        df['Дата_обработанная'] = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = df['Дата_обработанная'].notna()
        
        if valid_dates.sum() > 0:
            df_valid = df[valid_dates].copy()
            df_valid['Год'] = df_valid['Дата_обработанная'].dt.year
            df_valid['Месяц'] = df_valid['Дата_обработанная'].dt.month
            
            yearly = df_valid['Год'].value_counts().sort_index()
            fig = px.line(x=yearly.index, y=yearly.values, 
                         title='Пожары по годам')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Не удалось распознать даты в выбранной колонке")
            
    except Exception as e:
        st.error(f"Ошибка анализа дат: {str(e)}")

def analyze_districts(df, district_col):
    """Анализ по районам"""
    st.subheader("🏢 Анализ по районам")
    
    try:
        district_counts = df[district_col].value_counts().head(10)
        fig = px.bar(district_counts, orientation='h',
                    title='Топ-10 районов по количеству пожаров')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка анализа районов: {str(e)}")

def analyze_causes(df, cause_col):
    """Анализ причин"""
    st.subheader("🔍 Анализ причин")
    
    try:
        cause_counts = df[cause_col].value_counts().head(8)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(cause_counts, values=cause_counts.values,
                        names=cause_counts.index, title='Распределение причин')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(cause_counts, orientation='h',
                        title='Основные причины')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка анализа причин: {str(e)}")

def analyze_consequences(df, death_cols, injury_cols):
    """Анализ последствий"""
    st.subheader("💔 Анализ последствий")
    
    try:
        # Ищем числовые колонки с последствиями
        numeric_df = df.select_dtypes(include=['number'])
        
        if not numeric_df.empty:
            # Предполагаем, что первая числовая колонка может быть количеством
            first_numeric_col = numeric_df.columns[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Среднее значение", f"{numeric_df[first_numeric_col].mean():.1f}")
            
            with col2:
                st.metric("Максимальное значение", f"{numeric_df[first_numeric_col].max():.1f}")
            
            # Гистограмма распределения
            fig = px.histogram(numeric_df, x=first_numeric_col, 
                             title=f'Распределение {first_numeric_col}')
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка анализа последствий: {str(e)}")

if __name__ == "__main__":
    main()
