import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Анализ пожаров", page_icon="🔥", layout="wide")

def smart_column_detection(df):
    """Умное определение назначения колонок"""
    column_types = {}
    
    for col in df.columns:
        col_str = str(col).lower()
        
        # Игнорируем колонки с нумерацией
        if any(word in col_str for word in ['№', 'п/п', 'номер', 'num', 'id', 'index']):
            column_types[col] = 'ignore'
        
        # Даты
        elif any(word in col_str for word in ['дата', 'date', 'время']):
            column_types[col] = 'date'
        
        # Районы
        elif any(word in col_str for word in ['район', 'муниципальный', 'district', 'город']):
            column_types[col] = 'district'
        
        # Причины
        elif any(word in col_str for word in ['причина', 'cause', 'reason']):
            column_types[col] = 'cause'
        
        # Адреса/места
        elif any(word in col_str for word in ['адрес', 'address', 'улица', 'дом', 'место', 'object']):
            column_types[col] = 'location'
        
        # Объекты
        elif any(word in col_str for word in ['объект', 'object', 'здание', 'building', 'наименование']):
            column_types[col] = 'object'
        
        # Погибшие
        elif any(word in col_str for word in ['погибло', 'погиб', 'death', 'умер', 'смерт']):
            column_types[col] = 'deaths'
        
        # Травмированные
        elif any(word in col_str for word in ['травм', 'ранен', 'injury', 'пострадал']):
            column_types[col] = 'injured'
        
        # Числовые данные
        elif pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == len(df) and (df[col] == np.arange(1, len(df)+1)).all():
                column_types[col] = 'ignore'
            else:
                column_types[col] = 'numeric'
        
        else:
            column_types[col] = 'other'
    
    return column_types

def create_forecast(df, date_col):
    """Создание прогноза на будущее"""
    try:
        df_temp = df.copy()
        df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=['date'])
        
        monthly_data = df_temp.groupby([df_temp['date'].dt.year, df_temp['date'].dt.month]).size()
        monthly_data = monthly_data.reset_index(name='count')
        monthly_data['period'] = monthly_data['date'].dt.year * 12 + monthly_data['date'].dt.month
        
        if len(monthly_data) < 6:
            return None
        
        X = monthly_data[['period']]
        y = monthly_data['count']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Прогноз на 6 месяцев
        last_period = monthly_data['period'].max()
        future_periods = [last_period + i for i in range(1, 7)]
        predictions = model.predict(pd.DataFrame(future_periods, columns=['period']))
        
        return list(zip(future_periods, predictions))
    
    except:
        return None

def main():
    st.title("🔥 Полный анализ техногенных пожаров")
    
    uploaded_file = st.file_uploader("📁 Загрузите Excel файл с данными о пожарах", 
                                   type=['xlsx', 'xls'])
    
    if not uploaded_file:
        show_requirements()
        return
    
    try:
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Загружено {len(df)} записей")
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {str(e)}")
        return
    
    # Определение колонок
    column_types = smart_column_detection(df)
    
    # Боковая панель с фильтрами
    st.sidebar.header("🔧 Фильтры и настройки")
    
    # Фильтр по годам
    date_col = next((col for col, type_ in column_types.items() if type_ == 'date'), None)
    if date_col:
        df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df['date_parsed'].dt.year
        available_years = sorted(df['year'].dropna().unique())
        
        if available_years:
            selected_years = st.sidebar.multiselect(
                "Выберите годы:",
                options=available_years,
                default=available_years
            )
            df = df[df['year'].isin(selected_years)]
    
    # Основные метрики
    show_main_metrics(df, column_types)
    
    # Вкладки для всех анализов
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Динамика по годам", 
        "🗺️ Районы", 
        "🔍 Причины",
        "🏢 Места",
        "📅 Сезонность", 
        "📊 По районам",
        "🔄 Сравнение АППГ",
        "🔮 Прогноз"
    ])
    
    with tab1:
        show_yearly_dynamics(df, column_types)  # 1. Динамика по годам
    
    with tab2:
        show_district_analysis(df, column_types)  # 2. Районы
    
    with tab3:
        show_cause_analysis(df, column_types)  # 3. Причины
    
    with tab4:
        show_location_analysis(df, column_types)  # 4. Места возникновения
    
    with tab5:
        show_seasonality_analysis(df, column_types)  # 5. Сезонность
    
    with tab6:
        show_district_dynamics(df, column_types)  # 6. Динамика по районам
    
    with tab7:
        show_year_comparison(df, column_types)  # 7. Сравнение АППГ
    
    with tab8:
        show_forecast(df, column_types)  # Прогноз

def show_requirements():
    """Показ требований к данным"""
    st.info("""
    ### 📋 Требования к данным для полного анализа:
    
    **Обязательные колонки:**
    - № п/п (игнорируется)
    - Дата возникновения
    - Муниципальный район
    - Причина пожара
    
    **Рекомендуемые:**
    - Адрес / Место возникновения
    - Объект пожара
    - Погибло людей: Всего
    - Получили травмы: Всего
    - Этажность здания
    - Степень огнестойкости
    """)

def show_main_metrics(df, column_types):
    """Основные метрики"""
    st.header("📊 Ключевые показатели")
    
    total_fires = len(df)
    
    deaths_col = next((col for col, type_ in column_types.items() if type_ == 'deaths'), None)
    injured_col = next((col for col, type_ in column_types.items() if type_ == 'injured'), None)
    district_col = next((col for col, type_ in column_types.items() if type_ == 'district'), None)
    
    total_deaths = df[deaths_col].sum() if deaths_col and deaths_col in df.columns else 0
    total_injured = df[injured_col].sum() if injured_col and injured_col in df.columns else 0
    total_districts = df[district_col].nunique() if district_col and district_col in df.columns else 0
    
    # Сравнение с прошлым годом
    date_col = next((col for col, type_ in column_types.items() if type_ == 'date'), None)
    current_year = df['year'].max() if 'year' in df.columns else None
    previous_year = current_year - 1 if current_year else None
    
    if current_year and previous_year:
        current_year_fires = len(df[df['year'] == current_year])
        previous_year_fires = len(df[df['year'] == previous_year]) if previous_year in df['year'].values else 0
        change = current_year_fires - previous_year_fires
    else:
        change = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Всего пожаров", total_fires)
    
    with col2:
        st.metric("Погибло людей", int(total_deaths))
    
    with col3:
        st.metric("Травмировано", int(total_injured))
    
    with col4:
        st.metric("Районов", total_districts, delta=change)

def show_yearly_dynamics(df, column_types):
    """1. Динамика количества пожаров по годам"""
    st.header("1. Общая динамика количества пожаров по годам")
    
    if 'year' in df.columns:
        yearly_data = df['year'].value_counts().sort_index()
        
        fig = px.line(
            x=yearly_data.index, 
            y=yearly_data.values,
            title='Динамика количества пожаров по годам',
            labels={'x': 'Год', 'y': 'Количество пожаров'},
            markers=True
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
        
        # Дополнительная статистика
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Среднее в год", f"{yearly_data.mean():.1f}")
        with col2:
            st.metric("Максимум в год", yearly_data.max())
        with col3:
            st.metric("Минимум в год", yearly_data.min())
    else:
        st.warning("Для анализа динамики нужна колонка с датами")

def show_district_analysis(df, column_types):
    """2. Распределение пожаров по районам"""
    st.header("2. Распределение пожаров по районам")
    
    district_col = next((col for col, type_ in column_types.items() if type_ == 'district'), None)
    
    if district_col:
        # Рейтинг районов
        district_data = df[district_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Топ-15 районов
            fig = px.bar(
                district_data.head(15),
                orientation='h',
                title='Рейтинг районов по количеству пожаров',
                labels={'index': 'Район', 'value': 'Количество пожаров'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Круговой график для топ-10
            fig = px.pie(
                district_data.head(10),
                values=district_data.head(10).values,
                names=district_data.head(10).index,
                title='Доля пожаров по районам (Топ-10)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Карта (заглушка - в реальном проекте нужны координаты)
        st.subheader("Карта распределения пожаров")
        st.info("""
        🗺️ Для отображения на карте необходимы:
        - Геокоординаты в отдельном столбце
        - Или данные о широте/долготе
        """)
        
    else:
        st.warning("Для анализа по районам нужна соответствующая колонка")

def show_cause_analysis(df, column_types):
    """3. Основные причины возникновения пожаров"""
    st.header("3. Основные причины возникновения пожаров")
    
    cause_col = next((col for col, type_ in column_types.items() if type_ == 'cause'), None)
    
    if cause_col:
        cause_data = df[cause_col].value_counts().head(15)
        
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
        
        # Анализ причин по годам
        if 'year' in df.columns:
            st.subheader("Динамика основных причин")
            yearly_causes = df.groupby(['year', cause_col]).size().unstack(fill_value=0)
            top_causes = cause_data.head(5).index
            
            fig = go.Figure()
            for cause in top_causes:
                if cause in yearly_causes.columns:
                    fig.add_trace(go.Scatter(
                        x=yearly_causes.index,
                        y=yearly_causes[cause],
                        name=cause,
                        mode='lines+markers'
                    ))
            
            fig.update_layout(title='Динамика основных причин по годам')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Для анализа причин нужна соответствующая колонка")

def show_location_analysis(df, column_types):
    """4. Наиболее частые места возникновения пожаров"""
    st.header("4. Наиболее частые места возникновения пожаров")
    
    location_col = next((col for col, type_ in column_types.items() if type_ == 'location'), None)
    object_col = next((col for col, type_ in column_types.items() if type_ == 'object'), None)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if location_col:
            location_data = df[location_col].value_counts().head(10)
            fig = px.bar(
                location_data,
                orientation='h',
                title='Топ-10 мест возникновения',
                labels={'index': 'Место', 'value': 'Количество'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Добавьте колонку с адресами для анализа мест")
    
    with col2:
        if object_col:
            object_data = df[object_col].value_counts().head(10)
            fig = px.bar(
                object_data,
                orientation='h',
                title='Топ-10 объектов пожаров',
                labels={'index': 'Объект', 'value': 'Количество'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Добавьте колонку с объектами для анализа")

def show_seasonality_analysis(df, column_types):
    """5. Сезонность (распределение по месяцам)"""
    st.header("5. Сезонность пожаров")
    
    if 'date_parsed' in df.columns:
        df_temp = df.dropna(subset=['date_parsed']).copy()
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
        
        # Тепловая карта по годам и месяцам
        if 'year' in df_temp.columns:
            heatmap_data = df_temp.groupby(['year', 'month']).size().unstack(fill_value=0)
            fig = px.imshow(
                heatmap_data.T,
                title='Тепловая карта: пожары по годам и месяцам',
                labels=dict(x="Год", y="Месяц", color="Количество"),
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Для анализа сезонности нужны данные с датами")

def show_district_dynamics(df, column_types):
    """6. Динамика основных показателей по районам"""
    st.header("6. Динамика показателей по районам")
    
    district_col = next((col for col, type_ in column_types.items() if type_ == 'district'), None)
    
    if district_col and 'year' in df.columns:
        # Выбор показателя
        metric_options = ['Количество пожаров', 'Погибло', 'Травмировано']
        selected_metric = st.selectbox("Выберите показатель:", metric_options)
        
        if selected_metric == 'Количество пожаров':
            district_year_data = df.groupby([district_col, 'year']).size().reset_index(name='value')
        else:
            col_name = 'deaths_col' if selected_metric == 'Погибло' else 'injured_col'
            data_col = next((col for col, type_ in column_types.items() if type_ == col_name), None)
            
            if data_col:
                district_year_data = df.groupby([district_col, 'year'])[data_col].sum().reset_index()
                district_year_data.columns = [district_col, 'year', 'value']
            else:
                st.warning(f"Нет данных для показателя: {selected_metric}")
                return
        
        # Топ-5 районов по последнему году
        last_year = district_year_data['year'].max()
        top_districts = district_year_data[
            district_year_data['year'] == last_year
        ].nlargest(5, 'value')[district_col].tolist()
        
        filtered_data = district_year_data[
            district_year_data[district_col].isin(top_districts)
        ]
        
        fig = px.line(
            filtered_data,
            x='year',
            y='value',
            color=district_col,
            title=f'Динамика {selected_metric.lower()} по топ-5 районам',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Для анализа нужны данные по районам и годам")

def show_year_comparison(df, column_types):
    """7. Сравнение с аналогичным периодом прошлого года"""
    st.header("7. Сравнение с прошлым годом (АППГ)")
    
    if 'year' in df.columns:
        current_year = df['year'].max()
        previous_year = current_year - 1
        
        if previous_year in df['year'].values:
            current_data = df[df['year'] == current_year]
            previous_data = df[df['year'] == previous_year]
            
            # Сравниваемые показатели
            comparisons = []
            
            # Количество пожаров
            current_fires = len(current_data)
            previous_fires = len(previous_data)
            fires_change = current_fires - previous_fires
            fires_change_pct = (fires_change / previous_fires * 100) if previous_fires > 0 else 0
            comparisons.append(('Количество пожаров', current_fires, previous_fires, fires_change_pct))
            
            # Погибшие
            deaths_col = next((col for col, type_ in column_types.items() if type_ == 'deaths'), None)
            if deaths_col:
                current_deaths = current_data[deaths_col].sum()
                previous_deaths = previous_data[deaths_col].sum()
                deaths_change_pct = ((current_deaths - previous_deaths) / previous_deaths * 100) if previous_deaths > 0 else 0
                comparisons.append(('Погибло людей', current_deaths, previous_deaths, deaths_change_pct))
            
            # Травмированные
            injured_col = next((col for col, type_ in column_types.items() if type_ == 'injured'), None)
            if injured_col:
                current_injured = current_data[injured_col].sum()
                previous_injured = previous_data[injured_col].sum()
                injured_change_pct = ((current_injured - previous_injured) / previous_injured * 100) if previous_injured > 0 else 0
                comparisons.append(('Травмировано', current_injured, previous_injured, injured_change_pct))
            
            # Визуализация сравнения
            fig = go.Figure()
            
            years = [f'{previous_year}', f'{current_year}']
            for metric, current, previous, change in comparisons:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=years,
                    y=[previous, current],
                    text=[f'{previous}', f'{current}'],
                    textposition='auto',
                ))
            
            fig.update_layout(
                title=f'Сравнение показателей: {previous_year} vs {current_year}',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Детальная таблица сравнения
            st.subheader("Детальное сравнение")
            comparison_df = pd.DataFrame(comparisons, 
                                       columns=['Показатель', f'{current_year}', f'{previous_year}', 'Изменение %'])
            st.dataframe(comparison_df)
        
        else:
            st.warning("Недостаточно данных для сравнения с прошлым годом")
    
    else:
        st.warning("Для сравнения нужны данные с годами")

def show_forecast(df, column_types):
    """Прогнозирование пожаров"""
    st.header("🔮 Прогнозирование количества пожаров")
    
    date_col = next((col for col, type_ in column_types.items() if type_ == 'date'), None)
    
    if date_col:
        forecast_data = create_forecast(df, date_col)
        
        if forecast_data:
            # Подготовка исторических данных
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=['date'])
            
            monthly_historical = df_temp.groupby([df_temp['date'].dt.year, df_temp['date'].dt.month]).size()
            monthly_historical = monthly_historical.reset_index(name='count')
            monthly_historical['period'] = monthly_historical['date'].dt.year * 12 + monthly_historical['date'].dt.month
            monthly_historical['date_str'] = monthly_historical['date'].dt.strftime('%Y-%m')
            
            # Подготовка прогнозных данных
            forecast_df = pd.DataFrame(forecast_data, columns=['period', 'count'])
            forecast_df['year'] = (forecast_df['period'] // 12).astype(int)
            forecast_df['month'] = (forecast_df['period'] % 12).astype(int)
            forecast_df['date_str'] = forecast_df['year'].astype(str) + '-' + forecast_df['month'].astype(str).str.zfill(2)
            forecast_df['type'] = 'Прогноз'
            
            monthly_historical['type'] = 'История'
            
            # Объединение данных
            combined_data = pd.concat([
                monthly_historical[['date_str', 'count', 'type']],
                forecast_df[['date_str', 'count', 'type']]
            ])
            
            fig = px.line(
                combined_data,
                x='date_str',
                y='count',
                color='type',
                title='Прогноз количества пожаров на 6 месяцев',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Статистика прогноза
            st.subheader("Статистика прогноза")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_forecast = forecast_df['count'].mean()
                st.metric("Средний прогноз в месяц", f"{avg_forecast:.1f}")
            
            with col2:
                total_forecast = forecast_df['count'].sum()
                st.metric("Общий прогноз на 6 месяцев", f"{total_forecast:.1f}")
        
        else:
            st.warning("Недостаточно данных для прогнозирования")
    
    else:
        st.warning("Для прогнозирования нужны данные с датами")

if __name__ == "__main__":
    main()
