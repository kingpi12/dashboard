import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from pathlib import Path
import numpy as np    
from datetime import datetime
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Дашборд техногенных пожаров",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка и подготовка данных
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    
    # Преобразование дат
    date_columns = ['Дата возникновения', 'Время обнаружения, час. мин.']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Извлечение года и месяца
    if 'Дата возникновения' in df.columns:
        df['Год'] = df['Дата возникновения'].dt.year
        df['Месяц'] = df['Дата возникновения'].dt.month
        df['Месяц_название'] = df['Дата возникновения'].dt.month_name()
    
    # Заполнение пропущенных значений
    numeric_columns = ['Погибло людей: Всего', 'в  т.ч. погибло детей', 
                      'Получили травмы: Всего', 'в  т.ч. получили травмы: детей',
                      'Спасено на пожаре людей', 'Эвакуировано на пожаре людей']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def create_forecast_model(df):
    """Создание модели прогнозирования"""
    try:
        # Подготовка данных для прогнозирования
        monthly_data = df.groupby(['Год', 'Месяц']).size().reset_index(name='Количество_пожаров')
        monthly_data['period'] = monthly_data['Год'] * 12 + monthly_data['Месяц']
        
        if len(monthly_data) < 6:
            return None, "Недостаточно данных для прогнозирования"
        
        # Создание признаков
        X = monthly_data[['period']]
        y = monthly_data['Количество_пожаров']
        
        # Обучение модели
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Прогноз на следующие 6 месяцев
        last_period = monthly_data['period'].max()
        future_periods = [last_period + i for i in range(1, 7)]
        
        future_predictions = model.predict(pd.DataFrame(future_periods, columns=['period']))
        
        return list(zip(future_periods, future_predictions)), "Успешно"
    
    except Exception as e:
        return None, f"Ошибка прогнозирования: {str(e)}"

# Основное приложение
def main():
    st.title("🔥 Аналитический дашборд техногенных пожаров")
    
    # Загрузка файла
    uploaded_file = st.sidebar.file_uploader("Загрузите файл Excel с данными о пожарах", 
                                           type=['xlsx', 'xls'])
    
    if not uploaded_file:
        st.info("👆 Пожалуйста, загрузите Excel файл с данными о пожарах")
        return
    
    # Загрузка данных
    df = load_data(uploaded_file)
    
    # Сайдбар с фильтрами
    st.sidebar.header("Фильтры данных")
    
    # Фильтр по годам
    if 'Год' in df.columns:
        years = sorted(df['Год'].unique())
        selected_years = st.sidebar.multiselect(
            "Выберите годы",
            options=years,
            default=years
        )
        df = df[df['Год'].isin(selected_years)]
    
    # Фильтр по районам
    if 'Муниципальный район' in df.columns:
        districts = ['Все'] + list(df['Муниципальный район'].unique())
        selected_district = st.sidebar.selectbox(
            "Выберите район",
            options=districts
        )
        if selected_district != 'Все':
            df = df[df['Муниципальный район'] == selected_district]
    
    # Основные метрики
    st.header("📊 Ключевые показатели")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_fires = len(df)
        st.metric("Общее количество пожаров", total_fires)
    
    with col2:
        total_deaths = df['Погибло людей: Всего'].sum() if 'Погибло людей: Всего' in df.columns else 0
        st.metric("Погибло людей", int(total_deaths))
    
    with col3:
        total_injured = df['Получили травмы: Всего'].sum() if 'Получили травмы: Всего' in df.columns else 0
        st.metric("Травмировано людей", int(total_injured))
    
    with col4:
        if 'Год' in df.columns:
            current_year = df['Год'].max()
            prev_year = current_year - 1
            current_year_fires = len(df[df['Год'] == current_year])
            prev_year_fires = len(df[df['Год'] == prev_year]) if prev_year in df['Год'].values else 0
            
            change = current_year_fires - prev_year_fires if prev_year_fires > 0 else 0
            st.metric(f"Пожаров в {current_year}", current_year_fires, 
                     delta=f"{change:+}" if prev_year_fires > 0 else None)
    
    # Вкладки для разных анализов
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Общая динамика", 
        "🗺️ Распределение по районам", 
        "🔍 Причины пожаров",
        "🏢 Места возникновения",
        "📅 Сезонность",
        "📊 Динамика по районам",
        "🔮 Прогнозирование"
    ])
    
    with tab1:
        st.subheader("1. Общая динамика количества пожаров по годам")
        
        if 'Год' in df.columns:
            yearly_data = df.groupby('Год').size().reset_index(name='Количество пожаров')
            
            fig = px.line(
                yearly_data, 
                x='Год', 
                y='Количество пожаров',
                title='Динамика количества пожаров по годам',
                markers=True
            )
            fig.update_traces(line=dict(width=4), marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("В данных отсутствует информация о годе")
    
    with tab2:
        st.subheader("2. Распределение пожаров по районам")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Муниципальный район' in df.columns:
                district_counts = df['Муниципальный район'].value_counts().reset_index()
                district_counts.columns = ['Район', 'Количество пожаров']
                
                fig = px.bar(
                    district_counts.head(10),
                    x='Количество пожаров',
                    y='Район',
                    orientation='h',
                    title='Топ-10 районов по количеству пожаров'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Простая карта (заглушка - в реальном проекте нужны координаты)
            st.info("🗺️ Для отображения на карте необходимы координаты в столбце 'Геоточка'")
            
            if 'Геоточка' in df.columns and df['Геоточка'].notna().any():
                # Здесь можно добавить отображение на карте с использованием folium
                try:
                    # Пример простой карты
                    m = folium.Map(location=[55.7558, 37.6173], zoom_start=10)
                    
                    # Добавление маркеров (упрощенный пример)
                    for idx, row in df.dropna(subset=['Геоточка']).head(100).iterrows():
                        # Парсинг координат из геоточки
                        # Это упрощенный пример - в реальности нужен парсинг вашего формата координат
                        folium.CircleMarker(
                            location=[55.7558, 37.6173],  # Заглушка
                            radius=5,
                            popup=f"Пожар: {row.get('Наименование объекта', 'N/A')}",
                            color='red',
                            fill=True
                        ).add_to(m)
                    
                    st_folium(m, width=700, height=400)
                except:
                    st.warning("Не удалось отобразить карту. Проверьте формат координат.")
    
    with tab3:
        st.subheader("3. Основные причины возникновения пожаров")
        
        if 'Причина пожара' in df.columns:
            cause_counts = df['Причина пожара'].value_counts().reset_index()
            cause_counts.columns = ['Причина', 'Количество']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    cause_counts.head(10),
                    values='Количество',
                    names='Причина',
                    title='Распределение по причинам пожаров (Топ-10)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    cause_counts.head(15),
                    x='Количество',
                    y='Причина',
                    orientation='h',
                    title='Основные причины пожаров'
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("В данных отсутствует информация о причинах пожаров")
    
    with tab4:
        st.subheader("4. Наиболее частые места возникновения пожаров")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Объект пожара (загорания)' in df.columns:
                object_counts = df['Объект пожара (загорания)'].value_counts().head(15).reset_index()
                object_counts.columns = ['Объект', 'Количество']
                
                fig = px.bar(
                    object_counts,
                    x='Количество',
                    y='Объект',
                    orientation='h',
                    title='Типы объектов, где происходят пожары'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Этажность здания' in df.columns:
                floor_data = df['Этажность здания'].value_counts().head(10).reset_index()
                floor_data.columns = ['Этажность', 'Количество']
                
                fig = px.bar(
                    floor_data,
                    x='Этажность',
                    y='Количество',
                    title='Распределение по этажности зданий'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("5. Сезонность пожаров")
        
        if 'Месяц' in df.columns and 'Месяц_название' in df.columns:
            monthly_data = df.groupby(['Месяц', 'Месяц_название']).size().reset_index(name='Количество')
            monthly_data = monthly_data.sort_values('Месяц')
            
            fig = px.line(
                monthly_data,
                x='Месяц_название',
                y='Количество',
                title='Распределение пожаров по месяцам'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Тепловая карта по годам и месяцам
            if 'Год' in df.columns:
                heatmap_data = df.groupby(['Год', 'Месяц']).size().reset_index(name='Количество')
                heatmap_pivot = heatmap_data.pivot(index='Год', columns='Месяц', values='Количество')
                
                fig = px.imshow(
                    heatmap_pivot,
                    title='Тепловая карта пожаров по годам и месяцам',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.subheader("6. Динамика основных показателей по районам")
        
        if 'Муниципальный район' in df.columns and 'Год' in df.columns:
            # Выбор показателя для анализа
            metric_options = {
                'Количество пожаров': 'count',
                'Погибло людей': 'Погибло людей: Всего',
                'Травмировано людей': 'Получили травмы: Всего'
            }
            
            selected_metric = st.selectbox(
                "Выберите показатель для анализа",
                options=list(metric_options.keys())
            )
            
            if selected_metric == 'Количество пожаров':
                district_year_data = df.groupby(['Муниципальный район', 'Год']).size().reset_index(name='Значение')
            else:
                metric_column = metric_options[selected_metric]
                if metric_column in df.columns:
                    district_year_data = df.groupby(['Муниципальный район', 'Год'])[metric_column].sum().reset_index()
                    district_year_data.columns = ['Муниципальный район', 'Год', 'Значение']
                else:
                    st.warning(f"В данных отсутствует столбец {metric_column}")
                    district_year_data = pd.DataFrame()
            
            if not district_year_data.empty:
                # Топ-5 районов по последнему году
                last_year = district_year_data['Год'].max()
                top_districts = district_year_data[
                    district_year_data['Год'] == last_year
                ].nlargest(5, 'Значение')['Муниципальный район'].tolist()
                
                filtered_data = district_year_data[
                    district_year_data['Муниципальный район'].isin(top_districts)
                ]
                
                fig = px.line(
                    filtered_data,
                    x='Год',
                    y='Значение',
                    color='Муниципальный район',
                    title=f'Динамика {selected_metric.lower()} по топ-5 районам',
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab7:
        st.subheader("7. Прогнозирование количества пожаров")
        
        if 'Год' in df.columns and 'Месяц' in df.columns:
            forecast_data, forecast_status = create_forecast_model(df)
            
            if forecast_data:
                # Подготовка исторических данных
                historical_monthly = df.groupby(['Год', 'Месяц']).size().reset_index(name='Количество')
                historical_monthly['period'] = historical_monthly['Год'] * 12 + historical_monthly['Месяц']
                historical_monthly['date'] = pd.to_datetime(
                    historical_monthly['Год'].astype(str) + '-' + historical_monthly['Месяц'].astype(str) + '-01'
                )
                
                # Подготовка прогнозных данных
                forecast_df = pd.DataFrame(forecast_data, columns=['period', 'Количество'])
                forecast_df['Год'] = (forecast_df['period'] // 12).astype(int)
                forecast_df['Месяц'] = (forecast_df['period'] % 12).astype(int)
                forecast_df['date'] = pd.to_datetime(
                    forecast_df['Год'].astype(str) + '-' + forecast_df['Месяц'].astype(str) + '-01'
                )
                forecast_df['type'] = 'Прогноз'
                
                historical_monthly['type'] = 'История'
                
                # Объединение данных
                combined_data = pd.concat([
                    historical_monthly[['date', 'Количество', 'type']],
                    forecast_df[['date', 'Количество', 'type']]
                ])
                
                fig = px.line(
                    combined_data,
                    x='date',
                    y='Количество',
                    color='type',
                    title='Прогноз количества пожаров на следующие 6 месяцев',
                    markers=True
                )
                
                # Добавление вертикальной линии разделяющей историю и прогноз
                last_historical_date = historical_monthly['date'].max()
                fig.add_vline(
                    x=last_historical_date.timestamp() * 1000,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Начало прогноза"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Статистика прогноза
                st.subheader("Статистика прогноза")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_forecast = forecast_df['Количество'].mean()
                    st.metric("Средний прогноз в месяц", f"{avg_forecast:.1f}")
                
                with col2:
                    total_forecast = forecast_df['Количество'].sum()
                    st.metric("Общий прогноз на 6 месяцев", f"{total_forecast:.1f}")
                
                with col3:
                    last_historical = historical_monthly['Количество'].tail(6).mean()
                    change = ((avg_forecast - last_historical) / last_historical * 100) if last_historical > 0 else 0
                    st.metric("Изменение к среднему за последние 6 мес.", f"{change:+.1f}%")
            
            else:
                st.warning(forecast_status)
        
        # Сравнение с АППГ
        st.subheader("Сравнение с аналогичным периодом прошлого года (АППГ)")
        
        if 'Год' in df.columns:
            current_year = df['Год'].max()
            previous_year = current_year - 1
            
            if previous_year in df['Год'].values:
                current_data = df[df['Год'] == current_year]
                previous_data = df[df['Год'] == previous_year]
                
                comparison_metrics = []
                
                # Количество пожаров
                current_fires = len(current_data)
                previous_fires = len(previous_data)
                fires_change = ((current_fires - previous_fires) / previous_fires * 100) if previous_fires > 0 else 0
                comparison_metrics.append(('Количество пожаров', current_fires, previous_fires, fires_change))
                
                # Погибшие
                if 'Погибло людей: Всего' in df.columns:
                    current_deaths = current_data['Погибло людей: Всего'].sum()
                    previous_deaths = previous_data['Погибло людей: Всего'].sum()
                    deaths_change = ((current_deaths - previous_deaths) / previous_deaths * 100) if previous_deaths > 0 else 0
                    comparison_metrics.append(('Погибло людей', current_deaths, previous_deaths, deaths_change))
                
                # Травмированные
                if 'Получили травмы: Всего' in df.columns:
                    current_injured = current_data['Получили травмы: Всего'].sum()
                    previous_injured = previous_data['Получили травмы: Всего'].sum()
                    injured_change = ((current_injured - previous_injured) / previous_injured * 100) if previous_injured > 0 else 0
                    comparison_metrics.append(('Травмировано людей', current_injured, previous_injured, injured_change))
                
                # Отображение сравнения
                for metric, current, previous, change in comparison_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{metric} ({current_year})", current)
                    with col2:
                        st.metric(f"{metric} ({previous_year})", previous)
                    with col3:
                        st.metric("Изменение", f"{change:+.1f}%")
            else:
                st.warning("Недостаточно данных для сравнения с прошлым годом")

    # Дополнительная информация
    st.sidebar.header("О данных")
    st.sidebar.info(f"""
    **Загружено записей:** {len(df)}
    **Период данных:** {df['Год'].min() if 'Год' in df.columns else 'N/A'} - {df['Год'].max() if 'Год' in df.columns else 'N/A'}
    **Районов:** {df['Муниципальный район'].nunique() if 'Муниципальный район' in df.columns else 'N/A'}
    """)

if __name__ == "__main__":
    main()

