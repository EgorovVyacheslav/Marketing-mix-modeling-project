import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Попытка импорта Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')


class ForecastModule:
    
    def __init__(self, 
                 time_columns: Dict[str, str],
                 categorical_columns: List[str],
                 metric_columns: List[str],
                 forecast_periods: int = 4):

        self.time_columns = time_columns
        self.categorical_columns = categorical_columns
        self.metric_columns = metric_columns
        self.forecast_periods = forecast_periods
        
        # Определяем минимальную единицу времени
        self.min_time_unit = self._determine_min_time_unit()
        
        # Убрали подробный вывод при инициализации для ускорения работы
    
    def _determine_min_time_unit(self) -> str:
        # Определяет минимальную единицу времени из заданных столбцов
        priority = {'second': 1, 'minute': 2, 'hour': 3, 'day': 4, 'month': 5, 'year': 6}
        min_unit = 'year'
        min_priority = 6
        
        for col, unit in self.time_columns.items():
            if unit in priority and priority[unit] < min_priority:
                min_priority = priority[unit]
                min_unit = unit
        
        return min_unit
    
    def _get_time_column(self, unit: str) -> Optional[str]:
        # Получает название столбца для указанной единицы времени
        for col, col_unit in self.time_columns.items():
            if col_unit == unit:
                return col
        return None
    
    def _create_date_from_columns(self, df: pd.DataFrame) -> pd.Series:
        # Создает дату из временных столбцов
        if len(df) == 0:
            return pd.Series([], dtype='datetime64[ns]')
        
        year_col = self._get_time_column('year')
        month_col = self._get_time_column('month')
        day_col = self._get_time_column('day')
        
        if year_col and year_col in df.columns:
            year = df[year_col].astype(str)
        else:
            year = pd.Series(['2020'] * len(df), index=df.index)
        
        if month_col and month_col in df.columns:
            month = df[month_col].astype(str).str.zfill(2)
        else:
            month = pd.Series(['01'] * len(df), index=df.index)
        
        if day_col and day_col in df.columns:
            day = df[day_col].astype(str).str.zfill(2)
        else:
            day = pd.Series(['01'] * len(df), index=df.index)
        
        date_str = year + '-' + month + '-' + day
        return pd.to_datetime(date_str, errors='coerce')
    
    def _get_next_period_date(self, last_date: pd.Timestamp) -> pd.Timestamp:
        # Получает дату следующего периода в зависимости от минимальной единицы времени
        time_deltas = {
            'year': lambda d: d.replace(year=d.year + 1),
            'month': lambda d: d.replace(year=d.year + 1, month=1, day=1) if d.month == 12 else d.replace(month=d.month + 1, day=1),
            'day': lambda d: d + timedelta(days=1),
            'hour': lambda d: d + timedelta(hours=1),
            'minute': lambda d: d + timedelta(minutes=1),
            'second': lambda d: d + timedelta(seconds=1)
        }
        
        delta_func = time_deltas.get(self.min_time_unit, lambda d: d + timedelta(days=1))
        return delta_func(last_date)
    
    def _extract_time_components(self, date: pd.Timestamp) -> Dict[str, int]:
        # Извлекает компоненты времени из даты
        time_attrs = {
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute',
            'second': 'second'
        }
        
        components = {}
        for col, unit in self.time_columns.items():
            if unit in time_attrs:
                components[col] = getattr(date, time_attrs[unit])
        
        return components
    
    def load_data(self, file_path: str, encoding: str = None) -> pd.DataFrame:
        # Загрузка данных из CSV файла
        encodings = [encoding] if encoding else ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'latin-1', 'iso-8859-1']
        separators = [',', ';']
        
        df = None
        for enc in encodings:
            for sep in separators:
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=sep, on_bad_lines='skip')
                    break
                except (UnicodeDecodeError, Exception):
                    continue
            if df is not None:
                break
        
        # Последняя попытка с автоматическим определением
        if df is None:
            df = pd.read_csv(file_path, encoding=None, engine='python', on_bad_lines='skip')
        
        # Фильтруем только исторические данные
        if 'is_forecast' in df.columns:
            df = df[df['is_forecast'] == False].copy()
        
        # Создаем дату из временных столбцов
        df['_date'] = self._create_date_from_columns(df)
        df = df[df['_date'].notna()].copy()
        
        # Преобразуем метрики в числовой формат
        for metric in self.metric_columns:
            if metric in df.columns:
                df[metric] = pd.to_numeric(df[metric].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
        
        # Заполняем категориальные столбцы
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)
        
        df = df.sort_values('_date').reset_index(drop=True)
        
        return df
    
    def create_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        # Создание признаков для моделирования временного ряда
        df = df.copy()
        
        # Временные признаки
        df['_year'] = df['_date'].dt.year
        df['_month'] = df['_date'].dt.month
        df['_quarter'] = df['_date'].dt.quarter
        df['_day_of_year'] = df['_date'].dt.dayofyear
        
        # Лаговые признаки
        for lag in [1, 2, 3, 6, 12]:
            df[f'_lag_{lag}'] = df[target_col].shift(lag)
        
        # Скользящее среднее
        for window in [3, 6, 12]:
            df[f'_ma_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
        
        # Тренд
        df['_trend'] = range(len(df))
        
        # Заполняем пропуски
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        return df
    
    def train_random_forest(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame], 
                           target_col: str) -> Tuple[Optional[RandomForestRegressor], Optional[StandardScaler], 
                                                     Optional[List[str]], Optional[Dict]]:
        # Обучение модели Random Forest
        if len(train_data) < 3:
            return None, None, None, None
        
        # Создаем признаки
        train_features = self.create_features(train_data, target_col)
        
        if test_data is not None and len(test_data) > 0:
            test_features = self.create_features(test_data, target_col)
        else:
            test_features = None
        
        # Выбираем признаки (исключаем временные и категориальные)
        exclude_cols = ['_date', target_col] + self.categorical_columns + list(self.time_columns.keys())
        feature_cols = [col for col in train_features.columns 
                       if col not in exclude_cols and col.startswith('_')]
        
        if len(feature_cols) == 0:
            return None, None, None, None
        
        X_train = train_features[feature_cols].fillna(0).astype(float)
        y_train = train_features[target_col].fillna(0).astype(float)
        
        # Нормализация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Обучение модели
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Метрики на тестовой выборке
        metrics = None
        if test_features is not None and len(test_features) > 0:
            X_test = test_features[feature_cols].fillna(0).astype(float)
            X_test_scaled = scaler.transform(X_test)
            test_predictions = model.predict(X_test_scaled)
            y_test = test_features[target_col].fillna(0).astype(float).values
            
            # Проверка на валидность данных для r^2
            y_test_var = np.var(y_test)
            if y_test_var == 0 or len(y_test) < 2:
                # Если дисперсия равна нулю или недостаточно данных, используем альтернативный расчет
                r2 = 0.0 if np.allclose(y_test, test_predictions) else -np.inf
            else:
                # Используем встроенный метод score модели Random Forest для более точного расчета r^2
                r2 = model.score(X_test_scaled, y_test)
            
            mae = mean_absolute_error(y_test, test_predictions)
            rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            mape = np.mean(np.abs((y_test - test_predictions) / (y_test + 1e-8))) * 100
            
            metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
        
        return model, scaler, feature_cols, metrics
    
    def train_prophet(self, train_data: pd.DataFrame, test_data: Optional[pd.DataFrame],
                     target_col: str) -> Tuple[Optional[Prophet], Optional[Dict]]:
        # Обучение модели Prophet
        if not PROPHET_AVAILABLE:
            return None, None
        
        if len(train_data) < 3:
            return None, None
        
        # Подготовка данных для Prophet
        prophet_train = pd.DataFrame({
            'ds': train_data['_date'],
            'y': train_data[target_col].fillna(0)
        })
        
        # Обучение модели
        model = Prophet()
        model.fit(prophet_train)
        
        # Метрики на тестовой выборке
        metrics = None
        if test_data is not None and len(test_data) > 0:
            prophet_test = pd.DataFrame({
                'ds': test_data['_date'],
                'y': test_data[target_col].fillna(0)
            })
            
            forecast = model.predict(prophet_test[['ds']])
            y_test = prophet_test['y'].values
            y_pred = forecast['yhat'].values
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            
            metrics = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
        
        return model, metrics
    
    def forecast_random_forest(self, model: RandomForestRegressor, scaler: StandardScaler,
                              feature_cols: List[str], data: pd.DataFrame, 
                              target_col: str) -> pd.DataFrame:
        # Прогноз с помощью Random Forest
        if model is None:
            return None
        
        forecast_dates = []
        forecast_values = []
        
        current_data = data.copy().reset_index(drop=True)
        last_date = current_data['_date'].max()
        
        for i in range(self.forecast_periods):
            # Следующий период
            next_date = self._get_next_period_date(last_date)
            
            # Создаем временную строку
            temp_row = pd.DataFrame({
                '_date': [next_date],
                target_col: [current_data[target_col].iloc[-1] if len(current_data) > 0 else 0]
            })
            
            # Добавляем категориальные столбцы (берем из последней строки)
            for col in self.categorical_columns:
                if col in current_data.columns:
                    temp_row[col] = current_data[col].iloc[-1] if len(current_data) > 0 else ''
            
            temp_data = pd.concat([current_data, temp_row], ignore_index=True)
            temp_features = self.create_features(temp_data, target_col)
            
            # Берем последнюю строку
            X_forecast = temp_features.iloc[[-1]][feature_cols].fillna(0).astype(float)
            X_forecast_scaled = scaler.transform(X_forecast)
            
            # Прогноз
            pred_value = model.predict(X_forecast_scaled)[0]
            pred_value = max(0, pred_value)
            
            forecast_dates.append(next_date)
            forecast_values.append(pred_value)
            
            # Обновляем данные
            temp_row[target_col] = pred_value
            current_data = pd.concat([current_data, temp_row], ignore_index=True)
            last_date = next_date
        
        forecast_df = pd.DataFrame({
            '_date': forecast_dates,
            target_col: forecast_values
        })
        
        return forecast_df
    
    def forecast_prophet(self, model: Prophet, data: pd.DataFrame, 
                       target_col: str) -> pd.DataFrame:
        # Прогноз с помощью Prophet
        if model is None or not PROPHET_AVAILABLE:
            return None
        
        last_date = data['_date'].max()
        
        # Создаем даты для прогноза
        future_dates = []
        current_date = last_date
        for i in range(self.forecast_periods):
            current_date = self._get_next_period_date(current_date)
            future_dates.append(current_date)
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        forecast_df = pd.DataFrame({
            '_date': forecast['ds'],
            target_col: np.maximum(0, forecast['yhat'].values)
        })
        
        return forecast_df
    
    def forecast_metric(self, df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        # Прогнозирование одной метрики по всем срезам
        
        # Агрегируем данные по категориальным столбцам и дате
        group_cols = self.categorical_columns + ['_date']
        aggregated = df.groupby(group_cols)[metric_col].sum().reset_index()
        
        # Получаем уникальные комбинации категориальных столбцов
        if len(self.categorical_columns) > 0:
            unique_combinations = aggregated[self.categorical_columns].drop_duplicates()
        else:
            # Если нет категориальных столбцов, создаем одну "пустую" комбинацию
            unique_combinations = pd.DataFrame({'_dummy': [1]})
        
        all_forecasts = []
        all_metrics = []  # Список метрик для каждого среза
        total_combinations = len(unique_combinations)
        
        print(f"Всего срезов для обработки: {total_combinations}")
        
        for idx, (_, combo_row) in enumerate(unique_combinations.iterrows()):
            # Фильтруем данные для текущей комбинации
            if len(self.categorical_columns) > 0:
                filter_conditions = pd.Series([True] * len(aggregated))
                for col in self.categorical_columns:
                    filter_conditions = filter_conditions & (aggregated[col] == combo_row[col])
                group_data = aggregated[filter_conditions].copy().sort_values('_date').reset_index(drop=True)
            else:
                # Если нет категориальных столбцов, используем все данные
                group_data = aggregated.copy().sort_values('_date').reset_index(drop=True)
            
            if len(group_data) < 6:
                continue
            
            # Разделяем на train и test
            split_idx = max(int(len(group_data) * 0.8), len(group_data) - 3)
            train_group = group_data.iloc[:split_idx].copy()
            test_group = group_data.iloc[split_idx:].copy()
            
            # Обучаем обе модели
            rf_model, rf_scaler, rf_features, rf_metrics = self.train_random_forest(
                train_group, test_group, metric_col
            )
            prophet_model, prophet_metrics = self.train_prophet(
                train_group, test_group, metric_col
            )
            
            # Выбираем лучшую модель по MAPE
            best_model = None
            best_forecast = None
            best_metrics = None
            
            if rf_metrics and prophet_metrics:
                if rf_metrics['MAPE'] < prophet_metrics['MAPE']:
                    best_model = 'Random Forest'
                    best_forecast = self.forecast_random_forest(
                        rf_model, rf_scaler, rf_features, group_data, metric_col
                    )
                    best_metrics = rf_metrics
                else:
                    best_model = 'Prophet'
                    best_forecast = self.forecast_prophet(prophet_model, group_data, metric_col)
                    best_metrics = prophet_metrics
            elif rf_metrics:
                best_model = 'Random Forest'
                best_forecast = self.forecast_random_forest(
                    rf_model, rf_scaler, rf_features, group_data, metric_col
                )
                best_metrics = rf_metrics
            elif prophet_metrics:
                best_model = 'Prophet'
                best_forecast = self.forecast_prophet(prophet_model, group_data, metric_col)
                best_metrics = prophet_metrics
            
            if best_forecast is not None and len(best_forecast) > 0:
                # Добавляем категориальные столбцы
                if len(self.categorical_columns) > 0:
                    for col in self.categorical_columns:
                        best_forecast[col] = combo_row[col]
                
                all_forecasts.append(best_forecast)
                
                # Сохраняем метрики для этого среза
                if best_metrics:
                    slice_metrics = best_metrics.copy()
                    slice_metrics['model'] = best_model
                    slice_metrics['metric'] = metric_col
                    # Добавляем информацию о срезе
                    if len(self.categorical_columns) > 0:
                        for col in self.categorical_columns:
                            slice_metrics[f'slice_{col}'] = combo_row[col]
                    all_metrics.append(slice_metrics)
            
            # Выводим прогресс каждые 100 срезов или на последнем срезе
            current_progress = idx + 1
            if current_progress % 100 == 0 or current_progress == total_combinations:
                print(f"  Прогресс: {current_progress}/{total_combinations} ({current_progress * 100 // total_combinations}%)")
        
        if len(all_forecasts) == 0:
            return pd.DataFrame(), []
        
        # Объединяем все прогнозы
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        return forecast_df, all_metrics
    
    def forecast(self, file_path: str, output_path: Optional[str] = None, encoding: str = None) -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
        # Основной метод для прогнозирования всех метрик
        print("="*70)
        print("НАЧАЛО ПРОГНОЗИРОВАНИЯ")
        print("="*70)
        
        # Загрузка данных
        print("Загрузка данных...")
        df = self.load_data(file_path, encoding=encoding)
        print(f"Загружено строк: {len(df)}, период: {df['_date'].min()} - {df['_date'].max()}")
        
        # Прогнозируем каждую метрику отдельно
        all_forecasts = {}
        all_metrics_by_metric = {}  # Метрики по метрикам
        
        for metric_idx, metric_col in enumerate(self.metric_columns):
            if metric_col not in df.columns:
                print(f"Предупреждение: метрика {metric_col} не найдена в данных")
                continue
            
            print(f"\nПрогноз для метрики \"{metric_col}\" ({metric_idx + 1}/{len(self.metric_columns)})")
            forecast_df, metrics_list = self.forecast_metric(df, metric_col)
            if len(forecast_df) > 0:
                all_forecasts[metric_col] = forecast_df
                all_metrics_by_metric[metric_col] = metrics_list
        
        if len(all_forecasts) == 0:
            print("Ошибка: не удалось создать прогнозы")
            return pd.DataFrame(), {}
        
        # Объединяем прогнозы всех метрик
        print("\nОбъединение прогнозов всех метрик...")
        
        # Создаем список всех уникальных комбинаций (дата + категориальные столбцы)
        merge_cols = ['_date'] + self.categorical_columns
        
        # Собираем все уникальные комбинации из всех прогнозов
        all_keys = set()
        for forecast_df in all_forecasts.values():
            for _, row in forecast_df.iterrows():
                key = tuple[Any, ...](row[col] for col in merge_cols)
                all_keys.add(key)
        
        # Создаем базовый DataFrame со всеми комбинациями
        result_rows = []
        for key in all_keys:
            row = {}
            for i, col in enumerate(merge_cols):
                row[col] = key[i]
            result_rows.append(row)
        
        result_df = pd.DataFrame(result_rows)
        
        # Добавляем прогнозы каждой метрики
        for metric_col, forecast_df in all_forecasts.items():
            # Объединяем по ключевым столбцам
            result_df = result_df.merge(
                forecast_df[merge_cols + [metric_col]],
                on=merge_cols,
                how='left'
            )
        
        # Заполняем пропуски нулями
        for metric_col in self.metric_columns:
            if metric_col in result_df.columns:
                result_df[metric_col] = result_df[metric_col].fillna(0)
        
        # Заполняем временные компоненты
        time_attrs = {
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute',
            'second': 'second'
        }
        
        for col, unit in self.time_columns.items():
            if unit in time_attrs:
                result_df[col] = getattr(result_df['_date'].dt, time_attrs[unit])
        
        # Удаляем служебный столбец _date
        if '_date' in result_df.columns:
            result_df = result_df.drop(columns=['_date'])
        
        # Определяем максимальную историческую дату из исходных данных
        # df уже имеет столбец _date (создан в load_data)
        max_historical_date = df['_date'].max()
        
        # Добавляем маркер прогноза только для дат после максимальной исторической даты
        # Создаем временный столбец _date для проверки дат в result_df
        result_df_temp = result_df.copy()
        # Восстанавливаем _date из временных столбцов для проверки
        if 'year' in result_df_temp.columns and 'month' in result_df_temp.columns:
            result_df_temp['_date'] = pd.to_datetime(
                result_df_temp['year'].astype(str) + '-' + 
                result_df_temp['month'].astype(str).str.zfill(2) + '-01'
            )
        else:
            # Если нет year/month, используем текущую логику
            result_df_temp['_date'] = pd.to_datetime('2026-01-01')  # Fallback
        
        # Устанавливаем is_forecast = True только для дат после максимальной исторической даты
        result_df['is_forecast'] = result_df_temp['_date'] > max_historical_date
        
        # Объединяем с историческими данными
        historical_df = df.copy()
        # Удаляем служебный столбец _date из исторических данных
        if '_date' in historical_df.columns:
            historical_df = historical_df.drop(columns=['_date'])
        historical_df['is_forecast'] = False
        
        # Убеждаемся, что все колонки присутствуют
        for col in historical_df.columns:
            if col not in result_df.columns:
                result_df[col] = ''
        
        # Упорядочиваем колонки как в исторических данных
        result_cols = [col for col in historical_df.columns if col in result_df.columns]
        result_cols += [col for col in result_df.columns if col not in historical_df.columns]
        result_df = result_df[result_cols]
        
        # Объединяем исторические данные и прогнозы
        final_df = pd.concat([historical_df, result_df], ignore_index=True)
        
        # Сортируем по временным столбцам и категориальным
        sort_cols = list(self.time_columns.keys()) + self.categorical_columns
        sort_cols = [col for col in sort_cols if col in final_df.columns]
        final_df = final_df.sort_values(sort_cols).reset_index(drop=True)
        
        # Сохранение результата
        if output_path is None:
            output_path = 'forecast_result.csv'
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*70}")
        print("ПРОГНОЗИРОВАНИЕ ЗАВЕРШЕНО")
        print(f"{'='*70}")
        print(f"Результаты сохранены: {output_path}")
        print(f"Исторических записей: {len(historical_df)}")
        print(f"Прогнозных записей: {len(result_df)}")
        print(f"Всего записей: {len(final_df)}")
        
        return final_df, all_metrics_by_metric

