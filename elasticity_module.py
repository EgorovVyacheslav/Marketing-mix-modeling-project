

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize_scalar
import warnings

warnings.filterwarnings('ignore')


class ElasticityCalculator:
    
    def __init__(self):
        pass
    
    def calculate_elasticity(self, df: pd.DataFrame, x_metric: str, y_metric: str) -> Dict:
        # Проверяем наличие метрик
        if x_metric not in df.columns or y_metric not in df.columns:
            return {
                'success': False,
                'error': f'Метрики {x_metric} или {y_metric} не найдены в данных'
            }
        
        # Преобразуем метрики в числовой формат
        df = df.copy()
        df[x_metric] = pd.to_numeric(
            df[x_metric].astype(str).str.replace(',', '').str.replace(' ', ''), 
            errors='coerce'
        )
        df[y_metric] = pd.to_numeric(
            df[y_metric].astype(str).str.replace(',', '').str.replace(' ', ''), 
            errors='coerce'
        )
        
        # Удаляем строки с нулевыми или отрицательными значениями (для логарифма)
        df = df[(df[x_metric] > 0) & (df[y_metric] > 0)].copy()
        
        if len(df) < 3:
            return {
                'success': False,
                'error': 'Недостаточно данных для расчета (нужно минимум 3 точки)'
            }
        
        # Сортируем по дате, если есть
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        x_values = df[x_metric].values
        y_values = df[y_metric].values
        
        # Удаляем бесконечные и NaN значения
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values) & (x_values > 0) & (y_values > 0)
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]
        
        if len(x_values) < 3:
            return {
                'success': False,
                'error': 'Недостаточно валидных данных после фильтрации'
            }
        
        # Рассчитываем оба метода
        result_without_baseline = self._calculate_without_baseline(x_values, y_values)
        result_with_baseline = self._calculate_with_baseline(x_values, y_values)
        
        # Выбираем лучший метод по R^2
        results = [
            (result_without_baseline, 'without_baseline'),
            (result_with_baseline, 'with_baseline')
        ]
        
        valid_results = [(r, m) for r, m in results if r.get('success')]
        
        if not valid_results:
            return {
                'success': False,
                'error': 'Не удалось рассчитать эластичность ни одним из методов'
            }
        
        best_result, best_method = max(valid_results, key=lambda x: x[0].get('r2', -np.inf))
        best_result['method'] = best_method
        
        # Добавляем интерпретацию
        best_result['interpretation'] = self._get_elasticity_interpretation(
            best_result['elasticity'], 
            best_result.get('baseline', None)
        )
        
        return best_result
    
    def _calculate_without_baseline(self, x_values: np.ndarray, y_values: np.ndarray) -> Dict:
        """
        Расчет эластичности без baseline: log(y) = a + b*log(x)
        где b - коэффициент эластичности
        """
        try:
            log_x = np.log(x_values)
            log_y = np.log(y_values)
            
            # Удаляем бесконечные значения
            valid_mask = np.isfinite(log_x) & np.isfinite(log_y)
            log_x = log_x[valid_mask]
            log_y = log_y[valid_mask]
            
            if len(log_x) < 3:
                return {'success': False, 'error': 'Недостаточно валидных данных'}
            
            # Обучаем модель линейной регрессии
            X = log_x.reshape(-1, 1)
            y = log_y
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Коэффициент эластичности - это коэффициент регрессии
            elasticity = model.coef_[0]
            intercept = model.intercept_
            
            # Предсказания
            y_pred = model.predict(X)
            
            # Метрики качества модели
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
            
            return {
                'success': True,
                'elasticity': float(elasticity),
                'intercept': float(intercept),
                'baseline': None,
                'r2': float(r2),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'n_points': len(log_x)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка при расчете без baseline: {str(e)}'
            }
    
    def _calculate_with_baseline(self, x_values: np.ndarray, y_values: np.ndarray) -> Dict:
        try:
            # Определяем диапазон для поиска baseline
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            baseline_min = y_min * 0.01  # 1% от минимума данных
            baseline_max = y_min * 0.99  # 99% от минимума данных (но меньше минимума)
            
            if baseline_max <= baseline_min:
                return {'success': False, 'error': 'Некорректный диапазон для baseline'}
            
            # Поиск оптимального baseline методом золотого сечения
            def objective(baseline):
                try:
                    # Вычисляем y - baseline
                    y_adjusted = y_values - baseline
                    
                    # Проверяем, что все значения положительны
                    if np.any(y_adjusted <= 0):
                        return -np.inf
                    
                    log_x = np.log(x_values)
                    log_y_adjusted = np.log(y_adjusted)
                    
                    # Удаляем бесконечные значения
                    valid_mask = np.isfinite(log_x) & np.isfinite(log_y_adjusted)
                    log_x_valid = log_x[valid_mask]
                    log_y_valid = log_y_adjusted[valid_mask]
                    
                    if len(log_x_valid) < 3:
                        return -np.inf
                    
                    # Обучаем модель
                    X = log_x_valid.reshape(-1, 1)
                    y = log_y_valid
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Предсказания
                    y_pred = model.predict(X)
                    
                    # R^2
                    r2 = r2_score(y, y_pred)
                    return -r2  # Минимизируем отрицательный R^2 (максимизируем R^2)
                except:
                    return -np.inf
            
            # Поиск оптимального baseline
            result_opt = minimize_scalar(
                objective,
                bounds=(baseline_min, baseline_max),
                method='bounded'
            )
            
            if not result_opt.success:
                return {'success': False, 'error': 'Не удалось найти оптимальный baseline'}
            
            optimal_baseline = result_opt.x
            
            # Рассчитываем финальную модель с оптимальным baseline
            y_adjusted = y_values - optimal_baseline
            
            if np.any(y_adjusted <= 0):
                return {'success': False, 'error': 'Некорректные значения после вычитания baseline'}
            
            log_x = np.log(x_values)
            log_y_adjusted = np.log(y_adjusted)
            
            # Удаляем бесконечные значения
            valid_mask = np.isfinite(log_x) & np.isfinite(log_y_adjusted)
            log_x = log_x[valid_mask]
            log_y_adjusted = log_y_adjusted[valid_mask]
            
            if len(log_x) < 3:
                return {'success': False, 'error': 'Недостаточно валидных данных'}
            
            # Обучаем модель
            X = log_x.reshape(-1, 1)
            y = log_y_adjusted
            
            model = LinearRegression()
            model.fit(X, y)
            
            elasticity = model.coef_[0]
            intercept = model.intercept_
            
            # Предсказания
            y_pred = model.predict(X)
            
            # Метрики качества модели
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
            
            return {
                'success': True,
                'elasticity': float(elasticity),
                'intercept': float(intercept),
                'baseline': float(optimal_baseline),
                'r2': float(r2),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'n_points': len(log_x)
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Ошибка при расчете с baseline: {str(e)}'
            }
    
    def _get_elasticity_interpretation(self, elasticity: float, baseline: Optional[float] = None) -> str:
        if baseline is not None:
            return f"Изменение X на 1% приводит к изменению Y на {abs(elasticity):.2f}% (модель с baseline={baseline:.2f})"
        else:
            return f"Изменение X на 1% приводит к изменению Y на {abs(elasticity):.2f}%"
    

