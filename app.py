import os
import json
import uuid
import traceback
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename

from forecast_module import ForecastModule
from elasticity_module import ElasticityCalculator

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}
ENCODINGS = ['utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'latin-1', 'iso-8859-1']
SEPARATORS = [',', ';']


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_csv_with_encoding(filepath: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    for encoding in ENCODINGS:
        for sep in SEPARATORS:
            try:
                df = pd.read_csv(filepath, encoding=encoding, sep=sep, nrows=nrows, on_bad_lines='skip')
                return df, encoding
            except (UnicodeDecodeError, Exception):
                continue
    
    # Последняя попытка с автоматическим определением
    try:
        df = pd.read_csv(filepath, encoding=None, engine='python', nrows=nrows, on_bad_lines='skip')
        return df, 'utf-8'
    except Exception as e:
        raise ValueError(f'Не удалось прочитать файл: {str(e)}')


def clean_dataframe_for_json(df: pd.DataFrame, n_rows: Optional[int] = None) -> List[Dict]:
    if len(df) == 0:
        return []
    
    if n_rows:
        df = df.head(n_rows) if n_rows > 0 else df.tail(abs(n_rows))
    
    df = df.fillna('')
    records = df.to_dict('records')
    
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    
    return records


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    column_types = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        column_types[col] = 'numeric' if 'int' in dtype or 'float' in dtype else 'categorical'
    return column_types


def create_date_column(df: pd.DataFrame) -> pd.Series:
    if 'year' in df.columns and 'month' in df.columns:
        return pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
        )
    return pd.Series()


def get_forecast_filepath(session_id: str) -> str:
    return os.path.join('results', f'forecast_{session_id}.csv')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Файл не выбран'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Недопустимый формат файла'}), 400
    
    try:
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Не удалось сохранить файл на сервере'}), 500
        
        df, encoding = read_csv_with_encoding(filepath, nrows=100)
        
        if len(df.columns) == 0:
            return jsonify({'error': 'Файл не содержит столбцов'}), 400
        
        session['filepath'] = filepath
        session['filename'] = filename
        session['session_id'] = session_id
        session['encoding'] = encoding
        
        return jsonify({
            'success': True,
            'columns': list(df.columns),
            'column_types': detect_column_types(df),
            'preview': clean_dataframe_for_json(df.head(10)),
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        filepath = session.get('filepath') or data.get('filepath')
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({'error': 'Файл не найден. Пожалуйста, загрузите файл заново.'}), 400
        
        time_columns = data.get('time_columns', {})
        categorical_columns = data.get('categorical_columns', [])
        metric_columns = data.get('metric_columns', [])
        forecast_periods = int(data.get('forecast_periods', 4))
        
        if not time_columns:
            return jsonify({'error': 'Необходимо указать хотя бы один временной столбец'}), 400
        
        if not metric_columns:
            return jsonify({'error': 'Необходимо выбрать хотя бы одну целевую метрику'}), 400
        
        forecast_module = ForecastModule(
            time_columns=time_columns,
            categorical_columns=categorical_columns,
            metric_columns=metric_columns,
            forecast_periods=forecast_periods
        )
        
        session_id = session.get('session_id', str(uuid.uuid4()))
        output_path = get_forecast_filepath(session_id)
        encoding = session.get('encoding', 'utf-8')
        
        result_df, metrics_by_metric = forecast_module.forecast(
            file_path=filepath,
            output_path=output_path,
            encoding=encoding
        )
        
        if len(result_df) == 0:
            return jsonify({'error': 'Не удалось создать прогноз. Проверьте данные и настройки.'}), 400
        
        # Фильтрация столбцов
        columns_to_keep = list(time_columns.keys()) + categorical_columns + metric_columns + ['is_forecast']
        existing_columns = [col for col in columns_to_keep if col in result_df.columns]
        result_df = result_df[existing_columns].copy()
        
        # Сохранение результата
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # Разделение исторических данных и прогнозов
        historical_df = result_df[result_df['is_forecast'] == False]
        forecast_df = result_df[result_df['is_forecast'] == True]
        
        # Подготовка данных для ответа
        result_data = {
            'historical_count': len(historical_df),
            'forecast_count': len(forecast_df),
            'historical_preview': clean_dataframe_for_json(historical_df.tail(10)),
            'forecast_preview': clean_dataframe_for_json(forecast_df.head(20)),
            'download_url': f'/download/{session_id}',
            'metrics_summary': {},
            'quality_metrics': calculate_quality_metrics(metrics_by_metric, metric_columns)
        }
        
        return jsonify({'success': True, 'data': result_data})
        
    except Exception as e:
        return jsonify({'error': f'Ошибка при построении прогноза: {str(e)}\n{traceback.format_exc()}'}), 500


def calculate_quality_metrics(metrics_by_metric: Dict[str, List[Dict]], metric_columns: List[str]) -> Dict:
    quality_metrics = {}
    
    if not metrics_by_metric:
        return quality_metrics
    
    for metric in metric_columns:
        if metric not in metrics_by_metric or not metrics_by_metric[metric]:
            continue
        
        metric_list = metrics_by_metric[metric]
        
        # Извлечение значений метрик
        mae_values = [m.get('MAE') for m in metric_list if m.get('MAE') is not None]
        rmse_values = [m.get('RMSE') for m in metric_list if m.get('RMSE') is not None]
        r2_values = [m.get('R²') for m in metric_list if m.get('R²') is not None]
        mape_values = [m.get('MAPE') for m in metric_list if m.get('MAPE') is not None]
        
        # Подготовка деталей по срезам
        slices_details = []
        for slice_metrics in metric_list:
            slice_info = {
                'MAE': float(slice_metrics.get('MAE', 0)) if slice_metrics.get('MAE') is not None else None,
                'RMSE': float(slice_metrics.get('RMSE', 0)) if slice_metrics.get('RMSE') is not None else None,
                'R²': float(slice_metrics.get('R²', 0)) if slice_metrics.get('R²') is not None else None,
                'MAPE': float(slice_metrics.get('MAPE', 0)) if slice_metrics.get('MAPE') is not None else None,
                'model': slice_metrics.get('model', 'Unknown')
            }
            # Добавляем информацию о срезе (категориальные столбцы)
            slice_filters = {}
            for key, value in slice_metrics.items():
                if key.startswith('slice_'):
                    slice_filters[key.replace('slice_', '')] = value
            if slice_filters:
                slice_info['slice_filters'] = slice_filters
            slices_details.append(slice_info)
        
        # Усреднение метрик
        quality_metrics[metric] = {
            'MAE': float(np.mean(mae_values)) if mae_values else 0.0,
            'RMSE': float(np.mean(rmse_values)) if rmse_values else 0.0,
            'R²': float(np.mean(r2_values)) if r2_values else 0.0,
            'MAPE': float(np.mean(mape_values)) if mape_values else 0.0,
            'slices_count': len(metric_list),
            'models_distribution': count_models(metric_list),
            'slices_details': slices_details
        }
    
    return quality_metrics


def count_models(metric_list: List[Dict]) -> Optional[Dict[str, int]]:
    models = [m.get('model') for m in metric_list if m.get('model')]
    if not models:
        return None
    
    model_counts = defaultdict(int)
    for model in models:
        model_counts[model] += 1
    
    return dict(model_counts)


@app.route('/download/<session_id>')
def download_file(session_id):
    filepath = get_forecast_filepath(session_id)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name='forecast_result.csv')
    return jsonify({'error': 'Файл не найден'}), 404


@app.route('/api/forecast/list')
def list_forecasts():
    try:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            return jsonify({'forecasts': []})
        
        forecast_files = [f for f in os.listdir(results_dir) 
                          if f.startswith('forecast_') and f.endswith('.csv')]
        
        forecasts = []
        for filename in forecast_files:
            filepath = os.path.join(results_dir, filename)
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig', nrows=1)
                session_id = filename.replace('forecast_', '').replace('.csv', '')
                file_stat = os.stat(filepath)
                modified_time = pd.Timestamp.fromtimestamp(file_stat.st_mtime)
                
                forecasts.append({
                    'session_id': session_id,
                    'filename': filename,
                    'created': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'columns': list(df.columns)
                })
            except Exception:
                continue
        
        forecasts.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({'forecasts': forecasts})
    except Exception as e:
        return jsonify({'error': f'Ошибка при получении списка прогнозов: {str(e)}'}), 500


@app.route('/api/forecast/data/<session_id>')
def get_forecast_data(session_id):
    try:
        filepath = get_forecast_filepath(session_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл прогноза не найден'}), 404
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = create_date_column(df)
        
        if df['date'].isna().all():
            return jsonify({'error': 'В файле отсутствуют столбцы year и month'}), 400
        
        historical_df = df[df.get('is_forecast', False) == False]
        forecast_df = df[df.get('is_forecast', False) == True]
        
        # Определение категориальных столбцов
        time_columns = {'year', 'month', 'date', 'Quarter', 'Halfyear'}
        categorical_columns = []
        
        for col in df.columns:
            if col not in time_columns and col != 'is_forecast':
                unique_count = df[col].nunique()
                if 1 < unique_count <= 100:
                    try:
                        numeric_test = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                        if numeric_test.notna().sum() / len(df) < 0.5 or unique_count > 50:
                            categorical_columns.append(col)
                    except Exception:
                        categorical_columns.append(col)
        
        # Определение метрик
        forecast_metrics = []
        if len(forecast_df) > 0:
            for col in df.columns:
                if col not in time_columns and col not in categorical_columns and col != 'is_forecast':
                    try:
                        numeric_values = pd.to_numeric(
                            forecast_df[col].astype(str).str.replace(',', '').str.replace(' ', ''),
                            errors='coerce'
                        )
                        if numeric_values.notna().sum() > 0 and numeric_values.fillna(0).sum() > 0:
                            forecast_metrics.append(col)
                    except Exception:
                        pass
        
        # Получение уникальных значений категориальных столбцов
        categorical_values = {}
        for col in categorical_columns:
            unique_vals = sorted([str(c) for c in df[col].dropna().unique() 
                                 if c and str(c).strip() and str(c) != 'nan'])
            if unique_vals:
                categorical_values[col] = unique_vals
        
        return jsonify({
            'success': True,
            'metrics': forecast_metrics,
            'categorical_columns': categorical_values,
            'date_range': {
                'min': str(df['date'].min()),
                'max': str(df['date'].max())
            },
            'historical_count': len(historical_df),
            'forecast_count': len(forecast_df)
        })
    except Exception as e:
        return jsonify({'error': f'Ошибка при загрузке данных: {str(e)}\n{traceback.format_exc()}'}), 500


@app.route('/api/forecast/chart-data/<session_id>')
def get_chart_data(session_id):
    try:
        filepath = get_forecast_filepath(session_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл прогноза не найден'}), 404
        
        metric = request.args.get('metric', '')
        filters_json = request.args.get('filters', '{}')
        
        if not metric:
            return jsonify({'error': 'Не указана метрика'}), 400
        
        try:
            filters = json.loads(filters_json) if filters_json else {}
        except Exception:
            filters = {}
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = create_date_column(df)
        
        if metric not in df.columns:
            return jsonify({'error': f'Метрика {metric} не найдена в данных'}), 400
        
        df[metric] = pd.to_numeric(
            df[metric].astype(str).str.replace(',', '').str.replace(' ', ''),
            errors='coerce'
        ).fillna(0)
        
        # Применение фильтров
        if filters:
            for filter_col, filter_value in filters.items():
                if filter_col in df.columns and filter_value:
                    df = df[df[filter_col].astype(str) == str(filter_value)]
        
        # Агрегация по дате
        grouped = df.groupby(['date', 'is_forecast'])[metric].sum().reset_index()
        grouped = grouped.sort_values('date')
        
        historical = grouped[grouped['is_forecast'] == False]
        forecast = grouped[grouped['is_forecast'] == True]
        
        historical_dates = historical['date'].dt.strftime('%Y-%m-%d').tolist()
        forecast_dates = forecast['date'].dt.strftime('%Y-%m-%d').tolist()
        
        result = {
            'historical': {
                'dates': historical_dates,
                'values': historical[metric].tolist()
            },
            'forecast': {
                'dates': forecast_dates,
                'values': forecast[metric].tolist()
            },
            'transition_date': historical_dates[-1] if historical_dates else None,
            'last_historical_date': historical_dates[-1] if historical_dates else None,
            'first_forecast_date': forecast_dates[0] if forecast_dates else None
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Ошибка при получении данных графика: {str(e)}\n{traceback.format_exc()}'}), 500


def get_categorical_columns_info(df: pd.DataFrame) -> Dict[str, List[str]]:
    time_columns = {'year', 'month', 'date', 'Quarter', 'Halfyear', 'is_forecast'}
    sample_df = df.head(10000) if len(df) > 10000 else df
    
    # Определение метрик
    metric_columns = []
    for col in sample_df.columns:
        if col not in time_columns:
            try:
                numeric_test = pd.to_numeric(
                    sample_df[col].astype(str).str.replace(',', '').str.replace(' ', ''),
                    errors='coerce'
                )
                if numeric_test.notna().sum() / len(sample_df) > 0.7:
                    metric_columns.append(col)
            except Exception:
                pass
    
    # Определение категориальных столбцов
    categorical_info = {}
    for col in df.columns:
        if col not in time_columns and col not in metric_columns:
            sample_unique_count = sample_df[col].nunique()
            if 1 < sample_unique_count <= 200:
                if sample_unique_count < 50:
                    unique_values = sorted([
                        str(v) for v in sample_df[col].dropna().unique()
                        if str(v).strip() and str(v) != 'nan'
                    ])
                else:
                    unique_values = sorted([
                        str(v) for v in df[col].value_counts().head(200).index
                        if str(v).strip() and str(v) != 'nan'
                    ])
                
                if unique_values:
                    categorical_info[col] = unique_values
    
    return categorical_info


@app.route('/api/elasticity/calculate/<session_id>')
def calculate_elasticity(session_id):
    try:
        filepath = get_forecast_filepath(session_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл прогноза не найден'}), 404
        
        x_metric = request.args.get('x_metric', '')
        y_metric = request.args.get('y_metric', '')
        
        if not x_metric or not y_metric:
            return jsonify({'error': 'Не указаны метрики для расчета'}), 400
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = create_date_column(df)
        
        historical_df = df[df.get('is_forecast', False) == False]
        
        if len(historical_df) == 0:
            return jsonify({'error': 'Нет исторических данных для расчета эластичности'}), 400
        
        categorical_info = get_categorical_columns_info(historical_df)
        
        return jsonify({
            'success': True,
            'x_metric': x_metric,
            'y_metric': y_metric,
            'categorical_columns': categorical_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка при получении данных: {str(e)}\n{traceback.format_exc()}'
        }), 500


@app.route('/api/elasticity/calculate-slice/<session_id>')
def calculate_elasticity_slice(session_id):
    try:
        filepath = get_forecast_filepath(session_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл прогноза не найден'}), 404
        
        x_metric = request.args.get('x_metric', '')
        y_metric = request.args.get('y_metric', '')
        filters_json = request.args.get('filters', '{}')
        
        if not x_metric or not y_metric:
            return jsonify({'error': 'Не указаны метрики для расчета'}), 400
        
        try:
            filters = json.loads(filters_json) if filters_json else {}
        except Exception:
            filters = {}
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = create_date_column(df)
        
        historical_df = df[df.get('is_forecast', False) == False]
        
        # Применение фильтров
        if filters:
            for filter_col, filter_value in filters.items():
                if filter_col in historical_df.columns and filter_value and filter_value != 'all':
                    historical_df = historical_df[historical_df[filter_col].astype(str) == str(filter_value)]
        
        if len(historical_df) < 3:
            return jsonify({
                'success': False,
                'error': 'Недостаточно данных для расчета после применения фильтров (нужно минимум 3 точки)'
            }), 400
        
        calculator = ElasticityCalculator()
        result = calculator.calculate_elasticity(historical_df, x_metric, y_metric)
        
        if not result or not result.get('success'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Не удалось рассчитать эластичность')
            }), 400
        
        return jsonify({
            'success': True,
            'result': convert_result_to_json(result)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка при расчете эластичности: {str(e)}\n{traceback.format_exc()}'
        }), 500


def convert_result_to_json(result: Dict) -> Dict:
    if not result or result.get('success') is False:
        return {
            'success': False,
            'error': result.get('error', 'Ошибка расчета') if result else 'Неизвестная ошибка'
        }
    
    return {
        'success': True,
        'elasticity': float(result.get('elasticity', 0)) if result.get('elasticity') is not None else None,
        'intercept': float(result.get('intercept', 0)) if result.get('intercept') is not None else None,
        'baseline': float(result.get('baseline', 0)) if result.get('baseline') is not None else None,
        'method': result.get('method', 'without_baseline'),
        'r2': float(result.get('r2', 0)) if result.get('r2') is not None else None,
        'mae': float(result.get('mae', 0)) if result.get('mae') is not None else None,
        'rmse': float(result.get('rmse', 0)) if result.get('rmse') is not None else None,
        'mape': float(result.get('mape', 0)) if result.get('mape') is not None else None,
        'n_points': int(result.get('n_points', 0)) if result.get('n_points') is not None else 0,
        'interpretation': result.get('interpretation', '')
    }


@app.route('/api/correlation/predict')
def predict_correlation():
    try:
        x_value = float(request.args.get('x_value', 0))
        model_type = request.args.get('model_type', 'logarithmic')
        degree = int(request.args.get('degree', 1))
        coefficients_json = request.args.get('coefficients', '[]')
        intercept = float(request.args.get('intercept', 0))
        baseline = request.args.get('baseline', None)
        baseline = float(baseline) if baseline and baseline != 'None' else None
        
        try:
            coefficients = json.loads(coefficients_json) if coefficients_json else []
        except Exception:
            coefficients = []
        
        if x_value <= 0:
            return jsonify({
                'success': False,
                'error': 'Значение X должно быть положительным'
            }), 400
        
        y_pred = 0.0
        
        if model_type == 'logarithmic':
            elasticity = float(coefficients[0]) if coefficients else 0
            log_x = np.log(x_value)
            log_y = intercept + elasticity * log_x
            
            if baseline is not None:
                y_pred = baseline + np.exp(log_y)
            else:
                y_pred = np.exp(log_y)
        elif model_type == 'polynomial':
            y_pred = intercept
            for i in range(1, len(coefficients)):
                y_pred += coefficients[i] * (x_value ** i)
            
            if baseline is not None:
                y_pred = baseline + y_pred
        
        return jsonify({
            'success': True,
            'y_pred': float(max(0, y_pred))
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка при расчете: {str(e)}\n{traceback.format_exc()}'
        }), 500


@app.route('/api/scenario/data/<session_id>')
def get_scenario_data(session_id):
    """Получение агрегированных данных по временному ряду для моделирования сценариев"""
    try:
        filepath = get_forecast_filepath(session_id)
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл прогноза не найден'}), 404
        
        x_metric = request.args.get('x_metric', '')
        y_metric = request.args.get('y_metric', '')
        filters_json = request.args.get('filters', '{}')
        
        if not x_metric or not y_metric:
            return jsonify({'error': 'Не указаны метрики'}), 400
        
        try:
            filters = json.loads(filters_json) if filters_json else {}
        except Exception:
            filters = {}
        
        df = pd.read_csv(filepath, encoding='utf-8-sig')
        df['date'] = create_date_column(df)
        
        if 'is_forecast' not in df.columns:
            return jsonify({
                'success': False,
                'error': 'Столбец is_forecast не найден в данных'
            }), 400
        
        # Фильтрация прогнозных данных
        is_forecast_str = df['is_forecast'].astype(str).str.strip().str.lower()
        forecast_mask = is_forecast_str.isin(['true', '1', 'yes', 't'])
        
        if df['is_forecast'].dtype == 'bool':
            forecast_mask = forecast_mask | (df['is_forecast'] == True)
        elif df['is_forecast'].dtype in ['int64', 'int32', 'float64', 'float32']:
            forecast_mask = forecast_mask | (df['is_forecast'] == 1)
        
        forecast_df = df[forecast_mask]
        
        if len(forecast_df) == 0:
            return jsonify({
                'success': False,
                'error': 'Нет прогнозных данных (is_forecast=True) в файле'
            }), 400
        
        # Применение фильтров
        if filters:
            for filter_col, filter_value in filters.items():
                if filter_col in forecast_df.columns and filter_value and filter_value != 'all':
                    forecast_df = forecast_df[forecast_df[filter_col].astype(str) == str(filter_value)]
        
        if len(forecast_df) == 0:
            return jsonify({
                'success': False,
                'error': 'Нет прогнозных данных после применения фильтров'
            }), 400
        
        # Агрегация по дате
        if 'date' in forecast_df.columns:
            forecast_df[x_metric] = pd.to_numeric(
                forecast_df[x_metric].astype(str).str.replace(',', '').str.replace(' ', ''),
                errors='coerce'
            )
            forecast_df[y_metric] = pd.to_numeric(
                forecast_df[y_metric].astype(str).str.replace(',', '').str.replace(' ', ''),
                errors='coerce'
            )
            
            aggregated = forecast_df.groupby('date').agg({
                x_metric: 'sum',
                y_metric: 'sum'
            }).reset_index()
            
            aggregated = aggregated.sort_values('date')
            
            result_data = []
            for _, row in aggregated.iterrows():
                result_data.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'x_value': float(row[x_metric]) if pd.notna(row[x_metric]) else 0,
                    'y_value': float(row[y_metric]) if pd.notna(row[y_metric]) else 0
                })
            
            return jsonify({
                'success': True,
                'data': result_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Не найдена колонка с датой'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Ошибка при получении данных: {str(e)}\n{traceback.format_exc()}'
        }), 500


if __name__ == '__main__':
    import sys
    print("="*70)
    print("Запуск Flask приложения...")
    print("="*70)
    print(f"Python версия: {sys.version}")
    print(f"Папка загрузок: {app.config['UPLOAD_FOLDER']}")
    print(f"Папка результатов: results")
    print("="*70)
    print("Сервер будет доступен по адресу: http://localhost:5000")
    print("Нажмите Ctrl+C для остановки")
    print("="*70)
    
    try:
        app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\nОстановка сервера...")
    except Exception as e:
        print(f"\nОшибка при запуске: {e}")
        traceback.print_exc()
        sys.exit(1)
