import folium
from folium import plugins
from datetime import datetime, timedelta
import time
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import sys
import warnings
import traceback
from typing import Tuple, Optional, Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
import logging


logger = logging.getLogger('cmdstanpy')
logger.setLevel(logging.WARNING)


GPS_position = (61.285552, 73.347173, 8)


HISTORY_FILE = 'parking_history.csv'
SAVE_HISTORY = True
BATCH_SIZE = 30


N_SPLITS = 3
TEST_SIZE_HOURS = 72
INITIAL_TRAIN_SIZE_DAYS = 7
GAP_HOURS = 12


PROPHET_PARAMS_GRID = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.1, 1.0, 10.0],
    'seasonality_mode': ['additive'],
    'yearly_seasonality': [False],
    'weekly_seasonality': [True],
    'daily_seasonality': [True]
}


PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger('prophet').setLevel(logging.WARNING)


def filter_first_12_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует первые 12 часов данных из DataFrame"""
    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['datetime'])
    min_date = df['datetime'].min()
    cutoff_date = min_date + timedelta(hours=12)
    return df[df['datetime'] > cutoff_date].copy()


class ParkingAnalyzer:
    def __init__(self):
        self.parking_spots = []
        self.occupied_spots = []
        self.free_spots = []
        self.spot_counter = 1
        self.overpass_url = "http://overpass-api.de/api/interpreter"

    def load_parking_spots_from_csv(self, csv_file: str) -> None:
        df = pd.read_csv(csv_file)
        grouped = df.groupby('spot_id').first()

        for spot_id, row in grouped.iterrows():
            self.parking_spots.append({
                'id': int(spot_id),
                'center': (row['latitude'], row['longitude']),
                'orientation': row['orientation'],
                'length': row['spot_length'],
                'width': row['spot_width'],
                'vehicle_type': row['vehicle_type'],
                'occupied': False,
                'last_used': 0,
                'last_used_str': ''
            })
        print(f"Загружено {len(self.parking_spots)} парковочных мест")

    def update_occupancy(self, timestamp: float, csv_file: str) -> None:
        df = pd.read_csv(csv_file)
        current_data = df[df['timestamp'] == timestamp]

        if current_data.empty:
            return

        self.occupied_spots = []
        self.free_spots = []

        for _, row in current_data.iterrows():
            spot = next((s for s in self.parking_spots if s['id'] == row['spot_id']), None)
            if spot:
                spot['occupied'] = row['status'] == 'occupied'
                spot['last_used'] = timestamp
                spot['last_used_str'] = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

                if spot['occupied']:
                    self.occupied_spots.append(spot)
                else:
                    self.free_spots.append(spot)

    def get_parking_spots(self) -> list:
        return self.parking_spots

    def get_occupied_spots(self) -> list:
        return self.occupied_spots

    def get_free_spots(self) -> list:
        return self.free_spots

    def _is_point_in_polygon(self, point: Tuple[float, float], polygon_points: list) -> bool:
        x, y = point
        n = len(polygon_points)
        inside = False
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_spot_bbox(self, spot: dict) -> list:
        try:
            angle = spot['orientation']
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            half_l = spot['length'] / 2 / 0.2
            half_w = spot['width'] / 2 / 0.2

            corners = [
                [-half_l, -half_w],
                [-half_l, half_w],
                [half_l, half_w],
                [half_l, -half_w]
            ]

            rotated = []
            for x, y in corners:
                rx = x * cos_a - y * sin_a + spot['center'][0]
                ry = x * sin_a + y * cos_a + spot['center'][1]
                rotated.append([rx, ry])

            bbox = []
            for point in rotated:
                bbox.extend(point)
            return bbox
        except:
            return [0, 0, 0, 0, 0, 0, 0, 0]


def print_metrics_table(metrics: list) -> None:
    """Печатает таблицу с метриками по фолдам"""
    from tabulate import tabulate

    table_data = []
    for m in metrics:
        table_data.append([
            m['fold'],
            f"{m['MAE']:.4f}",
            f"{m['MSE']:.4f}",
            f"{m['RMSE']:.4f}",
            f"{m['R2']:.4f}",
            m['params']
        ])

    print("\nМЕТРИКИ КРОСС-ВАЛИДАЦИИ:")
    print(tabulate(
        table_data,
        headers=["Фолд", "MAE", "MSE", "RMSE", "R²", "Параметры"],
        tablefmt="grid",
        stralign="center"
    ))
    print()


def convert_pixel_to_gps(xy: Tuple[float, float],
                         image_size: Tuple[int, int] = (1080, 608),
                         gps_origin: Tuple[float, float] = (61.285552, 73.347173),
                         meters_per_pixel: float = 0.2) -> Tuple[float, float]:
    dx = xy[0] - image_size[0] / 2
    dy = image_size[1] / 2 - xy[1]
    dlat = dy * meters_per_pixel / 111_000
    dlon = dx * meters_per_pixel / (111_000 * np.cos(np.radians(gps_origin[0])))
    return gps_origin[0] + dlat, gps_origin[1] + dlon


def prepare_data_for_prophet(df: pd.DataFrame, train_mean: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Подготовка данных с предотвращением утечек"""
    df_prophet = df[['datetime', 'occupancy_rate']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])


    df_prophet['hour_sin'] = np.sin(2 * np.pi * df_prophet['ds'].dt.hour / 24)
    df_prophet['hour_cos'] = np.cos(2 * np.pi * df_prophet['ds'].dt.hour / 24)
    df_prophet['day_sin'] = np.sin(2 * np.pi * df_prophet['ds'].dt.dayofweek / 7)
    df_prophet['day_cos'] = np.cos(2 * np.pi * df_prophet['ds'].dt.dayofweek / 7)

    if train_mean is None:
        train_mean = {col: df_prophet[col].mean() for col in df_prophet.columns if col not in ['ds', 'y']}
        return df_prophet, train_mean
    else:
        for col in train_mean:
            if col in df_prophet:
                df_prophet[col] = train_mean[col]
        return df_prophet, None


def time_series_train_test_split(df: pd.DataFrame,
                                 test_size_hours: int = TEST_SIZE_HOURS,
                                 initial_train_size_days: int = INITIAL_TRAIN_SIZE_DAYS) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('ds')
    min_date = df['ds'].min()
    max_date = df['ds'].max()

    test_start_date = max_date - timedelta(hours=test_size_hours)
    train_end_date = min_date + timedelta(days=initial_train_size_days)

    if test_start_date < train_end_date:
        train_end_date = test_start_date - timedelta(hours=1)

    train = df[df['ds'] <= train_end_date]
    test = df[df['ds'] > test_start_date]
    return train, test


def evaluate_model(model: Prophet, train: pd.DataFrame, test: pd.DataFrame) -> Dict[str, float]:
    future = model.make_future_dataframe(periods=len(test), freq='H', include_history=False)
    forecast = model.predict(future)

    evaluation = test.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
    mse = mean_squared_error(evaluation['y'], evaluation['yhat'])
    rmse = np.sqrt(mse)
    r2 = r2_score(evaluation['y'], evaluation['yhat'])

    residuals = evaluation['y'] - evaluation['yhat']
    plt.figure(figsize=(12, 6))
    plt.plot(residuals)
    plt.title('График остатков')
    plt.savefig(os.path.join(PLOT_DIR, 'residuals_plot.png'))

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


def save_plot(fig, filename: str) -> None:
    """Сохраняет график в файл и закрывает фигуру"""
    plot_path = os.path.join(PLOT_DIR, filename)
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"График сохранен: {plot_path}")


def make_forecast_from_last_point(model: Prophet, last_point: pd.DataFrame, periods: int = 72) -> pd.DataFrame:
    """Создает прогноз начиная с последней точки данных"""
    future = model.make_future_dataframe(periods=periods, freq='H', include_history=False)


    if 'hour_sin' in last_point.columns:
        future['hour_sin'] = np.sin(2 * np.pi * future['ds'].dt.hour / 24)
        future['hour_cos'] = np.cos(2 * np.pi * future['ds'].dt.hour / 24)
        future['day_sin'] = np.sin(2 * np.pi * future['ds'].dt.dayofweek / 7)
        future['day_cos'] = np.cos(2 * np.pi * future['ds'].dt.dayofweek / 7)


        scaler = StandardScaler()
        features_to_normalize = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        scaler.fit(last_point[features_to_normalize])
        future[features_to_normalize] = scaler.transform(future[features_to_normalize])

    forecast = model.predict(future)
    return forecast


def analyze_parking_trends(history_file: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Анализ временных рядов загруженности парковки с кросс-валидацией и сохранением лучшей модели
    с устранением утечек данных
    """
    try:

        print("\n[1/5] Загрузка и подготовка данных...")
        df = pd.read_csv(history_file, parse_dates=['datetime'])
        df = filter_first_12_hours(df)

        if df.empty:
            raise ValueError("Недостаточно данных после фильтрации")


        print("[2/5] Обнаружение аномалий...")


        print("[3/5] Подготовка данных для Prophet...")
        df_prophet_base = df[['datetime', 'occupancy_rate']].copy()
        df_prophet_base.columns = ['ds', 'y']
        df_prophet_base['ds'] = pd.to_datetime(df_prophet_base['ds'])


        print("[4/5] Кросс-валидация и подбор параметров...")
        tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE_HOURS * 6, gap=GAP_HOURS * 6)

        best_params = None
        best_metrics = {
            'MAE': float('inf'),
            'MSE': float('inf'),
            'RMSE': float('inf'),
            'R2': -float('inf')
        }
        best_model = None
        all_metrics = []

        for params in tqdm(ParameterGrid(PROPHET_PARAMS_GRID), desc="Подбор параметров"):
            for fold, (train_idx, test_idx) in enumerate(tscv.split(df_prophet_base)):

                train_base, test_base = df_prophet_base.iloc[train_idx], df_prophet_base.iloc[test_idx]


                clf = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
                train_base['is_anomaly'] = clf.fit_predict(train_base[['y']])
                train_base = train_base[train_base['is_anomaly'] == 1]


                train = train_base.copy()
                test = test_base.copy()


                train.loc[:, 'hour_sin'] = np.sin(2 * np.pi * train['ds'].dt.hour / 24)
                train.loc[:, 'hour_cos'] = np.cos(2 * np.pi * train['ds'].dt.hour / 24)
                train.loc[:, 'day_sin'] = np.sin(2 * np.pi * train['ds'].dt.dayofweek / 7)
                train.loc[:, 'day_cos'] = np.cos(2 * np.pi * train['ds'].dt.dayofweek / 7)


                scaler = StandardScaler()
                features_to_normalize = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
                train[features_to_normalize] = scaler.fit_transform(train[features_to_normalize])


                model = Prophet(**params)
                for col in train.columns:
                    if col not in ['ds', 'y', 'is_anomaly']:
                        model.add_regressor(col)

                with suppress_stdout_stderr():
                    model.fit(train)


                future = model.make_future_dataframe(periods=len(test), freq='H')
                for col in features_to_normalize:
                    future[col] = 0


                forecast = model.predict(future)


                evaluation = test.merge(forecast[['ds', 'yhat']], on='ds')
                if not evaluation.empty:
                    mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])
                    mse = mean_squared_error(evaluation['y'], evaluation['yhat'])
                    rmse = np.sqrt(mse)
                    r2 = r2_score(evaluation['y'], evaluation['yhat'])
                else:
                    mae, mse, rmse, r2 = np.nan, np.nan, np.nan, np.nan

                current_metrics = {
                    'fold': fold + 1,
                    'params': str(params),
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                }
                all_metrics.append(current_metrics)

                if not np.isnan(rmse) and rmse < best_metrics['RMSE']:
                    best_metrics = current_metrics.copy()
                    best_params = params
                    best_model = model
                    best_rmse = rmse


        print_metrics_table(all_metrics)


        print("[5/5] Сохранение результатов...")
        if not best_model:
            raise RuntimeError("Не удалось обучить ни одну модель")


        final_train = df_prophet_base.copy()
        final_train['hour_sin'] = np.sin(2 * np.pi * final_train['ds'].dt.hour / 24)
        final_train['hour_cos'] = np.cos(2 * np.pi * final_train['ds'].dt.hour / 24)
        final_train['day_sin'] = np.sin(2 * np.pi * final_train['ds'].dt.dayofweek / 7)
        final_train['day_cos'] = np.cos(2 * np.pi * final_train['ds'].dt.dayofweek / 7)


        scaler = StandardScaler()
        features_to_normalize = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        final_train[features_to_normalize] = scaler.fit_transform(final_train[features_to_normalize])


        final_model = Prophet(**best_params)
        for col in final_train.columns:
            if col not in ['ds', 'y']:
                final_model.add_regressor(col)

        final_model.fit(final_train)


        future = final_model.make_future_dataframe(periods=72, freq='H')
        for col in features_to_normalize:
            future[col] = 0

        forecast = final_model.predict(future)


        model_dir = 'saved_models'
        os.makedirs(model_dir, exist_ok=True)
        model_filename = 'parking_model.json'
        model_path = os.path.join(model_dir, model_filename)

        with open(model_path, 'w') as fout:
            fout.write(model_to_json(final_model))

        print(f"\nМодель успешно сохранена в файл: {model_path}")


        last_point = final_train.iloc[[-1]].copy()
        forecast = make_forecast_from_last_point(final_model, last_point, periods=72)


        fig1 = final_model.plot(forecast)
        plt.title(f"Прогноз загруженности (RMSE={best_rmse:.4f}")
        save_plot(fig1, 'final_forecast.png')
        plt.close()

        fig2 = final_model.plot_components(forecast)
        save_plot(fig2, 'final_components.png')
        plt.close()


        evaluation = final_train.merge(forecast[['ds', 'yhat']], on='ds')
        residuals = evaluation['y'] - evaluation['yhat']

        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        ax1.scatter(evaluation['yhat'], residuals, alpha=0.5)
        ax1.axhline(0, color='red')
        ax1.set_xlabel('Предсказанные значения')
        ax1.set_ylabel('Остатки')
        ax1.set_title('Остатки vs Прогноз')

        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(0, color='red')
        ax2.set_title('Распределение остатков')

        save_plot(fig3, 'residuals_analysis.png')
        plt.close()


        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ".center(80))
        print("=" * 80)
        print(f"Лучшие параметры:".ljust(30) + f"{best_params}")
        print(f"Лучший RMSE:".ljust(30) + f"{best_rmse:.4f}")
        print(f"R²:".ljust(30) + f"{best_metrics.get('R2', 'N/A'):.4f}")
        print(f"Диапазон данных:".ljust(30) + f"{final_train['ds'].min().date()} - {final_train['ds'].max().date()}")
        print("=" * 80)

        return forecast, best_params

    except Exception as e:
        print(f"\nОШИБКА: {str(e)}")
        traceback.print_exc()
        return None, None


def visualize_cv_results(metrics: List[Dict]) -> None:
    """Визуализация результатов кросс-валидации"""
    if not metrics:
        print("Нет данных для визуализации")
        return

    try:
        df_metrics = pd.DataFrame(metrics)

        if 'params' not in df_metrics.columns:
            if len(metrics) > 0 and isinstance(metrics[0], list):
                df_metrics = pd.DataFrame([m for sublist in metrics for m in sublist])
            else:
                raise ValueError("Данные метрик не содержат информацию о параметрах")

        df_metrics['params_str'] = df_metrics['params'].apply(
            lambda x: "\n".join(f"{k}={v}" for k, v in eval(x).items()) if isinstance(x, str) else str(x)
        )

        if 'RMSE' in df_metrics.columns:
            fig1 = plt.figure(figsize=(12, 6))
            sns.boxplot(data=df_metrics, x='params_str', y='RMSE')
            plt.xticks(rotation=45, ha='right')
            plt.title('Распределение RMSE по параметрам')
            plt.tight_layout()
            save_plot(fig1, 'cv_rmse_distribution.png')
        else:
            print("Отсутствуют данные RMSE для визуализации")

        if all(col in df_metrics.columns for col in ['RMSE', 'R2']):
            pivot_table = df_metrics.pivot_table(
                index='params_str',
                values=['RMSE', 'R2'],
                aggfunc='mean'
            ).sort_values('RMSE')

            fig2 = plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title('Сравнение параметров моделей')
            plt.tight_layout()
            save_plot(fig2, 'params_heatmap.png')
        else:
            print("Отсутствуют данные для тепловой карты")

    except Exception as e:
        print(f"Ошибка при визуализации результатов: {str(e)}")
        traceback.print_exc()

class suppress_stdout_stderr:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr


def create_parking_map(parking_spots: list,
                       gps_origin: Tuple[float, float],
                       image_size: Tuple[int, int]) -> folium.Map:
    m = folium.Map(location=gps_origin, zoom_start=18)

    folium.Marker(
        location=gps_origin,
        popup="Камера наблюдения",
        tooltip="Камера наблюдения",
        icon=folium.Icon(color='blue', icon='video', prefix='fa')
    ).add_to(m)

    free_group = folium.FeatureGroup(name='Свободные места')
    occupied_group = folium.FeatureGroup(name='Занятые места')
    m.add_child(free_group)
    m.add_child(occupied_group)

    centers = [spot['center'] for spot in parking_spots]
    if centers:
        center_lat = sum(p[0] for p in centers) / len(centers)
        center_lon = sum(p[1] for p in centers) / len(centers)
        m.location = [center_lat, center_lon]

    for spot in parking_spots:
        gps_coords = spot['center']
        angle = spot['orientation']
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        half_l = spot['length'] / 2 / 0.2
        half_w = spot['width'] / 2 / 0.2
        corners_px = [
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, -half_w],
        ]
        rotated = []
        for x, y in corners_px:
            rx = x * cos_a - y * sin_a + 540
            ry = x * sin_a + y * cos_a + 304
            gps = convert_pixel_to_gps((rx, ry), image_size, gps_origin)
            rotated.append(gps)

        color = 'red' if spot['occupied'] else 'green'
        group = occupied_group if spot['occupied'] else free_group
        status = "Занято" if spot['occupied'] else "Свободно"

        folium.Polygon(
            locations=rotated,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.2,
            weight=1,
            popup=f"Место {spot['id']} ({'грузовик' if spot.get('vehicle_type') == 'truck' else 'авто'})\n"
                  f"Статус: {status}\n"
                  f"Размеры: {spot['length']:.1f}x{spot['width']:.1f} м\n"
                  f"Последнее использование: {spot['last_used_str']}"
        ).add_to(group)

        icon_color = 'red' if spot['occupied'] else 'green'
        folium.Marker(
            location=gps_coords,
            popup=f"Место {spot['id']}\nТип: {'грузовик' if spot.get('vehicle_type') == 'truck' else 'авто'}\nСтатус: {status}",
            icon=folium.Icon(color=icon_color,
                             icon='car' if spot.get('vehicle_type') == 'car' else 'truck',
                             prefix='fa')
        ).add_to(group)

    folium.LayerControl().add_to(m)
    minimap = plugins.MiniMap()
    m.add_child(minimap)

    return m


def main():
    analyzer = ParkingAnalyzer()
    image_size = (1080, 608)
    csv_file = "parking_data_UlKrylova.csv"

    print("Запуск анализа парковки...")
    analyzer.load_parking_spots_from_csv(csv_file)



    if SAVE_HISTORY and not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            f.write('timestamp,datetime,total_spots,occupied_spots,free_spots,occupancy_rate\n')

    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df = filter_first_12_hours(df)

    if df.empty:
        print("Недостаточно данных после исключения первых 12 часов")
        return

    timestamps = df['timestamp'].unique()

    for timestamp in tqdm(timestamps, desc="Обработка данных"):
        analyzer.update_occupancy(timestamp, csv_file)

        if SAVE_HISTORY:
            current_data = df[df['timestamp'] == timestamp]
            total = len(current_data['spot_id'].unique())
            occupied = sum(current_data['status'] == 'occupied')
            free = total - occupied
            rate = occupied / total if total > 0 else 0

            with open(HISTORY_FILE, 'a') as f:
                dt_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{timestamp},{dt_str},{total},{occupied},{free},{rate:.2f}\n")

    print("\nАнализ данных и прогнозирование...")
    forecast, metrics = analyze_parking_trends(HISTORY_FILE)

if __name__ == "__main__":
    print("Начало выполнения программы")
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nПрограмма завершена. Время выполнения: {end_time - start_time:.2f} секунд")