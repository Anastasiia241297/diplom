import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from prophet.serialize import model_from_json
from datetime import timedelta
import os
import logging
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
MODEL_FILE = os.path.join(MODEL_DIR, 'parking_model.json')
TEST_DATA_FILE = os.path.join(BASE_DIR, 'parking_data_test_photo.csv')
PLOT_DIR = os.path.join(BASE_DIR, 'prediction_plots')
LOG_FILE = os.path.join(BASE_DIR, 'parking_analysis.log')

FORECAST_HORIZON = 144
INTERVAL = '10min'

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_model():
    """Загрузка сохраненной модели Prophet"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_FILE}")

    logger.info(f"Попытка загрузки модели из: {MODEL_FILE}")

    with open(MODEL_FILE, 'r') as f:
        try:
            model = model_from_json(f.read())
            logger.info("Модель успешно загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise


def prepare_test_data():
    """Подготовка тестовых данных с расширенной проверкой"""
    try:
        df = pd.read_csv(TEST_DATA_FILE)
        if df.empty:
            raise ValueError("Тестовый файл пуст")

        df['datetime'] = pd.to_datetime(df['datetime'])

        df['occupied_spots'] = df['status'].apply(lambda x: 1 if x == 'occupied' else 0)


        if 'total_spots' not in df.columns or 'occupied_spots' not in df.columns:
            raise ValueError("Отсутствуют необходимые колонки в данных")


        if (df['occupied_spots'] < 0).any():
            logger.warning("Обнаружены отрицательные значения occupied_spots! Корректируем...")
            df['occupied_spots'] = df['occupied_spots'].clip(lower=0)

        if (df['occupied_spots'] > df['total_spots']).any():
            logger.warning("Обнаружены occupied_spots > total_spots! Корректируем...")
            df['occupied_spots'] = df['occupied_spots'].clip(upper=df['total_spots'])

        agg_data = df.groupby('datetime').agg({
            'total_spots': 'first',
            'status': lambda x: (x == 'occupied').sum()  # Считаем только occupied
        }).rename(columns={'status': 'occupied_spots'}).reset_index()

        agg_data['occupancy_rate'] = agg_data['occupied_spots'] / agg_data['total_spots']

        # Логирование текущего состояния
        last_record = agg_data.iloc[-1]
        logger.info(f"Текущее состояние парковки: "
                    f"Занято {last_record['occupied_spots']}/{last_record['total_spots']} мест "
                    f"({last_record['occupancy_rate']:.1%})")

        agg_data.set_index('datetime', inplace=True)
        return agg_data

    except Exception as e:
        logger.error(f"Ошибка подготовки данных: {str(e)}")
        raise


def make_forecast(model, data, forecast_hours=24):
    """
    Генерация прогноза на основе текущего состояния
    """
    try:

        if data.empty:
            raise ValueError("Нет данных для прогнозирования")

        last_point = data.iloc[-1]
        current_time = data.index[-1]
        current_occupancy = last_point['occupancy_rate']

        logger.info(f"Прогнозирование на основе данных: "
                    f"Время: {current_time}, "
                    f"Загруженность: {current_occupancy:.1%}")


        periods = forecast_hours * 6
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=current_time + pd.Timedelta(minutes=10),
                periods=periods,
                freq=INTERVAL
            )
        })


        future['initial_occupancy'] = current_occupancy


        future['hour_sin'] = np.sin(2 * np.pi * future['ds'].dt.hour / 24)
        future['hour_cos'] = np.cos(2 * np.pi * future['ds'].dt.hour / 24)
        future['day_sin'] = np.sin(2 * np.pi * future['ds'].dt.dayofweek / 7)
        future['day_cos'] = np.cos(2 * np.pi * future['ds'].dt.dayofweek / 7)


        forecast = model.predict(future)


        transition_points = 6
        for i in range(transition_points):
            if i >= len(forecast):
                break
            alpha = i / (transition_points - 1)
            forecast.loc[i, 'yhat'] = (1 - alpha) * current_occupancy + alpha * forecast.loc[i, 'yhat']
            forecast.loc[i, 'yhat_lower'] = (1 - alpha) * max(current_occupancy * 0.95, 0) + alpha * forecast.loc[i, 'yhat_lower']
            forecast.loc[i, 'yhat_upper'] = (1 - alpha) * min(current_occupancy * 1.05, 1) + alpha * forecast.loc[i, 'yhat_upper']


        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = forecast[col].clip(0, 1)

        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    except Exception as e:
        logger.error(f"Ошибка прогнозирования: {str(e)}")
        return pd.DataFrame()


def plot_combined(data, forecast):
    """Визуализация с правильным смещением и сглаживанием"""
    plt.figure(figsize=(18, 10), dpi=100)


    last_point = data.iloc[-1]
    current_time = data.index[-1]
    current_occupancy = last_point['occupancy_rate']
    total_spots = last_point['total_spots']
    occupied_spots = last_point['occupied_spots']
    free_spots = total_spots - occupied_spots


    hist_data = data[data.index >= (current_time - timedelta(hours=2))]
    plt.plot(hist_data.index, hist_data['occupancy_rate'],
             'b-', linewidth=2, label='Исторические данные')


    plt.scatter([current_time], [current_occupancy],
                color='blue', s=200, zorder=5,
                label=f'Текущее: {current_occupancy:.0%}')


    if not forecast.empty:

        initial_forecast = forecast['yhat'].iloc[0]
        offset = current_occupancy - initial_forecast


        adjusted_forecast = forecast.copy()
        adjusted_forecast['yhat'] += offset
        adjusted_forecast['yhat_lower'] += offset
        adjusted_forecast['yhat_upper'] += offset


        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            adjusted_forecast[col] = adjusted_forecast[col].clip(0, 1)


        adjusted_forecast.loc[0, 'yhat'] = current_occupancy
        adjusted_forecast.loc[0, 'yhat_lower'] = max(current_occupancy * 0.95, 0)
        adjusted_forecast.loc[0, 'yhat_upper'] = min(current_occupancy * 1.05, 1)


        transition_points = 18
        smoothing_factor = 0.2

        for i in range(1, min(transition_points, len(adjusted_forecast))):
            alpha = 1 - np.exp(-i / (transition_points * smoothing_factor))
            adjusted_forecast.loc[i, 'yhat'] = (1 - alpha) * current_occupancy + alpha * adjusted_forecast.loc[
                i, 'yhat']
            adjusted_forecast.loc[i, 'yhat_lower'] = (1 - alpha) * max(current_occupancy * 0.95, 0) + alpha * \
                                                     adjusted_forecast.loc[i, 'yhat_lower']
            adjusted_forecast.loc[i, 'yhat_upper'] = (1 - alpha) * min(current_occupancy * 1.05, 1) + alpha * \
                                                     adjusted_forecast.loc[i, 'yhat_upper']


        plt.plot(adjusted_forecast['ds'], adjusted_forecast['yhat'],
                 'r-', linewidth=3, label='Прогноз (скорректированный)')

        plt.fill_between(adjusted_forecast['ds'],
                         adjusted_forecast['yhat_lower'],
                         adjusted_forecast['yhat_upper'],
                         color='pink', alpha=0.2,
                         label='Доверительный интервал')

        final_forecast = adjusted_forecast
    else:
        final_forecast = None


    plt.axvline(x=current_time, color='green', linestyle='--',
                linewidth=2, label='Начало прогноза')


    plt.ylim(0, 1.05)
    if final_forecast is not None and not final_forecast.empty:
        plt.xlim([current_time - timedelta(hours=2),
                  final_forecast['ds'].iloc[-1] + timedelta(hours=2)])
    else:
        plt.xlim([current_time - timedelta(hours=2),
                  current_time + timedelta(hours=4)])


    info_text = (f"Занято: {occupied_spots:.0f}/{total_spots:.0f} мест\n"
                 f"Свободно: {free_spots:.0f} мест\n"
                 f"Загруженность: {current_occupancy:.1%}")

    plt.annotate(info_text,
                 xy=(current_time, current_occupancy),
                 xytext=(20, 20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.9),
                 fontsize=12)


    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
    plt.title('Прогноз загруженности парковки', fontsize=16, pad=20)
    plt.xlabel('Время', fontsize=14)
    plt.ylabel('Загруженность', fontsize=14)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(PLOT_DIR, 'current_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"График сохранен: {plot_path}")


def save_full_results(data, forecast):
    """Сохранение результатов с правильным смещением и обработкой ошибок"""
    try:

        last_point = data.iloc[-1]
        current_time = data.index[-1]
        current_occupancy = last_point['occupancy_rate']
        total_spots = last_point['total_spots']
        occupied_spots = last_point['occupied_spots']


        initial_forecast = forecast['yhat'].iloc[0]
        offset = current_occupancy - initial_forecast


        result = forecast.copy()
        result.columns = ['datetime', 'occupancy_rate', 'rate_lower', 'rate_upper']
        result['occupancy_rate'] += offset
        result['rate_lower'] += offset
        result['rate_upper'] += offset


        for col in ['occupancy_rate', 'rate_lower', 'rate_upper']:
            result[col] = result[col].clip(0, 1)


        result.loc[0, 'occupancy_rate'] = current_occupancy
        result.loc[0, 'rate_lower'] = max(current_occupancy * 0.95, 0)
        result.loc[0, 'rate_upper'] = min(current_occupancy * 1.05, 1)

        if not forecast.empty:
            initial_forecast = forecast['yhat'].iloc[0]
            offset = current_occupancy - initial_forecast

            adjusted_forecast = forecast.copy()
            adjusted_forecast['yhat'] += offset


            transition_points = 18
            smoothing_factor = 0.2

            for i in range(1, min(transition_points, len(adjusted_forecast))):
                alpha = 1 - np.exp(-i / (transition_points * smoothing_factor))
                adjusted_forecast.loc[i, 'yhat'] = (1 - alpha) * current_occupancy + alpha * adjusted_forecast.loc[
                    i, 'yhat']


        result['total_spots'] = total_spots
        result['occupied_spots_predicted'] = (result['occupancy_rate'] * total_spots).round().astype(int)
        result['free_spots_predicted'] = total_spots - result['occupied_spots_predicted']


        current_state = pd.DataFrame({
            'datetime': [current_time],
            'occupancy_rate': [current_occupancy],
            'rate_lower': [max(current_occupancy * 0.95, 0)],
            'rate_upper': [min(current_occupancy * 1.05, 1)],
            'total_spots': [total_spots],
            'occupied_spots_predicted': [occupied_spots],
            'free_spots_predicted': [total_spots - occupied_spots],
            'is_current': [True]
        })

        result['is_current'] = False
        result = pd.concat([current_state, result], ignore_index=True)


        csv_path = os.path.join(PLOT_DIR, 'full_forecast_results.csv')
        temp_path = csv_path + '.tmp'

        max_attempts = 3
        for attempt in range(max_attempts):
            try:

                result.to_csv(temp_path, index=False)


                if os.path.exists(csv_path):
                    os.replace(temp_path, csv_path)
                else:
                    os.rename(temp_path, csv_path)

                logger.info(f"Результаты сохранены: {csv_path}")
                return

            except PermissionError as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Попытка {attempt + 1}: Ошибка доступа, повтор через 1 сек...")
                    time.sleep(1)
                else:
                    logger.error(f"Не удалось сохранить после {max_attempts} попыток: {str(e)}")

                    home_dir = os.path.expanduser('~')
                    backup_path = os.path.join(home_dir, 'parking_forecast_backup.csv')
                    result.to_csv(backup_path, index=False)
                    logger.warning(f"Создана резервная копия: {backup_path}")

    except Exception as e:
        logger.error(f"Ошибка сохранения: {str(e)}")

        try:
            home_dir = os.path.expanduser('~')
            backup_path = os.path.join(home_dir, 'parking_forecast_backup.csv')
            result.to_csv(backup_path, index=False)
            logger.warning(f"Создана резервная копия: {backup_path}")
        except Exception as backup_error:
            logger.error(f"Не удалось создать резервную копию: {str(backup_error)}")


def main():
    """Основная функция выполнения прогнозирования"""
    try:

        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info("=== Запуск системы прогнозирования ===")


        model = load_model()


        test_data = prepare_test_data()
        print(test_data[['occupied_spots', 'total_spots']].tail())  # Для проверки


        forecast = make_forecast(model, test_data)
        if forecast.empty:
            raise ValueError("Не удалось сгенерировать прогноз")
        logger.info(f"Создан прогноз на {len(forecast)} временных точек")


        plot_combined(test_data, forecast)


        save_full_results(test_data, forecast)

        logger.info("=== Прогнозирование успешно завершено ===")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise
    finally:
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)


if __name__ == "__main__":
    main()