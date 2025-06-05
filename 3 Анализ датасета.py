import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)

def load_data(directory):
    timestamps = []
    vehicle_counts = []

    for filename in os.listdir(directory):
        if filename.endswith('.txt') and 'photo' in filename:
            match = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})', filename)
            if match:
                try:
                    dt_str = f"{match.group(1)} {match.group(2).replace('-', ':')}"
                    timestamp = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

                    with open(os.path.join(directory, filename), 'r') as f:
                        lines = [line for line in f if not line.startswith('#')]
                        vehicle_count = len(lines)

                    timestamps.append(timestamp)
                    vehicle_counts.append(vehicle_count)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    df = pd.DataFrame({'Timestamp': timestamps, 'Vehicles': vehicle_counts})
    df.sort_values('Timestamp', inplace=True)
    return df


def analyze_data(df):
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['DayName'] = df['Timestamp'].dt.day_name()
    df['Date'] = df['Timestamp'].dt.date
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])  # 5=суббота, 6=воскресенье

    print("\n=== Общая статистика ===")
    print(df['Vehicles'].describe())

    print("\n=== Анализ по дням недели ===")
    day_stats = df.groupby('DayName')['Vehicles'].agg(['mean', 'median', 'std', 'count'])
    print(day_stats)

    print("\n=== Анализ по выходным/будням ===")
    weekend_stats = df.groupby('IsWeekend')['Vehicles'].agg(['mean', 'median', 'std', 'count'])
    print(weekend_stats)

    print("\n=== Анализ по часам (выходные vs будни) ===")
    hour_weekend_stats = df.groupby(['IsWeekend', 'Hour'])['Vehicles'].agg(['mean', 'median', 'std'])
    print(hour_weekend_stats)

    print("\n=== Кластерный анализ ===")
    X = df[['Hour', 'Vehicles']].values
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    df['Cluster'] = kmeans.labels_
    print(df.groupby('Cluster')['Vehicles'].describe())

    return df

def visualize_data(df):
    plt.figure(figsize=(18, 12))

    plt.subplot(3, 2, 1)
    plt.plot(df['Timestamp'], df['Vehicles'], marker='o', markersize=4)
    plt.title('Общая динамика количества автомобилей')
    plt.xticks(rotation=45)

    plt.subplot(3, 2, 2)
    df.groupby('DayName')['Vehicles'].mean().sort_values().plot(kind='bar')
    plt.title('Среднее количество по дням недели')

    plt.subplot(3, 2, 3)
    for weekend, group in df.groupby('IsWeekend'):
        label = 'Выходные' if weekend else 'Будни'
        group.groupby('Hour')['Vehicles'].mean().plot(
            kind='line', marker='o', label=label)
    plt.title('Среднее количество по часам')
    plt.xticks(range(24))
    plt.legend()

    plt.subplot(3, 2, 4)
    weekend = df[df['IsWeekend']].groupby('Hour')['Vehicles'].mean()
    weekday = df[~df['IsWeekend']].groupby('Hour')['Vehicles'].mean()
    (weekend - weekday).plot(kind='bar', color='orange')
    plt.title('Разница между выходными и буднями')
    plt.xticks(range(24))
    plt.axhline(0, color='black', linestyle='--')

    plt.subplot(3, 2, 5)
    for cluster in sorted(df['Cluster'].unique()):
        for weekend, marker in [(False, 'o'), (True, 's')]:
            subset = df[(df['Cluster'] == cluster) & (df['IsWeekend'] == weekend)]
            label = f'Cluster {cluster} ({"Вых" if weekend else "Буд"})'
            plt.scatter(subset['Hour'], subset['Vehicles'],
                        label=label, alpha=0.6, marker=marker)
    plt.legend()
    plt.title('Кластерный анализ по часам')
    plt.xticks(range(24))

    plt.tight_layout()
    plt.savefig('full_analysis_with_weekends.png')
    plt.close()

def time_series_analysis(df):
    try:
        df_weekend = df[df['IsWeekend']].set_index('Timestamp').asfreq('h')
        df_weekday = df[~df['IsWeekend']].set_index('Timestamp').asfreq('h')

        for df_ts in [df_weekend, df_weekday]:
            if df_ts['Vehicles'].isnull().any():
                df_ts['Vehicles'] = df_ts['Vehicles'].interpolate(method='linear')

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        decomposition = seasonal_decompose(df_weekday['Vehicles'], period=24, extrapolate_trend='freq')
        decomposition.plot()
        plt.title('Декомпозиция будних дней')

        plt.subplot(2, 1, 2)
        decomposition = seasonal_decompose(df_weekend['Vehicles'], period=24, extrapolate_trend='freq')
        decomposition.plot()
        plt.title('Декомпозиция выходных дней')

        plt.tight_layout()
        plt.savefig('time_series_decomposition_weekend.png')
        plt.close()

    except Exception as e:
        print(f"\nОшибка при анализе временных рядов: {e}")


def main():
    directory = r'F:\PythonProjects\pythonProject8\output'

    df = load_data(directory)
    if df.empty:
        print("Нет данных для анализа")
        return

    df = analyze_data(df)

    visualize_data(df)

     time_series_analysis(df)

    print("\nАнализ завершен. Результаты сохранены в PNG файлах.")


if __name__ == "__main__":
    main()