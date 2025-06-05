import numpy as np
from collections import defaultdict
from scipy.spatial.distance import euclidean
import folium
from datetime import datetime, timedelta
import time
import requests
import os
import re
from shapely.geometry import Polygon
import csv
from sklearn.cluster import DBSCAN
import math



GPS_POSITION = (61.285552, 73.347173, 8)
IMAGE_SIZE = (1080, 608)
WORLD_SIZE = (25.0, 7.0)

class CoordinateTransformer:
    def __init__(self):
        self.origin_lat, self.origin_lon = GPS_POSITION[:2]

    def get_vehicle_corners(self, row):
        """Получает координаты углов автомобиля из строки CSV"""
        try:
            corners = [
                [float(row['corner1_lat']), float(row['corner1_lon'])],
                [float(row['corner2_lat']), float(row['corner2_lon'])],
                [float(row['corner3_lat']), float(row['corner3_lon'])],
                [float(row['corner4_lat']), float(row['corner4_lon'])]
            ]
            return corners
        except (KeyError, ValueError) as e:
            print(f"Ошибка получения углов автомобиля: {e}")
            return None

    def get_vehicle_center(self, row):
        """Вычисляет центр автомобиля как среднее угловых координат"""
        corners = self.get_vehicle_corners(row)
        if corners is None:
            return None
        lats = [c[0] for c in corners]
        lons = [c[1] for c in corners]
        return np.mean(lats), np.mean(lons)

    def get_vehicle_corners_gps(self, row):
        """Возвращает GPS-координаты 4 углов автомобиля"""
        try:
            return [
                (float(row['corner1_lat']), float(row['corner1_lon'])),
                (float(row['corner2_lat']), float(row['corner2_lon'])),
                (float(row['corner3_lat']), float(row['corner3_lon'])),
                (float(row['corner4_lat']), float(row['corner4_lon']))
            ]
        except (KeyError, ValueError) as e:
            print(f"Ошибка получения углов: {e}")
            return None


class ParkingAnalyzer:
    def __init__(self, parking_spot_length=5.3, parking_spot_width=2.5, max_time_threshold=30,
                 output_csv="parking_data_UlKrylova.csv"):
        self.parking_spot_length = parking_spot_length
        self.parking_spot_width = parking_spot_width
        self.max_time_threshold = max_time_threshold * 60
        self.tracked_vehicles = defaultdict(list)
        self.parking_spots = []
        self.confirmed_spots = []
        self.occupied_spots = []
        self.free_spots = []
        self.parking_history = defaultdict(list)
        self.spot_counter = 1
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.output_csv = output_csv
        self.data = []
        self.spots_info = {}
        self.processed_timestamps = set()
        self.coord_transformer = CoordinateTransformer()
        self.origin_lat, self.origin_lon = GPS_POSITION[:2]
        self.load_data()

    def load_data(self):
        """Загрузка данных из CSV файла"""
        if not os.path.exists(self.output_csv):
            self._init_csv_file()
            return

        with open(self.output_csv, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return

            for row in reader:
                try:

                    record = {
                        'timestamp': int(float(row['timestamp'])),
                        'datetime': row['datetime'],
                        'spot_id': int(row['spot_id']),
                        'status': row['status'],
                        'vehicle_type': row['vehicle_type'],
                        'parking_duration': float(row['parking_duration']),
                        'spot_length': float(row['spot_length']),
                        'spot_width': float(row['spot_width']),
                        'orientation': float(row['orientation']),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'total_spots': int(row['total_spots']),
                        'free_spots': int(row['free_spots']),
                        'occupied_spots': int(row['occupied_spots'])
                    }
                    self.data.append(record)


                    spot_id = record['spot_id']
                    if spot_id not in self.spots_info:
                        self.spots_info[spot_id] = {
                            'id': spot_id,
                            'length': record['spot_length'],
                            'width': record['spot_width'],
                            'orientation': record['orientation'],
                            'vehicle_type': record['vehicle_type'],
                            'latitude': record['latitude'],
                            'longitude': record['longitude']
                        }
                except (ValueError, KeyError) as e:
                    print(f"Ошибка обработки строки: {row}. Ошибка: {str(e)}")
                    continue

    def get_current_status(self, timestamp=None):
        """Получение текущего статуса парковки"""
        if not self.data:
            return 0, [], []

        if timestamp is None:

            last_record = self.data[-1]
            timestamp = last_record['timestamp']


        current_records = [r for r in self.data if r['timestamp'] == timestamp]
        if not current_records:
            return 0, [], []

        total_spots = current_records[0]['total_spots']
        occupied = [r for r in current_records if r['status'] == 'occupied']
        free = [r for r in current_records if r['status'] == 'free']

        return total_spots, occupied, free

    def prepare_history_for_analysis(self):
        """Подготовка данных для анализа без pandas"""
        history = {}

        for record in self.data:
            timestamp = record['timestamp']
            if timestamp not in history:
                history[timestamp] = {
                    'timestamp': timestamp,
                    'datetime': record['datetime'],
                    'total_spots': record['total_spots'],
                    'occupied_spots': 0,
                    'free_spots': 0
                }

            if record['status'] == 'occupied':
                history[timestamp]['occupied_spots'] += 1
            else:
                history[timestamp]['free_spots'] += 1


        result = []
        for ts, data in history.items():
            data['occupancy_rate'] = data['occupied_spots'] / data['total_spots'] if data['total_spots'] > 0 else 0
            result.append(data)

        return sorted(result, key=lambda x: x['timestamp'])

    def _init_csv_file(self):
        """Инициализирует CSV файл с заголовками, если он не существует"""
        headers = self._get_csv_headers()


        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        self.data = []

    def _get_csv_headers(self):
        return [
            'timestamp',
            'datetime',
            'spot_id',
            'status',
            'vehicle_type',
            'parking_duration',
            'spot_length',
            'spot_width',
            'orientation',
            'latitude',
            'longitude',
            'total_spots',
            'free_spots',
            'occupied_spots',
            'spot_density',
            'hour',
            'day_of_week',
            'is_weekend',
            'is_perpendicular',
            'is_parallel',
            'corner1_lat', 'corner1_lon',
            'corner2_lat', 'corner2_lon',
            'corner3_lat', 'corner3_lon',
            'corner4_lat', 'corner4_lon',
            'image_filename',
            'vehicle_id'
        ]

    def _save_to_csv(self, timestamp, spots):
        """Сохраняет данные о парковочных местах в CSV файл"""
        write_header = not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0
        datetime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        total_spots = len(spots)
        free_spots = len([s for s in spots if not s['occupied']])
        occupied_spots = total_spots - free_spots
        density = self._calculate_spot_density(spots)

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = int(day_of_week >= 5)

        recorded_spots = set()
        with open(self.output_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_headers())

            if write_header:
                writer.writeheader()


            for spot in spots:
                spot_key = (timestamp, spot['id'])
                if spot_key in recorded_spots:
                    continue
                recorded_spots.add(spot_key)

                corners = spot.get('corners_latlon', [])
                if len(corners) != 4:
                    corners = [(0, 0)] * 4

                orientation = spot.get('orientation', 0)
                is_perpendicular = int(abs(orientation) > 0.8)
                is_parallel = int(abs(orientation) <= 0.8)

                row = {
                    'timestamp': timestamp,
                    'datetime': datetime_str,
                    'spot_id': spot['id'],
                    'status': 'occupied' if spot['occupied'] else 'free',
                    'vehicle_type': spot.get('vehicle_type', 'car'),
                    'parking_duration': spot.get('parking_duration', 0),
                    'spot_length': spot.get('length', self.parking_spot_length),
                    'spot_width': spot.get('width', self.parking_spot_width),
                    'orientation': orientation,
                    'latitude': spot['center_latlon'][0],
                    'longitude': spot['center_latlon'][1],
                    'total_spots': total_spots,
                    'free_spots': free_spots,
                    'occupied_spots': occupied_spots,
                    'spot_density': density,
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'is_perpendicular': is_perpendicular,
                    'is_parallel': is_parallel,
                    'corner1_lat': corners[0][0],
                    'corner1_lon': corners[0][1],
                    'corner2_lat': corners[1][0],
                    'corner2_lon': corners[1][1],
                    'corner3_lat': corners[2][0],
                    'corner3_lon': corners[2][1],
                    'corner4_lat': corners[3][0],
                    'corner4_lon': corners[3][1],
                    'image_filename': self._get_image_filename(timestamp),
                    'vehicle_id': spot.get('vehicle_id', 'unknown')
                }
                writer.writerow(row)
                self.data.append(row)


    def _calculate_vehicle_orientation(self, bbox):
        side1 = euclidean(bbox[0:2], bbox[2:4])
        side2 = euclidean(bbox[2:4], bbox[4:6])
        if side1 > side2:
            dx = bbox[2] - bbox[0]
            dy = bbox[3] - bbox[1]
        else:
            dx = bbox[4] - bbox[2]
            dy = bbox[5] - bbox[3]
        return np.arctan2(dy, dx)

    def _is_same_vehicle(self, vehicle1, vehicle2, position_threshold=1.5, orientation_threshold=0.5):
        center1 = np.mean(np.array(vehicle1['bbox']).reshape(4, 2), axis=0)
        center2 = np.mean(np.array(vehicle2['bbox']).reshape(4, 2), axis=0)
        if euclidean(center1, center2) > position_threshold:
            return False
        if abs(vehicle1['orientation'] - vehicle2['orientation']) > orientation_threshold:
            return False
        return True

    def _is_on_road(self, gps_coords):
        lat, lon = gps_coords
        overpass_query = f"""
        [out:json];
        way(around:10,{lat},{lon})["highway"];
        (._;>;);
        out body;
        """
        try:
            response = requests.post(self.overpass_url, data=overpass_query)
            data = response.json()
            return len(data['elements']) > 0
        except:
            return False

    def _is_point_in_polygon(self, point, polygon_points):
        """Улучшенная реализация алгоритма трассировки лучей для проверки нахождения точки внутри полигона"""
        if polygon_points is None or len(polygon_points) < 3:
            return False

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

    def _check_parking_conditions(self, vehicle_history):
        total_time = vehicle_history[-1]['timestamp'] - vehicle_history[0]['timestamp']
        if total_time >= 200:
            return True

        parking_events = 0
        current_stay_start = None
        min_stay_duration = 30 * 60

        for i in range(1, len(vehicle_history)):
            time_diff = vehicle_history[i]['timestamp'] - vehicle_history[i - 1]['timestamp']
            pos_diff = euclidean(
                np.mean(np.array(vehicle_history[i]['bbox']).reshape(4, 2), axis=0),
                np.mean(np.array(vehicle_history[i - 1]['bbox']).reshape(4, 2), axis=0)
            )

            if pos_diff < 0.5:
                if current_stay_start is None:
                    current_stay_start = vehicle_history[i - 1]['timestamp']
            else:
                if current_stay_start is not None:
                    stay_duration = vehicle_history[i - 1]['timestamp'] - current_stay_start
                    if stay_duration >= min_stay_duration:
                        parking_events += 1
                    current_stay_start = None

        if current_stay_start is not None:
            stay_duration = vehicle_history[-1]['timestamp'] - current_stay_start
            if stay_duration >= min_stay_duration:
                parking_events += 1

        return parking_events >= 12

    def _calculate_spot_density(self, spots):
        """Вычисляет плотность парковочных мест (мест/м²) на основе их координат.

        Args:
            spots: Список парковочных мест (из self.parking_spots или self.confirmed_spots)

        Returns:
            float: Плотность в местах на квадратный метр
        """
        if len(spots) < 2:
            return 0.0

        try:

            coords = np.array([spot['center_latlon'] for spot in spots if 'center_latlon' in spot])


            lat_mean = np.mean(coords[:, 0])
            kms_per_degree = 111.32 * 1000
            x = coords[:, 1] * kms_per_degree * np.cos(np.radians(lat_mean))  # Долгота -> метры
            y = coords[:, 0] * kms_per_degree  # Широта -> метры


            from scipy.spatial import ConvexHull
            points = np.column_stack((x, y))
            hull = ConvexHull(points)
            area = hull.volume

            return len(spots) / max(area, 1e-6)
        except Exception as e:
            print(f"Ошибка вычисления плотности: {e}")
            return 0.0

    def _get_image_filename(self, timestamp):
        """Генерирует правильное имя файла изображения по timestamp"""
        dt = datetime.fromtimestamp(timestamp)
        filename = dt.strftime("%Y-%m-%d_%H-%M-%S") + "_photo.jpg"
        return os.path.join(r"F:\PythonProjects\pythonProject8\KrylovaDataset\UlKrylovaDataset", filename)


    def _estimate_spot_corners(self, vehicle_data):
        """Оценивает углы парковочного места, если нет точных данных"""
        center = np.mean(np.array(vehicle_data['bbox']).reshape(4, 2), axis=0)
        orientation = vehicle_data['orientation']
        length = self.parking_spot_length
        width = self.parking_spot_width


        half_l = length / 2 / 0.2
        half_w = width / 2 / 0.2

        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)

        corners = []
        for dx, dy in [(-half_l, -half_w), (-half_l, half_w),
                       (half_l, half_w), (half_l, -half_w)]:
            x = dx * cos_a - dy * sin_a + center[0]
            y = dx * sin_a + dy * cos_a + center[1]
            lat, lon = self.coord_transformer.pixel_to_latlon(x, y)
            corners.append((lat, lon))

        return corners

    def process_frame(self, vehicles, timestamp):
        if timestamp in self.processed_timestamps:
            return
        self.processed_timestamps.add(timestamp)

        for vehicle in vehicles:
            if 'row_data' in vehicle:
                vehicle['vehicle_id'] = vehicle['row_data']['vehicle_id']

        current_vehicle_ids = {v['row_data']['vehicle_id'] for v in vehicles if 'row_data' in v}


        for vehicle in vehicles:
            try:
                if 'row_data' not in vehicle:
                    continue

                vehicle_id = vehicle['row_data']['vehicle_id']
                orientation = self._calculate_vehicle_orientation(vehicle['bbox'])

                current_vehicle = {
                    'vehicle_id': vehicle_id,
                    'bbox': vehicle['bbox'],
                    'orientation': orientation,
                    'timestamp': timestamp,
                    'vehicle_type': vehicle.get('vehicle_type', 'car'),
                    'row_data': vehicle['row_data']
                }

                self.tracked_vehicles[vehicle_id].append(current_vehicle)

                if self._check_parking_conditions(self.tracked_vehicles[vehicle_id]):
                    self._create_parking_spot(vehicle_id)

            except Exception as e:
                print(f"Ошибка обработки vehicle: {e}")
                continue


        departed_vehicles = set(self.tracked_vehicles.keys()) - current_vehicle_ids
        for vehicle_id in departed_vehicles:
            history = self.tracked_vehicles[vehicle_id]
            if self._check_parking_conditions(history):
                self._create_parking_spot(vehicle_id)


        self._update_spot_occupancy(vehicles, timestamp)


        all_spots = self.parking_spots + self.confirmed_spots
        if all_spots:
            self._save_to_csv(timestamp, all_spots)

    def process_frame_batch(self, vehicles_batch, timestamps_batch):
        """Обрабатывает батч аннотаций"""
        for i, (vehicles, timestamp) in enumerate(zip(vehicles_batch, timestamps_batch)):

            if timestamp in self.processed_timestamps:
                print(f"Кадр с timestamp {timestamp} уже обработан, пропускаем")
                continue
            self.processed_timestamps.add(timestamp)

            print(
                f"=== Обработка аннотации {i + 1}/{len(vehicles_batch)} с временной меткой: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))} ===")


            current_vehicle_ids = {vehicle['row_data']['vehicle_id'] for vehicle in vehicles if 'row_data' in vehicle}

            for vehicle in vehicles:
                try:
                    if 'row_data' not in vehicle:
                        continue

                    vehicle_id = vehicle['row_data']['vehicle_id']
                    orientation = self._calculate_vehicle_orientation(vehicle['bbox'])

                    current_vehicle = {
                        'vehicle_id': vehicle_id,
                        'bbox': vehicle['bbox'],
                        'orientation': orientation,
                        'timestamp': timestamp,
                        'vehicle_type': vehicle.get('vehicle_type', 'car'),
                        'row_data': vehicle['row_data']
                    }

                    self.tracked_vehicles[vehicle_id].append(current_vehicle)

                    if self._check_parking_conditions(self.tracked_vehicles[vehicle_id]):
                        self._create_parking_spot(vehicle_id)

                except Exception as e:
                    print(f"Ошибка обработки vehicle: {e}")
                    continue


            all_known_vehicles = set(self.tracked_vehicles.keys())
            departed_vehicles = all_known_vehicles - current_vehicle_ids

            for vehicle_id in departed_vehicles:
                history = self.tracked_vehicles[vehicle_id]
                if not history:
                    continue

                if self._check_parking_conditions(history):
                    self._create_parking_spot(vehicle_id)

            self._update_spot_occupancy(vehicles, timestamp)

            # Сохраняем данные в CSV
            all_spots = self.parking_spots + self.confirmed_spots
            if all_spots:
                self._save_to_csv(timestamp, all_spots)


    def _create_parking_spot(self, vehicle_id):
        history = self.tracked_vehicles.get(vehicle_id, [])
        if len(history) < 2:
            return


        corners = None
        if 'row_data' in history[-1]:
            corners = self.coord_transformer.get_vehicle_corners_gps(history[-1]['row_data'])

        parking_duration = history[-1]['timestamp'] - history[0]['timestamp']
        if parking_duration < 10 * 1800:
            return

        last_vehicle = history[-1]


        if 'row_data' in last_vehicle:
            center = self.coord_transformer.get_vehicle_center(last_vehicle['row_data'])
            if center is None:
                return
            center_lat, center_lon = center
        else:

            center = np.mean(np.array(last_vehicle['bbox']).reshape(4, 2), axis=0)
            center_lat, center_lon = self.coord_transformer.pixel_to_latlon(*center)

        if 'row_data' in last_vehicle:
            spot_corners = self.coord_transformer.get_vehicle_corners(last_vehicle['row_data'])
            if spot_corners is None:
                return
        else:
            spot_bbox = self._get_spot_bbox({
                'center': center,
                'orientation': last_vehicle['orientation'],
                'length': self.parking_spot_length if last_vehicle.get('vehicle_type') != 'truck' else 7.5,
                'width': self.parking_spot_width if last_vehicle.get('vehicle_type') != 'truck' else 2.8
            })
            spot_corners = [[spot_bbox[i], spot_bbox[i + 1]] for i in range(0, len(spot_bbox), 2)]


        new_spot_poly = Polygon(spot_corners)
        for spot in self.confirmed_spots + self.parking_spots:
            if 'corners_latlon' in spot:
                spot_poly = Polygon(spot['corners_latlon'])
            else:
                spot_bbox = self._get_spot_bbox(spot)
                spot_corners = [[spot_bbox[i], spot_bbox[i + 1]] for i in range(0, len(spot_bbox), 2)]
                spot_poly = Polygon(spot_corners)

            iou = new_spot_poly.intersection(spot_poly).area / new_spot_poly.union(spot_poly).area
            if iou > 0.3:
                return

        hours, remainder = divmod(int(parking_duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        parking_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        last_used = history[-1]['timestamp']
        last_used_str = time.strftime("%H:%M:%S", time.localtime(last_used))

        vehicle_type = last_vehicle.get('vehicle_type', 'car')
        if vehicle_type == 'truck':
            length, width = 7.5, 2.8
        else:
            length, width = self.parking_spot_length, self.parking_spot_width

        new_spot = {
            'id': self.spot_counter,
            'center_latlon': (center_lat, center_lon),
            'corners_latlon': spot_corners,
            'orientation': last_vehicle['orientation'],
            'length': length,
            'width': width,
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,
            'parking_time': parking_time_str,
            'parking_duration': parking_duration,
            'last_used': last_used,
            'last_used_str': last_used_str,
            'created_at': time.time(),
            'occupied': False,
            'confirmed': False,
            'corners_latlon': corners if corners else self._estimate_spot_corners(history[-1]),
            'id': self.spot_counter,
            'vehicle_id': vehicle_id
        }

        self.parking_spots.append(new_spot)
        self.free_spots.append(new_spot)
        self.spot_counter += 1

        print(f"Создано место {new_spot['id']} от {vehicle_type} ID={vehicle_id}, уехал в: {last_used_str}")

    def _update_spot_occupancy(self, vehicles, timestamp):
        """Обновляет статус занятости парковочных мест с улучшенной логикой"""
        all_spots = self.parking_spots + self.confirmed_spots
        self.occupied_spots = []
        self.free_spots = []

        current_vehicle_info = []
        for vehicle in vehicles:
            try:

                center_pixel = np.mean(np.array(vehicle['bbox']).reshape(4, 2), axis=0)
                orientation = self._calculate_vehicle_orientation(vehicle['bbox'])


                if 'row_data' in vehicle:
                    corners_gps = self.coord_transformer.get_vehicle_corners(vehicle['row_data'])
                    center_gps = self.coord_transformer.get_vehicle_center(vehicle['row_data'])

                    vehicle_id = vehicle['row_data'].get('vehicle_id', 'none')
                else:
                    corners_gps = None
                    center_gps = self.coord_transformer.pixel_to_latlon(*center_pixel)

                    vehicle_id = 'none'

                current_vehicle_info.append({
                    'center_pixel': center_pixel,
                    'center_gps': center_gps,
                    'orientation': orientation,
                    'bbox': vehicle['bbox'],
                    'vehicle_type': vehicle.get('vehicle_type', 'car'),
                    'polygon_gps': corners_gps if corners_gps else None,
                    'polygon_pixel': np.array(vehicle['bbox']).reshape(4, 2).tolist(),
                    'row_data': vehicle.get('row_data'),

                    'vehicle_id': vehicle_id
                })
            except Exception as e:
                print(f"Ошибка обработки vehicle: {e}")
                continue

        for spot in all_spots:
            is_occupied = False

            occupying_vehicle_id = 'none'


            spot_polygon = spot.get('corners_latlon')
            if spot_polygon is None:
                temp_spot = {
                    'center': spot.get('center', [0, 0]),
                    'orientation': spot.get('orientation', 0),
                    'length': spot.get('length', self.parking_spot_length),
                    'width': spot.get('width', self.parking_spot_width)
                }
                spot_bbox = self._get_spot_bbox(temp_spot)
                spot_polygon = [[spot_bbox[i], spot_bbox[i + 1]] for i in range(0, len(spot_bbox), 2)]

            spot_center = spot.get('center_latlon', (0, 0))


            best_match = None
            max_iou = 0.0

            for vehicle in current_vehicle_info:
                try:

                    if vehicle['center_gps'] and self._is_point_in_polygon(vehicle['center_gps'], spot_polygon):

                        if vehicle['polygon_gps']:
                            vehicle_poly = Polygon(vehicle['polygon_gps'])
                            spot_poly = Polygon(spot_polygon)
                            iou = spot_poly.intersection(vehicle_poly).area / spot_poly.union(vehicle_poly).area

                            if iou > max_iou:
                                max_iou = iou
                                best_match = vehicle
                        else:
                            if best_match is None:
                                best_match = vehicle


                    if vehicle['polygon_gps'] and self._is_point_in_polygon(spot_center, vehicle['polygon_gps']):
                        if best_match is None:  # Если еще не нашли лучший вариант
                            best_match = vehicle

                except Exception as e:
                    print(f"Ошибка проверки занятости: {e}")
                    continue


            if best_match:
                is_occupied = True
                occupying_vehicle_id = best_match['vehicle_id']
                spot['last_vehicle_type'] = best_match['vehicle_type']

            spot['occupied'] = is_occupied

            spot['vehicle_id'] = occupying_vehicle_id

            if is_occupied:
                self.occupied_spots.append(spot)
                spot['last_used'] = timestamp
                if not spot['confirmed'] and spot in self.parking_spots:
                    spot['confirmed'] = True
                    self.confirmed_spots.append(spot)
                    self.parking_spots.remove(spot)
                    print(f"Место {spot['id']} подтверждено как парковочное (авто {occupying_vehicle_id})")
            else:
                self.free_spots.append(spot)

        print(
            f"Статистика после обработки: Занято мест: {len(self.occupied_spots)}, Свободно мест: {len(self.free_spots)}")


    def _get_spot_bbox(self, spot):
        """Возвращает bbox парковочного места в пикселях"""
        try:
            angle = spot['orientation']
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            half_l = spot['length'] / 2 / 0.2  # convert meters to pixels
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
                ry = x * sin_a + y * cos_a + spot['center_latlon'][1]
                rotated.append([rx, ry])


            bbox = []
            for point in rotated:
                bbox.extend(point)
            return bbox
        except Exception as e:
            print(f"Ошибка в _get_spot_bbox: {e}")
            return [0, 0, 0, 0, 0, 0, 0, 0]

    def _calculate_iou(self, bbox1, bbox2):
        """Вычисляет IoU (Intersection over Union) двух полигонов"""
        poly1 = Polygon(np.array(bbox1).reshape(4, 2))
        poly2 = Polygon(np.array(bbox2).reshape(4, 2))

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        return intersection / union if union > 0 else 0.0

    def get_parking_spots(self):
        return self.parking_spots + self.confirmed_spots

    def get_occupied_spots(self):
        return self.occupied_spots

    def create_generalized_parking_map(self, output_file="generalized_parking_map.html"):
        """Создает обобщенную карту с исправленной кластеризацией"""
        all_spots = self.parking_spots + self.confirmed_spots

        if not all_spots:
            print("Нет данных для создания обобщенной карты")
            return None


        coords = []
        orientations = []
        for spot in all_spots:
            lat, lon = spot['center_latlon']

            x = (lon - self.coord_transformer.origin_lon) * 111320 * math.cos(
                math.radians(self.coord_transformer.origin_lat))
            y = (lat - self.coord_transformer.origin_lat) * 111320
            coords.append([x, y])
            orientations.append(spot['orientation'])

        coords = np.array(coords)


        eps_meters = 2.5
        db = DBSCAN(eps=eps_meters, min_samples=3).fit(coords)
        labels = db.labels_


        generalized_spots = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_spots = [all_spots[i] for i in cluster_indices]


            avg_center = np.mean([coords[i] for i in cluster_indices], axis=0)
            avg_lon = self.origin_lon + avg_center[0] / (111320 * math.cos(math.radians(self.origin_lat)))
            avg_lat = self.origin_lat + avg_center[1] / 111320


            orientation_samples = [orientations[i] for i in cluster_indices]
            dominant_orientation = np.median(orientation_samples)


            lengths = [s['length'] for s in cluster_spots]
            widths = [s['width'] for s in cluster_spots]
            median_length = np.median(lengths)
            median_width = np.median(widths)


            types = [s['vehicle_type'] for s in cluster_spots]
            dominant_type = max(set(types), key=types.count)


            corners = self._calculate_rotated_rectangle(
                (avg_lat, avg_lon),
                median_length,
                median_width,
                dominant_orientation
            )

            generalized_spots.append({
                'id': cluster_id,
                'center': (avg_lat, avg_lon),
                'orientation': dominant_orientation,
                'vehicle_type': dominant_type,
                'count': len(cluster_spots),
                'polygon': corners
            })


        m = folium.Map(location=GPS_POSITION[:2], zoom_start=18)


        for spot in generalized_spots:
            folium.Polygon(
                locations=spot['polygon'],
                color='blue' if spot['vehicle_type'] == 'truck' else 'green',
                fill=True,
                fill_opacity=0.6,
                weight=2,
                popup=f"Кластер {spot['id']}: {spot['count']} мест\nТип: {spot['vehicle_type']}"
            ).add_to(m)


        original_layer = folium.FeatureGroup(name='Оригинальные места', show=False)
        for spot in all_spots:
            if 'corners_latlon' in spot:
                folium.Polygon(
                    locations=spot['corners_latlon'],
                    color='gray',
                    fill=True,
                    fill_opacity=0.2,
                    weight=1
                ).add_to(original_layer)
        m.add_child(original_layer)

        m.save(output_file)
        return m

    def _calculate_rotated_rectangle(self, center, length, width, angle_deg):
        """Создает полигон с правильной ориентацией"""
        angle_rad = math.radians(angle_deg)
        half_l = length / 2 / 111320
        half_w = width / 2 / (111320 * math.cos(math.radians(center[0])))


        corners = [
            [-half_w, -half_l],
            [half_w, -half_l],
            [half_w, half_l],
            [-half_w, half_l]
        ]


        rotated = []
        for dx, dy in corners:
            x_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
            y_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
            lat = center[0] + y_rot
            lon = center[1] + x_rot / math.cos(math.radians(center[0]))
            rotated.append([lat, lon])

        return rotated

    def _calculate_typical_corners(self, center, orientation, vehicle_type):
        """Вычисляет углы типичного парковочного места"""
        length = 7.5 if vehicle_type == 'truck' else 5.3
        width = 2.8 if vehicle_type == 'truck' else 2.5


        kms_per_degree = 111.32 * 1000
        length_deg = length / (kms_per_degree * np.cos(np.radians(center[0])))
        width_deg = width / kms_per_degree

        half_l = length_deg / 2
        half_w = width_deg / 2


        corners = [
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, half_w],
            [half_l, -half_w]
        ]


        cos_a = np.cos(orientation)
        sin_a = np.sin(orientation)

        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a + center[1]
            ry = x * sin_a + y * cos_a + center[0]
            rotated_corners.append([ry, rx])

        return rotated_corners



    def get_free_spots(self):
        return self.free_spots


def create_parking_map(analyzer, parking_spots, gps_origin, image_size=IMAGE_SIZE):
    """Создает интерактивную карту с парковочными местами"""
    m = folium.Map(location=gps_origin, zoom_start=18)


    folium.Marker(
        location=gps_origin,
        popup="Камера",
        icon=folium.Icon(color='blue', icon='video', prefix='fa')
    ).add_to(m)


    free_group = folium.FeatureGroup(name='Свободные')
    occupied_group = folium.FeatureGroup(name='Занятые')
    m.add_child(free_group)
    m.add_child(occupied_group)

    for spot in parking_spots:

        if 'corners_latlon' in spot:
            corners_gps = spot['corners_latlon']
        else:

            spot_bbox = analyzer._get_spot_bbox(spot)
            corners_gps = []
            for i in range(0, len(spot_bbox), 2):
                lat, lon = analyzer.coord_transformer.pixel_to_latlon(spot_bbox[i], spot_bbox[i + 1])
                corners_gps.append([lat, lon])


        center_lat, center_lon = spot['center_latlon']


        group = occupied_group if spot['occupied'] else free_group
        folium.Polygon(
            locations=corners_gps,
            color='red' if spot['occupied'] else 'green',
            fill=True,
            fill_opacity=0.2,
            popup=f"Место {spot['id']} ({spot.get('vehicle_type', 'car')})"
        ).add_to(group)


        folium.Marker(
            location=[center_lat, center_lon],
            icon=folium.Icon(
                color='red' if spot['occupied'] else 'green',
                icon='truck' if spot.get('vehicle_type') == 'truck' else 'car',
                prefix='fa'
            )
        ).add_to(group)

    folium.LayerControl().add_to(m)
    return m


def sort_csv_by_timestamp(input_file="parking_data_UlKrylova.csv", output_file=None):
    """Сортирует CSV-файл по столбцу timestamp с обработкой ошибок"""
    if output_file is None:
        output_file = input_file

    try:
        with open(input_file, mode='r', encoding='utf-8') as f:

            lines = f.readlines()

            if not lines:
                print("Файл пуст, сортировка не требуется")
                return


            header = lines[0].strip()
            data_lines = lines[1:]


            def get_timestamp(line):
                try:

                    if ':' in line and '=' in line:
                        ts_part = line.split("'timestamp':")[1].split(',')[0].strip(" '")
                    else:
                        ts_part = line.split(',')[0]
                    return int(float(ts_part))
                except (ValueError, IndexError) as e:
                    print(f"Ошибка обработки timestamp в строке: {line[:50]}... Ошибка: {e}")
                    return 0


            sorted_lines = sorted(data_lines, key=get_timestamp)


        with open(output_file, mode='w', encoding='utf-8') as f:
            f.write(header + '\n')
            f.writelines(sorted_lines)

        print(f"Файл успешно отсортирован и сохранен в {output_file}")

    except Exception as e:
        print(f"Критическая ошибка при сортировке: {str(e)}")
        raise


def load_detections_from_csv(csv_path):
    """Загружает детекции из CSV с обработкой ошибок"""
    detections = defaultdict(list)

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                filename = row['image']

                time_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
                if not time_match:
                    print(f"Не удалось извлечь время из: {filename}")
                    continue

                time_str = time_match.group(1)
                try:
                    timestamp = int(datetime.strptime(time_str, "%Y-%m-%d_%H-%M-%S").timestamp())
                except ValueError as e:
                    print(f"Ошибка формата времени {time_str}: {str(e)}")
                    continue


                try:
                    x = float(row['bbox_x'])
                    y = float(row['bbox_y'])
                    w = float(row['bbox_w'])
                    h = float(row['bbox_h'])
                    bbox = [x, y, x + w, y, x + w, y + h, x, y + h]
                    vehicle_id = int(row['vehicle_id'])
                except (ValueError, KeyError) as e:
                    print(f"Ошибка обработки bbox или vehicle_id: {e}")
                    continue


                try:
                    vehicle_type = 'truck' if int(row['class']) == 1 else 'car'
                except (ValueError, KeyError):
                    vehicle_type = 'car'

                detections[timestamp].append({
                    'bbox': bbox,
                    'vehicle_type': vehicle_type,
                    'row_data': row
                })

            except Exception as e:
                print(f"Ошибка обработки строки: {str(e)}")
                continue

    return detections





if __name__ == "__main__":
    analyzer = ParkingAnalyzer()


    csv_path = r"F:\PythonProjects\pythonProject8\output\detection_results_UlKrylova.csv"
    detections = load_detections_from_csv(csv_path)


    sorted_detections = sorted(detections.items(), key=lambda x: x[0])


    BATCH_SIZE = 30


    for i in range(0, len(sorted_detections), BATCH_SIZE):

        batch = sorted_detections[i:i + BATCH_SIZE]


        timestamps = [item[0] for item in batch]
        vehicles_list = [item[1] for item in batch]


        analyzer.process_frame_batch(vehicles_list, timestamps)


        time.sleep(0.5)


        print(f"Обработано {min(i + BATCH_SIZE, len(sorted_detections))}/{len(sorted_detections)} кадров")


    parking_spots = analyzer.get_parking_spots()
    m = create_parking_map(analyzer, parking_spots, GPS_POSITION[:2])
    m.save('parking_map_UlKrylova.html')


    occupied_spots = analyzer.get_occupied_spots()
    free_spots = analyzer.get_free_spots()
    print(f"\nИтоговая статистика:")
    print(f"Всего парковочных мест: {len(parking_spots)}")
    print(f"Занято мест: {len(occupied_spots)}")
    print(f"Свободно мест: {len(free_spots)}")


    sort_csv_by_timestamp()