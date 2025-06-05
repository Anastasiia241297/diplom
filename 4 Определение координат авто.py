import os
import cv2
import numpy as np
import folium
import pandas as pd
from ultralytics import YOLO
import math
import re
import pytesseract
import gc
import torch
from tqdm import tqdm
import time

pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\tesseract.exe'


class VehicleTracker:
    def __init__(self):
        self.tracks = dict()
        self.current_id = 1
        self.max_distance = 0.2
        self.max_area_diff = 0.2
        self.max_history = 5
        self.max_age_seconds = 60

    def update(self, current_vehicles, timestamp):
        updated_vehicles = []
        used_ids = set()

        for track_id in list(self.tracks.keys()):
            if len(self.tracks[track_id]['positions']) > self.max_history:
                self.tracks[track_id]['positions'] = self.tracks[track_id]['positions'][-self.max_history:]
                self.tracks[track_id]['areas'] = self.tracks[track_id]['areas'][-self.max_history:]

        for current_vehicle in current_vehicles:
            best_match = None
            min_distance = float('inf')

            for track_id, track_data in self.tracks.items():
                try:
                    last_pos = track_data['positions'][-1]
                    last_area = track_data['areas'][-1]

                    distance = math.hypot(
                        current_vehicle['world_coords'][0] - last_pos[0],
                        current_vehicle['world_coords'][1] - last_pos[1]
                    )
                    area_diff = abs(current_vehicle['bbox'][2] * current_vehicle['bbox'][3] - last_area) / last_area


                    if distance < self.max_distance and area_diff < self.max_area_diff:
                        if distance < min_distance:
                            min_distance = distance
                            best_match = track_id
                except KeyError:
                    continue


            if best_match is not None:
                current_vehicle['id'] = best_match

                if 'vehicle_class' in self.tracks[best_match]:
                    current_vehicle['class'] = self.tracks[best_match]['vehicle_class']
                self.tracks[best_match]['positions'].append(current_vehicle['world_coords'])
                self.tracks[best_match]['areas'].append(current_vehicle['bbox'][2] * current_vehicle['bbox'][3])
                self.tracks[best_match]['last_update'] = time.time()
                used_ids.add(best_match)
                current_vehicle['is_same_car'] = True
            else:

                current_vehicle['id'] = self.current_id
                self.tracks[self.current_id] = {
                    'positions': [current_vehicle['world_coords']],
                    'areas': [current_vehicle['bbox'][2] * current_vehicle['bbox'][3]],
                    'last_update': time.time(),
                    'vehicle_class': current_vehicle['class']
                }
                used_ids.add(self.current_id)
                current_vehicle['is_same_car'] = False
                self.current_id += 1

            updated_vehicles.append(current_vehicle)


        self._remove_unused_tracks(used_ids)


        self._clear_old_tracks()

        return updated_vehicles

    def _remove_unused_tracks(self, used_ids):
        """Удаляет треки, которые не были обновлены в текущем кадре"""
        self.tracks = {k: v for k, v in self.tracks.items() if k in used_ids}

    def _clear_old_tracks(self):
        """Очищает треки, не обновлявшиеся более max_age_seconds секунд"""
        current_time = time.time()
        for track_id in list(self.tracks.keys()):
            if current_time - self.tracks[track_id]['last_update'] > self.max_age_seconds:
                del self.tracks[track_id]


def save_detection_results(output_folder, results_data):
    """Сохраняет результаты детекции в CSV файл"""
    csv_path = os.path.join(output_folder, "detection_results_UlKrylova.csv")

    file_exists = os.path.exists(csv_path)
    df = pd.DataFrame(results_data)

    if file_exists:
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.drop_duplicates(subset=['image', 'vehicle_id'], keep='last', inplace=True)
        updated_df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"Результаты сохранены в: {csv_path}")


def clean_filename(text):
    """Очищает строку для использования в имени файла"""
    return re.sub(r'[<>:"/\\|?*]', '', text).strip()


def extract_and_format_timestamp(image_path):
    """Извлекает временную метку с изображения"""
    image = cv2.imread(image_path)
    cropped_image = image[0:100, 0:500]
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    raw_text = pytesseract.image_to_string(gray_image, config='--psm 6')

    timestamp_pattern = r'(\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2})'
    match = re.search(timestamp_pattern, raw_text)

    if match:
        timestamp = match.group(1)
        return f"# Timestamp: {timestamp}", timestamp.replace('/', '-').replace(' ', '_').replace(':', '-')
    else:
        cleaned_text = re.sub(r'[^\d/: ]', '', raw_text).strip()
        parts = cleaned_text.split()
        if len(parts) >= 2:
            timestamp = ' '.join(parts[:2])
            return f"# Timestamp: {timestamp}", timestamp.replace('/', '-').replace(' ', '_').replace(':', '-')
        else:
            timestamp = cleaned_text if cleaned_text else "Not_recognized"
            return f"# Timestamp: {timestamp}", clean_filename(timestamp)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_CLAHE(image, clipLimit=2.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def enhance_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.8, edges_bgr, 0.2, 0)


def apply_nms(boxes, scores, iou_threshold=0.45):
    """Применяем Non-Maximum Suppression для устранения дублирующих боксов"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def image_to_world(x, y, H_matrix):
    pt = np.array([[x, y]], dtype=np.float32)
    pt_hom = cv2.perspectiveTransform(pt[None, :, :], H_matrix)
    return pt_hom[0, 0, 0], pt_hom[0, 0, 1]


def world_to_latlon(X, Y, lat0, lon0):
    d_lat = Y / 111320.0
    d_lon = X / (111320.0 * np.cos(np.radians(lat0)))
    return lat0 + d_lat, lon0 + d_lon


def calculate_orientation(bbox, H_matrix, img_width, img_height, y_center):
    x1, y1, w, h = bbox
    corners = [
        (x1, y1),
        (x1 + w, y1),
        (x1 + w, y1 + h),
        (x1, y1 + h)
    ]

    world_corners = [image_to_world(x, y, H_matrix) for x, y in corners]
    vectors = [
        (world_corners[1][0] - world_corners[0][0], world_corners[1][1] - world_corners[0][1]),
        (world_corners[2][0] - world_corners[1][0], world_corners[2][1] - world_corners[1][1]),
        (world_corners[3][0] - world_corners[2][0], world_corners[3][1] - world_corners[2][1]),
        (world_corners[0][0] - world_corners[3][0], world_corners[0][1] - world_corners[3][1])
    ]

    longest_idx = np.argmax([np.linalg.norm(v) for v in vectors])
    direction_vector = vectors[longest_idx]
    angle_rad = math.atan2(direction_vector[1], direction_vector[0])
    angle_deg = math.degrees(angle_rad)

    if longest_idx in [1, 2]:
        angle_deg = (angle_deg + 180) % 360

    if y_center > img_height - 180:
        angle_deg = (angle_deg + 90) % 360
    else:
        angle_deg = (angle_deg + 0) % 360

    return angle_deg, angle_rad


def create_parking_spot_polygon(center_lat, center_lon, width_m, length_m, angle_deg):
    width_deg = width_m / 111320.0
    length_deg = length_m / 111320.0
    angle_rad = math.radians(angle_deg)
    cos_val = math.cos(angle_rad)
    sin_val = math.sin(angle_rad)

    half_width = width_deg / 2
    half_length = length_deg / 2
    corners = [
        (-half_width, -half_length),
        (half_width, -half_length),
        (half_width, half_length),
        (-half_width, half_length)
    ]

    rotated_corners = []
    for dx, dy in corners:
        x_rot = dx * cos_val - dy * sin_val
        y_rot = dx * sin_val + dy * cos_val
        lat = center_lat + y_rot
        lon = center_lon + x_rot / math.cos(math.radians(center_lat))
        rotated_corners.append((lat, lon))

    return rotated_corners


def save_annotation_file(image_path, detections, output_folder):
    """Сохраняет аннотации в формате YOLO только для отфильтрованных авто"""
    base_name = os.path.basename(image_path).replace('.jpg', '')
    annotation_path = os.path.join(output_folder, f"{base_name}.txt")

    # Получаем временную метку
    timestamp_comment, _ = extract_and_format_timestamp(image_path)

    with open(annotation_path, 'w') as f:

        f.write(f"{timestamp_comment}\n")


        if not detections:
            return annotation_path

        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        for det in detections:
            x1, y1, w, h = det['bbox']
            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            width = w / img_width
            height = h / img_height


            f.write(f"{det['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return annotation_path


def process_single_image(image_path, output_folder, tracker=None, model=None):

    car_conf_threshold = 0.25
    truck_conf_threshold = 0.85
    iou_threshold = 0.45
    use_augment = True
    apply_preprocessing = True
    gamma_value = 1.5
    clahe_clipLimit = 2.0
    clahe_tileGridSize = (8, 8)
    apply_undistortion = True
    camera_matrix = np.array([[1000, 0, 1065], [0, 1000, 527.5], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)


    main_min_X, main_max_X = -1, 25
    main_min_Y, main_max_Y = -1, 7
    min_box_area = 100
    max_box_area = 50000
    branch_threshold = 1700
    branch_offset = (0, 6)
    branch_valid_y_range = (5.5, 12)


    parking_spot_size = (2.5, 5.3)


    origin_lat = 61.285367
    origin_lon = 73.347130
    camera_lat = 61.285552
    camera_lon = 73.347173


    orig_image = cv2.imread(image_path)
    if orig_image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None

    image_height, image_width = orig_image.shape[:2]


    if apply_undistortion:
        image = cv2.undistort(orig_image, camera_matrix, dist_coeffs)
    else:
        image = orig_image.copy()

    if apply_preprocessing:
        image = adjust_gamma(image, gamma=gamma_value)
        image = apply_CLAHE(image, clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
        image = enhance_edges(image)


    with torch.no_grad():
        results = model.predict(
            source=image,
            conf=car_conf_threshold,
            iou=iou_threshold,
            augment=use_augment,
            verbose=False
        )

    boxes = []
    scores = []
    classes = []
    detections = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            w = x2 - x1
            h = y2 - y1
            area = w * h


            if cls not in [2, 7]:
                continue


            if cls == 7 and conf < truck_conf_threshold:
                cls = 2


            if area < min_box_area or area > max_box_area:
                continue


            mapped_class = 0 if cls == 2 else 1
            boxes.append([x1, y1, w, h])
            scores.append(conf)
            classes.append(mapped_class)


    keep_indices = apply_nms(boxes, scores)


    for idx in keep_indices:
        x1, y1, w, h = boxes[idx]
        cx = x1 + w / 2
        cy = y1 + h / 2

        detections.append({
            "class": classes[idx],
            "conf": scores[idx],
            "bbox": (x1, y1, w, h),
            "center": (cx, cy)
        })


    pts_image = np.array([
        [image_width - 100, 100],
        [100, 100],
        [100, image_height - 100],
        [image_width - 100, image_height - 100]
    ], dtype=np.float32)

    pts_world = np.array([
        [0.0, 0.0],
        [25.0, 0.0],
        [25.0, 5.3],
        [0.0, 5.3]
    ], dtype=np.float32)

    H, status = cv2.findHomography(pts_image, pts_world, cv2.RANSAC, 5.0)
    if H is None:
        print("Не удалось вычислить матрицу гомографии.")
        return None


    _, timestamp_str = extract_and_format_timestamp(image_path)


    vehicles = []
    for i, det in enumerate(detections):
        cx, cy = det["center"]
        try:

            if cy <= image_height - 320:
                continue

            Xw, Yw = image_to_world(cx, cy, H)


            if cx > branch_threshold:
                Xw += branch_offset[0]
                Yw += branch_offset[1]
                if not (branch_valid_y_range[0] <= Yw <= branch_valid_y_range[1]):
                    continue
            else:
                if not (main_min_X <= Xw <= main_max_X and main_min_Y <= Yw <= main_max_Y):
                    continue


            orientation_deg, orientation_rad = calculate_orientation(det["bbox"], H, image_width, image_height, cy)
            lat, lon = world_to_latlon(Xw, Yw, origin_lat, origin_lon)


            corners = create_parking_spot_polygon(lat, lon, parking_spot_size[0], parking_spot_size[1], orientation_deg)

            vehicle_data = {
                "world_coords": (Xw, Yw),
                "lat": lat,
                "lon": lon,
                "conf": det["conf"],
                "bbox": det["bbox"],
                "class": det["class"],
                "orientation_deg": orientation_deg,
                "orientation_rad": orientation_rad,
                "position": "bottom",
                "timestamp": timestamp_str,
                "corner1_lat": corners[0][0],
                "corner1_lon": corners[0][1],
                "corner2_lat": corners[1][0],
                "corner2_lon": corners[1][1],
                "corner3_lat": corners[2][0],
                "corner3_lon": corners[2][1],
                "corner4_lat": corners[3][0],
                "corner4_lon": corners[3][1]
            }

            vehicles.append(vehicle_data)
        except Exception as e:
            print(f"Ошибка обработки объекта {i + 1}: {str(e)}")


    if tracker is not None:
        vehicles = tracker.update(vehicles, timestamp_str)
    else:
        for i, vehicle in enumerate(vehicles):
            vehicle['id'] = i + 1
            vehicle['is_same_car'] = False


    drawn_image = image.copy()
    for v in vehicles:
        x1, y1, w, h = v["bbox"]
        color = (0, 0, 255) if v["class"] == 1 else (255, 0, 0)
        cv2.rectangle(drawn_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)


        cv2.putText(drawn_image, f"ID {v['id']}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


        class_text = "Truck" if v["class"] == 1 else "Car"
        conf_text = f"{v['conf']:.2f}"
        cv2.putText(drawn_image, f"{class_text} {conf_text}",
                    (int(x1), int(y1 + h + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    if vehicles:
        center_lat = sum(v["lat"] for v in vehicles) / len(vehicles)
        center_lon = sum(v["lon"] for v in vehicles) / len(vehicles)
    else:
        center_lat, center_lon = camera_lat, camera_lon

    m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

    for v in vehicles:
        spot_polygon = [
            [v['corner1_lat'], v['corner1_lon']],
            [v['corner2_lat'], v['corner2_lon']],
            [v['corner3_lat'], v['corner3_lon']],
            [v['corner4_lat'], v['corner4_lon']]
        ]

        folium.Polygon(
            locations=spot_polygon,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            popup=f"ID:{v['id']} Угол:{int(v['orientation_deg'])}° Позиция:{v['position']}"
        ).add_to(m)

        folium.CircleMarker(
            location=[v["lat"], v["lon"]],
            radius=3,
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(m)

    folium.Marker(
        location=[camera_lat, camera_lon],
        icon=folium.Icon(color="green", icon="camera", prefix="fa")
    ).add_to(m)


    filtered_detections = []
    for v in vehicles:
        filtered_detections.append({
            'class': v['class'],
            'bbox': v['bbox']
        })

    save_annotation_file(image_path, filtered_detections, output_folder)

    return {
        'image_path': image_path,
        'vehicles': vehicles,
        'drawn_image': drawn_image,
        'map': m,
        'image_size': (image_width, image_height),
        'detections': filtered_detections
    }


def process_folder(input_folder, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)


    model_path = "yolov8l.pt"
    model = YOLO(model_path)


    image_files = [f for f in os.listdir(input_folder)
                   if (re.match(r'photo_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.jpg$', f)
                       or re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_photo\.jpg$', f))]


    image_files.sort(key=lambda x: re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', x).group(1))

    all_results = []
    processed_count = 0
    tracker = VehicleTracker()
    batch_size = 10

    for i in tqdm(range(0, len(image_files), batch_size), desc="Обработка батчами"):

        torch.cuda.empty_cache()
        gc.collect()


        batch_files = image_files[i:i + batch_size]
        batch_results = []

        for image_file in batch_files:
            try:
                image_path = os.path.join(input_folder, image_file)
                result = process_single_image(image_path, output_folder, tracker, model)
                if result is None:
                    continue


                output_image_path = os.path.join(output_folder, f"processed_{image_file}")
                cv2.imwrite(output_image_path, cv2.cvtColor(result['drawn_image'], cv2.COLOR_RGB2BGR))


                map_path = os.path.join(output_folder, f"map_{os.path.splitext(image_file)[0]}.html")
                result['map'].save(map_path)


                for vehicle in result['vehicles']:
                    batch_results.append({
                        'image': image_file,
                        'vehicle_id': vehicle['id'],
                        'class': vehicle['class'],
                        'confidence': vehicle['conf'],
                        'timestamp': vehicle['timestamp'],
                        'latitude': vehicle['lat'],
                        'longitude': vehicle['lon'],
                        'world_x': vehicle['world_coords'][0],
                        'world_y': vehicle['world_coords'][1],
                        'bbox_x': vehicle['bbox'][0],
                        'bbox_y': vehicle['bbox'][1],
                        'bbox_w': vehicle['bbox'][2],
                        'bbox_h': vehicle['bbox'][3],
                        'orientation_deg': vehicle['orientation_deg'],
                        'orientation_rad': vehicle['orientation_rad'],
                        'corner1_lat': vehicle['corner1_lat'],
                        'corner1_lon': vehicle['corner1_lon'],
                        'corner2_lat': vehicle['corner2_lat'],
                        'corner2_lon': vehicle['corner2_lon'],
                        'corner3_lat': vehicle['corner3_lat'],
                        'corner3_lon': vehicle['corner3_lon'],
                        'corner4_lat': vehicle['corner4_lat'],
                        'corner4_lon': vehicle['corner4_lon'],
                        'is_same_car': vehicle.get('is_same_car', False),
                        'position': vehicle['position'],
                        'image_width': result['image_size'][0],
                        'image_height': result['image_size'][1]
                    })

                processed_count += 1


                del result
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"\nОшибка при обработке {image_file}: {str(e)}")


        if batch_results:
            try:
                save_detection_results(output_folder, batch_results)
            except Exception as e:
                print(f"\nОшибка сохранения CSV для батча: {str(e)}")


        del batch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nОбработка завершена. Успешно обработано {processed_count}/{len(image_files)} изображений")


if __name__ == "__main__":
    input_folder = r"F:\PythonProjects\pythonProject8\test_photo"
    process_folder(input_folder, output_folder=r"F:\PythonProjects\pythonProject8\output\test_photo")