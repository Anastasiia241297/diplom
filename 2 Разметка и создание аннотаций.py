import os
import cv2
import numpy as np
import folium
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import math
import re
from PIL import Image
import pytesseract
import gc

# Настройки Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\tesseract.exe'


def save_detection_results(output_folder, results_data):
    """Сохраняет результаты детекции в CSV файл с ВСЕМИ данными"""
    csv_path = os.path.join(output_folder, "detection_results_test_photo.csv")

    # Определяем полный набор полей
    fieldnames = [
        'image', 'vehicle_id', 'class', 'confidence',
        'latitude', 'longitude', 'world_x', 'world_y',
        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'orientation', 'position', 'image_width', 'image_height'
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = pd.DataFrame(results_data)

        if file_exists:
            # Читаем существующие данные и объединяем
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, writer], ignore_index=True)
            # Удаляем дубликаты
            updated_df.drop_duplicates(
                subset=['image', 'vehicle_id'],
                keep='last',
                inplace=True
            )
            updated_df.to_csv(csv_path, index=False)
        else:
            writer.to_csv(csv_path, index=False)

    print(f"Результаты сохранены в: {csv_path}")


def clean_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '', text).strip()


def extract_and_format_timestamp(image_path):
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


def read_yolo_annotations(txt_path, img_width, img_height):
    detections = []
    if not os.path.exists(txt_path):
        print(f"Файл аннотаций не найден: {txt_path}")
        return detections

    with open(txt_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2

                detections.append({
                    "class": class_id,
                    "conf": 1.0,
                    "bbox": (x1, y1, width, height),
                    "center": (x_center, y_center)
                })
            except ValueError:
                continue
    return detections


def image_to_world(x, y, H_matrix):
    pt = np.array([[x, y]], dtype=np.float32)
    pt_hom = cv2.perspectiveTransform(pt[None, :, :], H_matrix)
    return pt_hom[0, 0, 0], pt_hom[0, 0, 1]


def world_to_latlon(X, Y, lat0, lon0):
    d_lat = Y / 111320.0
    d_lon = X / (111320.0 * np.cos(np.radians(lat0)))
    return lat0 + d_lat, lon0 + d_lon


def find_annotation_file(image_path, output_folder):
    base_name = os.path.basename(image_path)
    if base_name.endswith('.jpg'):
        return os.path.join(output_folder, base_name.replace(".jpg", ".txt"))
    return None


def apply_nms(boxes, scores, iou_threshold=0.45):
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

    return angle_deg


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


def save_annotation_file(image_path, vehicles, output_folder):
    base_name = os.path.basename(image_path).replace(".jpg", "")
    annotation_path = os.path.join(output_folder, f"{base_name}.txt")

    with open(annotation_path, 'w') as f:
        img = cv2.imread(image_path)
        img_height, img_width = img.shape[:2]

        timestamp_comment, _ = extract_and_format_timestamp(image_path)
        f.write(f"{timestamp_comment}\n")

        for vehicle in vehicles:
            x1, y1, w, h = vehicle['bbox']
            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            f.write(f"{vehicle['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return annotation_path


def clean_annotations_from_filtered(image_path, output_folder, filtered_vehicles):
    base_name = os.path.basename(image_path).replace(".jpg", "")
    annotation_path = os.path.join(output_folder, f"{base_name}.txt")

    if not os.path.exists(annotation_path):
        return

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    comments = [line for line in lines if line.startswith('#')]

    valid_annotations = []
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    for vehicle in filtered_vehicles:
        x1, y1, w, h = vehicle['bbox']
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        width = w / img_width
        height = h / img_height

        annotation_line = f"{vehicle['class']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        valid_annotations.append(annotation_line)

    # Перезаписываем файл
    with open(annotation_path, 'w') as f:
        f.writelines(comments)
        f.writelines(valid_annotations)


def process_single_image(image_path, output_folder=None, use_yolo=False, model=None):
    conf_threshold = 0.2
    truck_conf_threshold = 0.7
    iou_threshold = 0.4
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
    top_crop_height = 320

    parking_spot_size = (2.5, 5.3)

    origin_lat = 61.229408
    origin_lon = 73.453131
    camera_lat = 61.229462
    camera_lon = 73.453166

    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    image_height, image_width = orig_image.shape[:2]

    if apply_undistortion:
        image = cv2.undistort(orig_image, camera_matrix, dist_coeffs)
    else:
        image = orig_image.copy()

    if apply_preprocessing:
        image = adjust_gamma(image, gamma=gamma_value)
        image = apply_CLAHE(image, clipLimit=clahe_clipLimit, tileGridSize=clahe_tileGridSize)
        image = enhance_edges(image)

    detections = []

    if use_yolo:
        if model is None:
            model_path = "yolov8x.pt"
            model = YOLO(model_path)

        results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold, augment=use_augment)

        boxes = []
        scores = []
        classes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                area = w * h

                if cls not in [2, 7]:
                    continue

                if area < min_box_area or area > max_box_area:
                    continue

                if cls == 7 and conf < truck_conf_threshold:
                    cls = 2

                mapped_class = 0 if cls == 2 else 1
                boxes.append([x1, y1, w, h])
                scores.append(conf)
                classes.append(mapped_class)

        keep_indices = apply_nms(boxes, scores, iou_threshold=0.45)

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

            marked_frame = image.copy()
        for det in detections:
            x1, y1, w, h = det['bbox']
            color = (0, 255, 0) if det['class'] == 0 else (0, 0, 255)
            label = "Car" if det['class'] == 0 else "Truck"
            cv2.rectangle(marked_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(marked_frame, f"{label} {det['conf']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if 'id' in det:
                cv2.putText(marked_frame, f"ID: {det['id']}",
                            (x1, y1 + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        txt_path = find_annotation_file(image_path, output_folder)
        if not txt_path or not os.path.exists(txt_path):
            print(f"Файл аннотаций не найден для: {os.path.basename(image_path)}")
            return None

        detections = read_yolo_annotations(txt_path, image_width, image_height)
        marked_frame = image.copy()


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
        raise ValueError("Не удалось вычислить матрицу гомографии.")


    vehicles = []
    filtered_out = []
    vehicle_counter = 1
    filtered_counter = 1001

    for i, det in enumerate(detections):
        cx, cy = det.get("center", (None, None))
        bbox = det.get("bbox", (0, 0, 0, 0))
        x1, y1, w, h = bbox
        reason = None
        Xw, Yw = None, None

        try:

            if None in (cx, cy) or None in bbox:
                reason = "Неполные данные объекта"
                raise ValueError(reason)


            if cy <= image_height - top_crop_height:
                reason = "Верхняя часть изображения"
                raise Exception(reason)

            Xw, Yw = image_to_world(cx, cy, H)

            is_branch = cx > branch_threshold

            if is_branch:
                Xw += branch_offset[0]
                Yw += branch_offset[1]

                if not (branch_valid_y_range[0] <= Yw <= branch_valid_y_range[1]):
                    reason = f"Ветка: Y={Yw:.2f} вне диапазона {branch_valid_y_range}"
                    raise Exception(reason)
            else:
                if not (main_min_X <= Xw <= main_max_X and main_min_Y <= Yw <= main_max_Y):
                    reason = f"Основная зона: X={Xw:.2f}, Y={Yw:.2f} вне диапазона X[{main_min_X},{main_max_X}], Y[{main_min_Y},{main_max_Y}]"
                    raise Exception(reason)

            orientation = calculate_orientation(bbox, H, image_width, image_height, cy)

            lat, lon = world_to_latlon(Xw, Yw, origin_lat, origin_lon)

            vehicle_data = {
                "id": vehicle_counter,
                "world_coords": (Xw, Yw),
                "lat": lat,
                "lon": lon,
                "conf": det.get("conf", 0),
                "bbox": bbox,
                "class": det.get("class", 0),
                "orientation": orientation,
                "position": "branch" if is_branch else "main"
            }

            det['id'] = vehicle_counter
            vehicles.append(vehicle_data)
            vehicle_counter += 1

        except Exception as e:
            filtered_out.append({
                "id": filtered_counter,
                "class": det.get("class", -1),
                "center": (cx, cy),
                "bbox": bbox,
                "reason": str(e) if reason is None else reason,
                "world_coords": (Xw, Yw) if Xw is not None and Yw is not None else None
            })
            filtered_counter += 1

    drawn_image = image.copy()

    for v in vehicles:
        x1, y1, w, h = v["bbox"]
        color = (255, 0, 0)
        cv2.rectangle(drawn_image, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        cv2.putText(drawn_image, f"{v['conf']:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(drawn_image, f"ID:{v['id']}", (int(x1), int(y1 + h + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if vehicles:
        center_lat = sum(v["lat"] for v in vehicles) / len(vehicles)
        center_lon = sum(v["lon"] for v in vehicles) / len(vehicles)
    else:
        center_lat, center_lon = camera_lat, camera_lon

    m = folium.Map(location=[center_lat, center_lon], zoom_start=20)

    for v in vehicles:
        spot_polygon = create_parking_spot_polygon(
            v["lat"], v["lon"],
            parking_spot_size[0], parking_spot_size[1],
            v["orientation"]
        )

        folium.Polygon(
            locations=spot_polygon,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
            popup=f"ID:{v['id']} Угол:{int(v['orientation'])}° Позиция:{v['position']}"
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

    return {
        'image_path': image_path,
        'vehicles': vehicles,
        'filtered_out': filtered_out,
        'drawn_image': drawn_image,
        'map': m,
        'image_size': (image_width, image_height),
        'all_detections': detections
    }


def save_detection_results(output_folder, results_data):
    csv_path = os.path.join(output_folder, "detection_results_test_photo.csv")

    fieldnames = [
        'image', 'vehicle_id', 'class', 'confidence',
        'latitude', 'longitude', 'world_x', 'world_y',
        'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'orientation', 'position', 'image_width', 'image_height'
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = pd.DataFrame(results_data)

        if file_exists:
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, writer], ignore_index=True)
            updated_df.drop_duplicates(
                subset=['image', 'vehicle_id'],
                keep='last',
                inplace=True
            )
            updated_df.to_csv(csv_path, index=False)
        else:
            writer.to_csv(csv_path, index=False)

    print(f"Результаты сохранены в: {csv_path}")


def process_folder(input_folder, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder)
                   if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_photo\.jpg$', f)]

    print(f"Найдено {len(image_files)} изображений для обработки")

    all_results = []
    processed_count = 0
    batch_size = 10

    model = YOLO("yolov8x.pt")

    for batch_idx in tqdm(range(0, len(image_files), batch_size),
                          desc="Обработка батчей",
                          unit="batch"):
        batch = image_files[batch_idx:batch_idx + batch_size]
        batch_results = []

        for image_file in batch:
            try:
                image_path = os.path.join(input_folder, image_file)
                result = process_single_image(image_path, output_folder, use_yolo=True, model=model)

                if not result or not result['vehicles']:
                    continue

                marked_img_path = os.path.join(output_folder, f"marked_{image_file}")
                cv2.imwrite(marked_img_path, result['drawn_image'])

                save_annotation_file(image_path, result['vehicles'], output_folder)
                clean_annotations_from_filtered(image_path, output_folder, result['vehicles'])

                map_path = os.path.join(output_folder, f"map_{os.path.splitext(image_file)[0]}.html")
                result['map'].save(map_path)

                   for vehicle in result['vehicles']:
                    vehicle_data = {
                        'image': image_file,
                        'vehicle_id': vehicle['id'],
                        'class': vehicle['class'],
                        'confidence': round(float(vehicle['conf']), 6) if vehicle['conf'] else 0,
                        'latitude': round(float(vehicle['lat']), 8) if vehicle['lat'] else 0,
                        'longitude': round(float(vehicle['lon']), 8) if vehicle['lon'] else 0,
                        'world_x': round(float(vehicle['world_coords'][0]), 6) if vehicle['world_coords'] else 0,
                        'world_y': round(float(vehicle['world_coords'][1]), 6) if vehicle['world_coords'] else 0,
                        'bbox_x': round(float(vehicle['bbox'][0])) if vehicle['bbox'] else 0,
                        'bbox_y': round(float(vehicle['bbox'][1])) if vehicle['bbox'] else 0,
                        'bbox_w': round(float(vehicle['bbox'][2])) if vehicle['bbox'] else 0,
                        'bbox_h': round(float(vehicle['bbox'][3])) if vehicle['bbox'] else 0,
                        'orientation': round(float(vehicle['orientation']), 1) if vehicle['orientation'] else 0,
                        'position': vehicle['position'],
                        'image_width': result['image_size'][0],
                        'image_height': result['image_size'][1]
                    }
                    batch_results.append(vehicle_data)

                processed_count += 1

            except Exception as e:
                print(f"\nОшибка при обработке {image_file}: {str(e)}")
                continue

            finally:
                gc.collect()

        if batch_results:
            try:
                save_detection_results(output_folder, batch_results)
                all_results.extend(batch_results)
            except Exception as e:
                print(f"\nОшибка сохранения CSV: {str(e)}")

    print(f"\nОбработка завершена. Успешно обработано {processed_count}/{len(image_files)} изображений")
    print(f"Всего обнаружено {len(all_results)} транспортных средств")


if __name__ == "__main__":
    input_folder = r"F:\PythonProjects\pythonProject8\test_photo"
    output_folder = r"F:\PythonProjects\pythonProject8\output\test_photo"

    gc.collect()

    process_folder(input_folder, output_folder)