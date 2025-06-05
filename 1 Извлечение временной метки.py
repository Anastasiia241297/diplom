import os
import cv2
import re
import pytesseract
from tqdm import tqdm
from datetime import datetime

# Настройки Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\tesseract.exe'


def clean_filename(text):
    return re.sub(r'[<>:"/\\|?*]', '', text).strip()


def is_valid_timestamp(timestamp):
    pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
    return re.match(pattern, timestamp) is not None


def extract_timestamp(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None

        height, width = image.shape[:2]
        cropped_image = image[0:min(100, height), 0:min(500, width)]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        raw_text = pytesseract.image_to_string(gray_image, config='--psm 6')

        patterns = [
            r'(\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2})',  # 2024/04/22 15:04:57
            r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})',  # 2024-04-22 15:04:57
            r'(\d{4}\.\d{2}\.\d{2}\s\d{2}:\d{2}:\d{2})',  # 2024.04.22 15:04:57
            r'(\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2})',  # 22/04/2024 15:04:57
            r'(\d{2}-\d{2}-\d{4}\s\d{2}:\d{2}:\d{2})',  # 22-04-2024 15:04:57
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_text)
            if match:
                timestamp = match.group(1)
                try:
                     dt = None
                    if '/' in timestamp:
                        if len(timestamp.split('/')[0]) == 4:
                            dt = datetime.strptime(timestamp, '%Y/%m/%d %H:%M:%S')
                        else:
                            dt = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S')
                    elif '-' in timestamp:
                        if len(timestamp.split('-')[0]) == 4:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        else:
                            dt = datetime.strptime(timestamp, '%d-%m-%Y %H:%M:%S')
                    elif '.' in timestamp:
                        dt = datetime.strptime(timestamp, '%Y.%m.%d %H:%M:%S')

                    if dt:
                        formatted = dt.strftime('%Y-%m-%d_%H-%M-%S')
                        if is_valid_timestamp(formatted):
                            return formatted
                except ValueError:
                    continue

        return None
    except Exception as e:
        print(f"Ошибка при обработке временной метки: {str(e)}")
        return None


def rename_photos():
    input_folder = r"F:\PythonProjects\pythonProject8\test_photo"

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    image_files = [f for f in os.listdir(input_folder)
                   if os.path.isfile(os.path.join(input_folder, f)) and f.lower().endswith(supported_formats)]

    if not image_files:
        print("Не найдено изображений для обработки.")
        return

    renamed_count = 0
    for image_file in tqdm(image_files, desc="Обработка фото"):
        try:
            image_path = os.path.join(input_folder, image_file)

            timestamp = extract_timestamp(image_path)

            if timestamp:
                file_ext = os.path.splitext(image_file)[1].lower()

                new_name = f"{timestamp}_photo{file_ext}"
                new_path = os.path.join(input_folder, new_name)

                 if os.path.exists(new_path):
                    base, ext = os.path.splitext(new_name)
                    counter = 1
                    while os.path.exists(os.path.join(input_folder, f"{base}_{counter}{ext}")):
                        counter += 1
                    new_name = f"{base}_{counter}{ext}"
                    new_path = os.path.join(input_folder, new_name)

                os.rename(image_path, new_path)
                renamed_count += 1
                print(f"Переименовано: {image_file} -> {new_name}")

        except Exception as e:
            print(f"Ошибка при обработке {image_file}: {str(e)}")

    print(f"\nОбработка завершена! Переименовано {renamed_count} из {len(image_files)} файлов.")


if __name__ == "__main__":
    rename_photos()