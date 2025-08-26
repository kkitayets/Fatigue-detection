import cv2
import dlib
import numpy as np
import os
import time

# Путь к файлу с предсказателем ключевых точек
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Инициализация детектора лиц и предсказателя ключевых точек
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Параметры
EAR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 48
EYE_CLOSE_THRESHOLD = 0.3
EYE_CLOSE_CONSEC_FRAMES = 30
MOUTH_OPEN_THRESHOLD = 0.5
HEAD_TILT_THRESHOLD = 10.0
EYE_CLOSED_DURATION = 5  # Длительность закрытых глаз для усталости

frame_count = 0
eye_close_count = 0
mouth_open_count = 0
last_blink_time = time.time()

# Функция для получения ключевых точек лица
def get_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        landmarks = predictor(gray, rect)
        return landmarks, rect
    return None, None

# Функция для вычисления соотношения сторон глаза (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Функция для вычисления показателя усталости
def calculate_fatigue(landmarks):
    left_eye = np.array([[landmarks.part(36).x, landmarks.part(36).y],
                         [landmarks.part(37).x, landmarks.part(37).y],
                         [landmarks.part(38).x, landmarks.part(38).y],
                         [landmarks.part(39).x, landmarks.part(39).y],
                         [landmarks.part(40).x, landmarks.part(40).y],
                         [landmarks.part(41).x, landmarks.part(41).y]], np.int32)
    right_eye = np.array([[landmarks.part(42).x, landmarks.part(42).y],
                          [landmarks.part(43).x, landmarks.part(43).y],
                          [landmarks.part(44).x, landmarks.part(44).y],
                          [landmarks.part(45).x, landmarks.part(45).y],
                          [landmarks.part(46).x, landmarks.part(46).y],
                          [landmarks.part(47).x, landmarks.part(47).y]], np.int32)

    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    # Определение состояния рта
    mouth = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 68)], np.int32)
    mouth_width = np.linalg.norm(mouth[0] - mouth[6])
    mouth_height = np.linalg.norm(mouth[3] - mouth[9])
    mouth_open_ratio = mouth_height / mouth_width

    # Определение наклона головы
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])
    chin = np.array([landmarks.part(8).x, landmarks.part(8).y])
    head_tilt = np.degrees(np.arctan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0]))

    return avg_ear, mouth_open_ratio, head_tilt, left_eye, right_eye, mouth

# Основной цикл распознавания усталости
cap = cv2.VideoCapture(0)

# Установка низкого разрешения для повышения производительности
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, rect = get_landmarks(frame, detector, predictor)
    if landmarks:
        avg_ear, mouth_open_ratio, head_tilt, left_eye, right_eye, mouth = calculate_fatigue(landmarks)

        # Проверка на частоту морганий
        if avg_ear < EAR_THRESHOLD:
            frame_count += 1
            last_blink_time = time.time()  # Сброс времени последнего моргания
        else:
            frame_count = 0

        # Проверка на длительное закрытие глаз
        if avg_ear < EYE_CLOSE_THRESHOLD:
            eye_close_count += 1
        else:
            eye_close_count = 0

        # Проверка на открытие рта
        if mouth_open_ratio > MOUTH_OPEN_THRESHOLD:
            mouth_open_count += 1
        else:
            mouth_open_count = 0

        # Проверка на наклон головы
        head_tilt_detected = abs(head_tilt) > HEAD_TILT_THRESHOLD

        # Проверка на усталость по отсутствию морганий
        current_time = time.time()
        if current_time - last_blink_time > EYE_CLOSED_DURATION:
            cv2.putText(frame, "Drowsiness Detected (No Blink)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif frame_count >= EYE_AR_CONSEC_FRAMES or eye_close_count >= EYE_CLOSE_CONSEC_FRAMES:
            cv2.putText(frame, "Drowsiness Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mouth_open_count > EYE_AR_CONSEC_FRAMES:
            cv2.putText(frame, "Mouth Open Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if head_tilt_detected:
            cv2.putText(frame, " ", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Отображение областей вокруг глаз и рта
        if rect:
            (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
            cv2.polylines(frame, [mouth], True, (255, 0, 0), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
