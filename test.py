import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from playsound import playsound
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# تحميل الموديل
model = load_model("eye_state_model (1).h5")

# إعداد MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                    max_num_faces=1,
                                    refine_landmarks=True,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

IMG_WIDTH, IMG_HEIGHT = 96, 96
ALARM_DURATION = 5 # threshold for closing eyes 
eyes_closed_start = None

def play_alarm(): # play sound alarm when eyes become close for  'ALARM_DURATION'
    playsound("alarm.wav")

def extract_eye_with_box(frame, landmarks, eye_points):
    h, w = frame.shape[:2]
    eye_region = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_points])
    x, y, w_eye, h_eye = cv2.boundingRect(eye_region)
    eye = frame[y:y+h_eye, x:x+w_eye]
    eye_resized = cv2.resize(eye, (IMG_WIDTH, IMG_HEIGHT))
    eye_gray = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2GRAY)
    eye_input = eye_gray.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1).astype(np.float32) / 255.0
    return eye_input, (x, y, w_eye, h_eye)

def process_frame(frame):
    global eyes_closed_start
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        try:
            right_eye_img, (x1, y1, w1, h1) = extract_eye_with_box(frame, landmarks, RIGHT_EYE)
            left_eye_img, (x2, y2, w2, h2) = extract_eye_with_box(frame, landmarks, LEFT_EYE)

            right_pred = model.predict(right_eye_img, verbose=0)[0][0]
            left_pred = model.predict(left_eye_img, verbose=0)[0][0]

            right_status = "Closed" if right_pred < 0.5 else "Open"
            left_status = "Closed" if left_pred < 0.5 else "Open"

            # تغيير لون الحواف حسب الحالة
            right_color = (0, 0, 255) if right_pred < 0.5 else (0, 255, 0)
            left_color = (0, 0, 255) if left_pred < 0.5 else (0, 255, 0)

            # عرض الحواف أرفع حول العينين
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), right_color, 1)
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), left_color, 1)

            # عرض الحالة والنسبة بشكل متتابع (الاسم فوق النسبة)
            cv2.putText(frame, f'{right_status}', (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f'{right_pred:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.putText(frame, f'{left_status}', (x2, y2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, f'{left_pred:.2f}', (x2, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # رسم مستطيل حول الوجه بالكامل
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    face_landmarks = face_landmarks.landmark
                    x_min = min([landmark.x for landmark in face_landmarks]) * w
                    y_min = min([landmark.y for landmark in face_landmarks]) * h
                    x_max = max([landmark.x for landmark in face_landmarks]) * w
                    y_max = max([landmark.y for landmark in face_landmarks]) * h
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

            if right_pred < 0.5 and left_pred < 0.5:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                elif time.time() - eyes_closed_start >= ALARM_DURATION:
                    cv2.putText(frame, "ALERT: Eyes Closed!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    threading.Thread(target=play_alarm).start()
            else:
                eyes_closed_start = None
        except Exception as e:
            print(f"Error: {e}")
    return frame

# واجهة المستخدم
class EyeStateApp:
    def __init__(self, root):
        self.root = root
        self.cap = None
        self.is_camera_running = False
        self.stop_video = False

        self.start_camera_button = tk.Button(self.root, text="تشغيل الكاميرا", width=25, command=self.start_camera)
        self.start_camera_button.pack(pady=10)

        self.stop_camera_button = tk.Button(self.root, text="إيقاف الكاميرا", width=25, command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_button.pack(pady=10)

        self.open_image_button = tk.Button(self.root, text="اختيار صورة", width=25, command=self.open_image)
        self.open_image_button.pack(pady=10)

        self.open_video_button = tk.Button(self.root, text="تشغيل فيديو", width=25, command=self.open_video)
        self.open_video_button.pack(pady=10)

        self.stop_video_button = tk.Button(self.root, text="إيقاف الفيديو", width=25, command=self.stop_video_playback, state=tk.DISABLED)
        self.stop_video_button.pack(pady=10)

    def start_camera(self):
        self.is_camera_running = True
        self.stop_camera_button.config(state=tk.NORMAL)
        self.start_camera_button.config(state=tk.DISABLED)
        self.cap = cv2.VideoCapture(0)
        self.show_camera_frame()

    def stop_camera(self):
        self.is_camera_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.stop_camera_button.config(state=tk.DISABLED)
        self.start_camera_button.config(state=tk.NORMAL)

    def show_camera_frame(self):
        if self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed = process_frame(frame)
                cv2.imshow("Camera Eye Detection", processed)
            self.root.after(10, self.show_camera_frame)

    def open_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        img = cv2.imread(path)
        if img is not None:
            processed = process_frame(img)
            cv2.imshow("Image Eye Detection", processed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def open_video(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.stop_video = False
        self.stop_video_button.config(state=tk.NORMAL)
        self.show_video_frame()

    def show_video_frame(self):
        if self.cap and self.cap.isOpened() and not self.stop_video:
            ret, frame = self.cap.read()
            if ret:
                processed = process_frame(frame)
                cv2.imshow("Video Eye Detection", processed)
                self.root.after(10, self.show_video_frame)
            else:
                self.stop_video_playback()
        else:
            self.stop_video_playback()

    def stop_video_playback(self):
        self.stop_video = True
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.stop_video_button.config(state=tk.DISABLED)

# تشغيل التطبيق
root = tk.Tk()
root.title("Eye State Detection GUI")
app = EyeStateApp(root)
root.mainloop()
