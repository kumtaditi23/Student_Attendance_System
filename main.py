import csv

def mark_attendance(name):
    file_exists = os.path.isfile("attendance.csv")

    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        writer.writerow([name, date, time])

import cv2
import face_recognition
import dlib
import numpy as np
import os
from datetime import datetime
from scipy.spatial import distance

# ---------------- SETTINGS ----------------
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# ---------------- DLIB ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ---------------- FOLDER ----------------
output_folder = "attendance_images"
os.makedirs(output_folder, exist_ok=True)

# ---------------- LOAD KNOWN FACE ----------------
known_face_encodings = []
known_face_names = []

image = face_recognition.load_image_file("students/aditi.jpeg")
encodings = face_recognition.face_encodings(image)

if len(encodings) == 0:
    print("No face found in student image ❌")
    exit()

known_face_encodings.append(encodings[0])
known_face_names.append("Aditi")

# ---------------- BLINK FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# ---------------- CAMERA ----------------
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

blink_counter = 0
blink_detected = False
blink_cooldown = 0   # 👈 important
marked_attendance = set()

print("System Started... Press Q to exit")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = video_capture.read()

    if not ret or frame is None:
        print("Camera not working ❌")
        continue

    frame = cv2.resize(frame, (640, 480))

    # show camera first
    cv2.imshow("Smart Attendance System", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # -------- BLINK DETECTION --------
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape_np[LEFT_EYE]
        rightEye = shape_np[RIGHT_EYE]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        # 👇 FIXED BLINK LOGIC
        if ear < EYE_AR_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= EYE_AR_CONSEC_FRAMES:
                if blink_cooldown == 0:
                    blink_detected = True
                    blink_cooldown = 30   # 👈 delay (controls spam)
                    print("Blink detected ✅")
            blink_counter = 0

    # 👇 cooldown reduce every frame
    if blink_cooldown > 0:
        blink_cooldown -= 1

    # -------- FACE RECOGNITION --------
    face_locations = face_recognition.face_locations(rgb_frame)

    for face_location in face_locations:
        enc = face_recognition.face_encodings(rgb_frame, [face_location])

        if len(enc) == 0:
            continue

        face_encoding = enc[0]

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

            # 👇 ONLY AFTER BLINK
            if blink_detected and name not in marked_attendance:
                marked_attendance.add(name)

                time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{name}_{time_now}.jpg"
                filepath = os.path.join(output_folder, filename)

                cv2.imwrite(filepath, frame)
                mark_attendance(name)

                print(f"{name} marked present ✅ Image saved 📸")

                # 👇 RESET (IMPORTANT)
                blink_detected = False
                blink_counter = 0

        # draw box
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # show updated frame
    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
video_capture.release()
cv2.destroyAllWindows()