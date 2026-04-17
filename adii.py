import cv2
import face_recognition

# webcam start
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # convert BGR to RGB
    rgb_frame = frame[:, :, ::-1]

    # detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    # draw box on faces
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()