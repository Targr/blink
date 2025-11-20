import cv2
import dlib
import csv
import time
from scipy.spatial import distance as dist

# is your face a face
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# where are your eyes and how far apart are they probably
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# clearview 68 features face detection (evil)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/juicejambouree/Downloads/eyes/shape_predictor_68_face_landmarks.dat")

# sensitivity params
EAR_THRESHOLD = 0.21
blink_counter = 0
blink_in_progress = False

# make the csv
LOG_INTERVAL = 5.0   # seconds
last_log_time = time.time()
prev_total_blinks = 0

csv_filename = "blink_log.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "blinks_in_last_5s"])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [landmarks[i] for i in LEFT_EYE]
        right_eye = [landmarks[i] for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < EAR_THRESHOLD and not blink_in_progress:
            blink_in_progress = True
        elif ear >= EAR_THRESHOLD and blink_in_progress:
            blink_counter += 1
            blink_in_progress = False

        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # log every five secs
    current_time = time.time()
    if current_time - last_log_time >= LOG_INTERVAL:
        blinks_in_interval = blink_counter - prev_total_blinks
        prev_total_blinks = blink_counter
        last_log_time = current_time

        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), blinks_in_interval])

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
