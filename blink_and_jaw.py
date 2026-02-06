import cv2
import dlib
import csv
import time
from scipy.spatial import distance as dist

# ----- EYE UTIL -----
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# ----- FACE SETUP -----
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/Users/juicejambouree/Downloads/eyes/shape_predictor_68_face_landmarks.dat"
)

# ----- BLINK PARAMS -----
EAR_THRESHOLD = 0.21
blink_counter = 0
blink_in_progress = False

# ----- LOGGING PARAMS -----
LOG_INTERVAL = 5.0      # blink aggregation
JAW_LOG_INTERVAL = 0.5  # new: jaw every half-second

last_log_time = time.time()
last_jaw_time = time.time()

prev_total_blinks = 0
baseline_jaw_dist = None

csv_filename = "blink_jaw_log.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "blinks_in_last_5s",
        "jaw_delta_pixels"
    ])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    jaw_delta = 0

    for face in faces:
        shape = predictor(gray, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        # ----- BLINK STUFF -----
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

        # ----- JAW MOVEMENT -----
        chin = landmarks[8]      # bottom of chin
        nose_base = landmarks[33]

        current_jaw_dist = dist.euclidean(chin, nose_base)

        # establish baseline on first detection
        if baseline_jaw_dist is None:
            baseline_jaw_dist = current_jaw_dist

        jaw_delta = current_jaw_dist - baseline_jaw_dist

        # ----- DISPLAY -----
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Jaw dY: {jaw_delta:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    # ----- BLINK LOG (5s) -----
    current_time = time.time()
    if current_time - last_log_time >= LOG_INTERVAL:
        blinks_in_interval = blink_counter - prev_total_blinks
        prev_total_blinks = blink_counter
        last_log_time = current_time

        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                blinks_in_interval,
                f"{jaw_delta:.2f}"
            ])

    # ----- JAW LOG (0.5s) -----
    if current_time - last_jaw_time >= JAW_LOG_INTERVAL:
        last_jaw_time = current_time

        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "",
                f"{jaw_delta:.2f}"
            ])

    cv2.imshow("Blink + Jaw Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
