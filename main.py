import cv2 as cv
import mediapipe as mp
import math
import time


window_name = "SitSmart"
CAMERA_INDEX = 0

video = cv.VideoCapture(CAMERA_INDEX, cv.CAP_DSHOW)

if not video.isOpened():
    print("can't open camera")
    exit(1)

video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

WIDTH = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

pose_model = mp.solutions.pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

face_model = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

KAM_PELK = 10
ZIAD_PELK = 25
TOOL_BASTE = 0.5
START_TIME = time.time()
GARM_SHODAN = 60
FASELE_HOSHDAR = 120
last_alert = 0
alert_until = 0

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

SHOW_DEBUG = True
ASTANE_EAR = 0.21



blink_count = 0
CHESHM_BASTE = False
SHOROE_BASTE = 0
PELK_HA = []

def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def eye_ratio(face, idx):
    p1, p2, p3, p4 , p5, p6 = [face[i] for i in idx]
    vertical = distance(p2, p6) + distance(p3, p5)
    horizontal = distance(p1, p4)
    return vertical / (2 * horizontal)




cv.namedWindow(window_name, cv.WINDOW_NORMAL)

while True:
    ret, frame = video.read()
    if not ret:
        break



    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    pose_result = pose_model.process(rgb)
    face_results = face_model.process(rgb)

    if pose_result.pose_landmarks:
        landmarks = pose_result.pose_landmarks.landmark

        for i in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER):
            x = int(landmarks[i].x * WIDTH)
            y = int(landmarks[i].y * HEIGHT)
            cv.circle(frame, (x, y), 8, (30, 230, 30), -1)

        lx = int(landmarks[LEFT_SHOULDER].x * WIDTH)
        ly = int(landmarks[LEFT_SHOULDER].y * HEIGHT)
        rx = int(landmarks[RIGHT_SHOULDER].x * WIDTH)
        ry = int(landmarks[RIGHT_SHOULDER].y * HEIGHT)
        cv.line(frame, (lx, ly), (rx, ry), (230, 150, 30), 2)

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0].landmark

        ear = (eye_ratio(face, LEFT_EYE) + eye_ratio(face, RIGHT_EYE)) / 2
        now = time.time()

        if ear < ASTANE_EAR:
            if not CHESHM_BASTE:
                CHESHM_BASTE = True
                SHOROE_BASTE = now

        else:
            if CHESHM_BASTE:
                CHESHM_BASTE = False
                if now - SHOROE_BASTE < 1.0:
                    PELK_HA.append(now)

        PELK_HA = [t for t in PELK_HA if now - t <= 60]

        cv.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv.putText(frame, f"Blinks/min: {len(PELK_HA)}", (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
        closed_for = now - SHOROE_BASTE if CHESHM_BASTE else 0

        KHASTE = (
            closed_for > TOOL_BASTE
            or (now - START_TIME > GARM_SHODAN
                and (len(PELK_HA) < KAM_PELK or len(PELK_HA) > ZIAD_PELK))
        )

        if KHASTE and now - last_alert > FASELE_HOSHDAR:
            last_alert = now
            alert_until = now + 5

        if now < alert_until:
            cv.putText(frame, "Your eyes are tired! take Break" ,
                       (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (0, 0, 255), 2)


        for i in LEFT_EYE + RIGHT_EYE:
            x = int(face[i].x * WIDTH)
            y = int(face[i].y * HEIGHT)
            if SHOW_DEBUG:
                cv.circle(frame, (x,y), 2, (30, 30, 230), -1)

    cv.imshow(window_name, frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        print("ESC pressed")
        break

pose_model.close()
face_model.close()
video.release()
cv.destroyAllWindows()



