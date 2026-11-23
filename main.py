import cv2 as cv
import mediapipe as mp


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

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


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

        for i in LEFT_EYE + RIGHT_EYE:
            x = int(face[i].x * WIDTH)
            y = int(face[i].y * HEIGHT)
            cv.circle(frame, (x, y), 2, (30, 30, 230), -1)

    cv.imshow(window_name, frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        print("ESC pressed")
        break

pose_model.close()
face_model.close()
video.release()
cv.destroyAllWindows()



