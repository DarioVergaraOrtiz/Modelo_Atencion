import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS_IDX = [474, 475, 476, 477]
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# Inicializar Face Mesh (usa refine_landmarks=True para iris)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def estimate_gaze(frame, fm):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = fm.process(rgb)
    if not results.multi_face_landmarks:
        return None, frame

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    # Centro de iris
    def iris_center(idxs):
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idxs])
        return pts.mean(axis=0).astype(int)
    lc = iris_center(LEFT_IRIS_IDX)
    rc = iris_center(RIGHT_IRIS_IDX)

    # Centro global
    center = ((lc + rc) // 2).tolist()
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h]).astype(int)

    gaze_vec = np.array(center) - nose

    # Dibujar guía
    cv2.circle(frame, tuple(lc), 2, (0,255,0), -1)
    cv2.circle(frame, tuple(rc), 2, (0,255,0), -1)
    cv2.arrowedLine(frame, tuple(nose), tuple(center), (0,0,255), 2, tipLength=0.2)
    return gaze_vec, frame
