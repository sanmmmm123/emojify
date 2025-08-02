import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

def detect_emotion(landmarks, image_shape):
    h, w, _ = image_shape

    def get_point(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    # Mouth
    left = get_point(61)
    right = get_point(291)
    top = get_point(13)
    bottom = get_point(14)
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)

    # Eyes
    left_eye_top = get_point(159)
    left_eye_bottom = get_point(145)
    eye_openness = np.linalg.norm(left_eye_top - left_eye_bottom)

    # Eyebrows (for angry)
    brow_left = get_point(70)
    brow_right = get_point(300)
    brow_center = get_point(10)
    brow_dist = np.linalg.norm(brow_left - brow_right)

    # Mouth corners
    corner_left = get_point(61)
    corner_right = get_point(291)
    corner_diff = abs(corner_left[1] - corner_right[1])

    # Emotion decision logic
    if mar > 0.6 and eye_openness < 5:
        return "🥱 Yawning"
    elif eye_openness < 3:
        return "😴 Sleeping"
    elif mar > 0.5 and eye_openness > 6:
        return "😮 Surprised"
    elif corner_diff > 20:
        return "😏 Smirk"
    elif brow_dist < 90:
        return "😡 Angry"
    elif mar > 0.35:
        return "😊 Smiling"
    elif mar > 0.25:
        return "😐 Neutral"
    else:
        return "😢 Crying"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            emotion = detect_emotion(landmarks.landmark, frame.shape)
            mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
            cv2.putText(frame, emotion, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)

    cv2.imshow("Facial Emotion Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # Esc or q to quit
        break

cap.release()
cv2.destroyAllWindows()
