from flask import Flask, render_template
import threading
import webbrowser
import time
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Global flag
trigger_analysis = False

# Feature detection logic
def analyze_features():
    cap = cv2.VideoCapture(0)
    global trigger_analysis

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Get lip and eye landmarks
                jaw_left = face_landmarks.landmark[61]    # Left lip
                jaw_right = face_landmarks.landmark[291]  # Right lip
                eye_top = face_landmarks.landmark[159]    # Top of left eye
                eye_bottom = face_landmarks.landmark[145] # Bottom of left eye

                # Calculate distances
                lip_width = int(np.linalg.norm(
                    np.array([jaw_right.x * w, jaw_right.y * h]) -
                    np.array([jaw_left.x * w, jaw_left.y * h])
                ))
                eye_height = int(np.linalg.norm(
                    np.array([eye_top.x * w, eye_top.y * h]) -
                    np.array([eye_bottom.x * w, eye_bottom.y * h])
                ))

                # Trigger when user presses 'r'
                if trigger_analysis:
                    trigger_analysis = False  # reset flag

                    if lip_width > 90 and eye_height > 10:
                        emotion = "happy"
                    elif lip_width < 70 and eye_height > 12:
                        emotion = "sad"
                    else:
                        emotion = "neutral"

                    emoji_path = f"static/emojis/{emotion}.png"
                    if os.path.exists(emoji_path):
                        emoji_img = cv2.imread(emoji_path)
                        emoji_img = cv2.resize(emoji_img, (200, 200))
                        cv2.imshow("Detected Emotion", emoji_img)
                    else:
                        cv2.putText(frame, f"Image not found for {emotion}", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Show webcam window
        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # ESC or q to quit
            break
        elif key == ord('r'):
            trigger_analysis = True

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == '__main__':
    t1 = threading.Thread(target=analyze_features)
    t1.daemon = True
    t1.start()

    threading.Thread(target=open_browser).start()
    app.run(debug=False)
