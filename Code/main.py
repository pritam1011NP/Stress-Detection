"""
Modified Stress Detection (merged, improved)
Features:
- Uses MediaPipe Face Mesh for landmarks (no dlib required)
- Loads included mini_XCEPTION model for emotion recognition (_mini_XCEPTION.102-0.66.hdf5)
- Computes a heuristic stress score (0-100) combining emotion + eyebrow metric
- Overlays score and colored bounding box on webcam frames
- Logs timestamp and score to CSV
- Saves final stress trend plot when exiting
Usage: python Code/main.py --source 0
"""
import cv2
import mediapipe as mp
import numpy as np
import time, csv, argparse, os
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam or path to file)')
parser.add_argument('--output', type=str, default='stress_log.csv', help='CSV file to write stress log')
args = parser.parse_args()

# --- Load models ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '_mini_XCEPTION.102-0.66.hdf5')
emotion_model = None
if os.path.exists(MODEL_PATH):
    emotion_model = load_model(MODEL_PATH, compile=False)
else:
    print("Warning: emotion model not found at", MODEL_PATH)
    print("Emotion-based scoring will be disabled. Place the _mini_XCEPTION...hdf5 file at repository root.")

# --- MediaPipe setup ---
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Utilities ---
def normalize(v, lo, hi):
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi>lo else 0.0

# emotion to stress mapping (heuristic)
emotion_stress_map = {
    'angry': 0.95, 'disgust':0.75, 'fear':0.9, 'sad':0.6, 'neutral':0.3, 'happy':0.1, 'surprise':0.7
}
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']  # ordering used by model

# compute eyebrow inner distance using landmarks (mediapipe indexes for mesh)
def eyebrow_inner_distance(landmarks, w, h):
    # use two inner eyebrow points approximations
    # left inner eyebrow ~ 55, right inner eyebrow ~ 285 on mediapipe 468 mesh
    try:
        p1 = landmarks[55]
        p2 = landmarks[285]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return np.hypot(x2-x1, y2-y1)
    except Exception:
        return None

# preprocess face for emotion model
def preprocess_face_for_emotion(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48,48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

# compute stress score (0-100)
def compute_stress_score(emotion_prob, brow_dist, baseline_brow, baseline_range=(0.8,1.2)):
    # emotion component
    emot_comp = 0.0
    if emotion_prob is not None:
        # map to label with highest prob
        idx = int(np.argmax(emotion_prob))
        label = emotion_labels[idx] if idx < len(emotion_labels) else 'neutral'
        emot_comp = emotion_stress_map.get(label, 0.4)
    # brow component: smaller inner distance (furrowed) => higher stress.
    brow_comp = 0.5
    if brow_dist is not None and baseline_brow is not None and baseline_brow>0:
        ratio = brow_dist / baseline_brow
        # normalize: ratio < 0.9 -> higher stress, ratio > 1.1 -> lower stress
        if ratio < 0.9:
            brow_comp = normalize((0.9 - ratio), 0.0, 0.9)  # 0..1
        else:
            brow_comp = 0.0
    # combine
    combined = 0.6 * emot_comp + 0.4 * brow_comp
    return int(combined * 100), emot_comp, brow_comp

# --- Main loop ---
source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)
time.sleep(1.0)

# Baseline calculation: collect first N valid brow distances
baseline_samples = 40
baseline_vals = []
rolling = deque(maxlen=200)
timestamps = []
scores = []

# Prepare CSV
csv_file = open(args.output, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'score', 'emotion_component', 'brow_component'])

plt.ion()
fig, ax = plt.subplots(figsize=(6,2))
line, = ax.plot([], [], marker='o')
ax.set_ylim(0,100)
ax.set_xlim(0,200)
ax.set_title("Live stress (last samples)")
ax.set_ylabel("Score (0-100)")

frame_count = 0
baseline_brow = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h,w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    brow_dist = None
    emotion_prob = None
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        brow_dist = eyebrow_inner_distance(landmarks, w, h)
        # compute baseline
        if len(baseline_vals) < baseline_samples and brow_dist is not None:
            baseline_vals.append(brow_dist)
            baseline_brow = float(np.mean(baseline_vals))
        # crop face for emotion if model present
        if emotion_model is not None:
            # estimate bounding box from landmarks
            xs = [int(p.x*w) for p in landmarks]
            ys = [int(p.y*h) for p in landmarks]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            if x2-x1 > 20 and y2-y1 > 20:
                face_img = frame[y1:y2, x1:x2]
                try:
                    face_input = preprocess_face_for_emotion(face_img)
                    emotion_prob = emotion_model.predict(face_input)[0]
                except Exception as e:
                    emotion_prob = None
    # compute score
    score, emot_c, brow_c = compute_stress_score(emotion_prob, brow_dist, baseline_brow)
    rolling.append(score)
    timestamps.append(time.time())
    scores.append(score)
    csv_writer.writerow([timestamps[-1], score, emot_c, brow_c])
    frame_count += 1

    # annotate frame
    # box color
    if score < 35:
        color = (0,255,0)
    elif score < 65:
        color = (0,255,255)
    else:
        color = (0,0,255)
    cv2.putText(frame, f"Stress: {score}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"Baseline brow: {int(baseline_brow) if baseline_brow else 0}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    # draw small bar
    cv2.rectangle(frame, (20,100), (220,120), (50,50,50), -1)
    cv2.rectangle(frame, (20,100), (20 + int(2*score),120), color, -1)

    cv2.imshow("Stress Detector", frame)

    # update live plot every few frames
    if frame_count % 5 == 0:
        ydata = list(rolling)
        line.set_xdata(range(len(ydata)))
        line.set_ydata(ydata)
        ax.set_xlim(0, max(50, len(ydata)))
        fig.canvas.draw()
        fig.canvas.flush_events()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
# save final plot
plt.ioff()
plt.figure(figsize=(8,3))
plt.plot(scores, marker='o')
plt.title("Stress over time")
plt.xlabel("Frame index")
plt.ylabel("Stress (0-100)")
plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'stress_trend.png'))
print("Saved stress_trend.png and CSV:", args.output)
