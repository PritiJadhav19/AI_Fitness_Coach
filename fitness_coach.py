import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def angle(a, b, c):
    """Angle at point b for points a-b-c (in degrees)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang

# --- squat state ---
squat_count = 0
squat_stage = "up"   # up/down
score = 100

# thresholds (tune)
KNEE_DOWN = 100     # knee angle <= this => down
KNEE_UP = 160       # knee angle >= this => up
MIN_DEPTH = 95      # deeper squat => smaller knee angle

# cooldown for rep counting
last_rep_time = 0
REP_COOLDOWN = 0.4

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

start_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    feedback = []
    form_ok = True

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        def pt(i):
            return (lm[i].x * w, lm[i].y * h)

        # Use LEFT side landmarks (works fine for most). You can average left+right later.
        hip = pt(mp_pose.PoseLandmark.LEFT_HIP.value)
        knee = pt(mp_pose.PoseLandmark.LEFT_KNEE.value)
        ankle = pt(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        shoulder = pt(mp_pose.PoseLandmark.LEFT_SHOULDER.value)

        knee_ang = angle(hip, knee, ankle)
        hip_ang = angle(shoulder, hip, knee)  # checks torso/hip bend

        # --- Squat logic ---
        now = time.time()
        if knee_ang <= KNEE_DOWN:
            squat_stage = "down"
        if knee_ang >= KNEE_UP and squat_stage == "down" and (now - last_rep_time) > REP_COOLDOWN:
            squat_count += 1
            squat_stage = "up"
            last_rep_time = now

        # --- Form checks ---
        # 1) Depth
        if squat_stage == "down" and knee_ang > MIN_DEPTH:
            form_ok = False
            feedback.append("Go deeper (squat depth)")

        # 2) Excess forward lean (hip angle too small)
        if hip_ang < 65:
            form_ok = False
            feedback.append("Straighten back (less forward lean)")

        # 3) Knee tracking: rough check (knee not too far ahead of ankle in x)
        if abs(knee[0] - ankle[0]) > 120:
            form_ok = False
            feedback.append("Keep knee aligned over foot")

        # scoring
        if form_ok:
            score = min(100, score + 0.2)
        else:
            score = max(0, score - 0.6)

        # draw skeleton
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show angles
        cv2.putText(frame, f"Knee: {int(knee_ang)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Hip: {int(hip_ang)}", (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # UI overlay
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, "AI Fitness Coach - Squats", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Reps: {squat_count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Stage: {squat_stage}", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Score: {int(score)}", (420, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Time: {elapsed}s", (600, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    y = 190
    if feedback:
        for msg in feedback[:3]:
            cv2.putText(frame, f"- {msg}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            y += 32
    else:
        cv2.putText(frame, "Good form ✅", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)

    cv2.imshow("AI Fitness Coach", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()