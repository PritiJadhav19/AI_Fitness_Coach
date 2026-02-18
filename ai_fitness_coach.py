import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import csv
from datetime import datetime

# -------------------- Voice (pyttsx3) --------------------
VOICE_ENABLED = True
try:
    import pyttsx3
    engine = pyttsx3.init("sapi5")  # better on Windows
    engine.setProperty("rate", 175)
except Exception:
    VOICE_ENABLED = False
    engine = None

_last_said = {}
def say(text: str, cooldown: float = 2.0):
    """Non-blocking voice feedback with cooldown per phrase."""
    if not VOICE_ENABLED:
        return
    now = time.time()
    last = _last_said.get(text, 0)
    if now - last < cooldown:
        return
    _last_said[text] = now

    def _run():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


# -------------------- MediaPipe Pose --------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def get_pt(lm, idx, w, h):
    p = lm[idx]
    return (p.x * w, p.y * h)

def avg_pts(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

# -------------------- App State --------------------
MODE_SQUAT = 1
MODE_PUSHUP = 2
mode = MODE_SQUAT

# squat state
squat_count = 0
squat_stage = "up"
squat_score = 100.0
last_squat_rep = 0.0

# pushup state
pushup_count = 0
pushup_stage = "up"
pushup_score = 100.0
last_pushup_rep = 0.0

REP_COOLDOWN = 0.45

# Squat thresholds
KNEE_DOWN = 105
KNEE_UP = 165
MIN_DEPTH = 100

# Push-up thresholds (elbow angle)
ELBOW_DOWN = 95
ELBOW_UP = 160

# -------------------- NEW FEATURES: Goals / Sets / Rest / Logging --------------------
rep_goal_squat = 10
rep_goal_pushup = 10
sets_goal = 3

current_set = 1
rest_seconds = 20
resting = False
rest_end_time = 0.0

halfway_spoken = {"squat": False, "pushup": False}
session_log = []  # (timestamp, exercise, set, reps, score)

def start_rest():
    global resting, rest_end_time
    resting = True
    rest_end_time = time.time() + rest_seconds
    say(f"Rest for {rest_seconds} seconds", cooldown=0.1)

def save_history():
    if not session_log:
        return
    with open("workout_history.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "exercise", "set", "reps", "score"])
        for row in session_log:
            writer.writerow(row)

# -------------------- Camera --------------------
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 540)

start_time = time.time()
say("AI Fitness Coach started. Press 1 for squats, 2 for push ups.", cooldown=0.1)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Rest logic (skip counting while resting)
    if resting:
        remaining = int(rest_end_time - time.time())
        if remaining <= 0:
            resting = False
            if current_set < sets_goal:
                current_set += 1
                say(f"Set {current_set}", cooldown=0.1)
            else:
                say("Workout complete", cooldown=0.1)

        cv2.putText(frame, f"REST: {max(0, remaining)}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        # still show UI
        elapsed = int(time.time() - start_time)
        mode_name = "SQUATS" if mode == MODE_SQUAT else "PUSH-UPS"
        cv2.putText(frame, f"AI Fitness Coach | Mode: {mode_name}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Set: {current_set}/{sets_goal}", (650, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"Time: {elapsed}s", (650, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, "Keys: 1=Squat  2=Pushup  R=Reset  Q=Quit",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("AI Fitness Coach", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset everything
            squat_count, pushup_count = 0, 0
            squat_stage, pushup_stage = "up", "up"
            squat_score, pushup_score = 100.0, 100.0
            current_set = 1
            resting = False
            halfway_spoken = {"squat": False, "pushup": False}
            session_log.clear()
            say("Reset", cooldown=0.1)
        elif key == ord('1'):
            mode = MODE_SQUAT
            say("Squats mode", cooldown=0.1)
        elif key == ord('2'):
            mode = MODE_PUSHUP
            say("Push ups mode", cooldown=0.1)
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    feedback = []
    form_ok = True
    now = time.time()

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # LEFT
        l_sh = get_pt(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
        l_el = get_pt(lm, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
        l_wr = get_pt(lm, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)
        l_hip = get_pt(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
        l_knee = get_pt(lm, mp_pose.PoseLandmark.LEFT_KNEE.value, w, h)
        l_ank = get_pt(lm, mp_pose.PoseLandmark.LEFT_ANKLE.value, w, h)

        # RIGHT
        r_sh = get_pt(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
        r_el = get_pt(lm, mp_pose.PoseLandmark.RIGHT_ELBOW.value, w, h)
        r_wr = get_pt(lm, mp_pose.PoseLandmark.RIGHT_WRIST.value, w, h)
        r_hip = get_pt(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)
        r_knee = get_pt(lm, mp_pose.PoseLandmark.RIGHT_KNEE.value, w, h)
        r_ank = get_pt(lm, mp_pose.PoseLandmark.RIGHT_ANKLE.value, w, h)

        # Averages
        sh = avg_pts(l_sh, r_sh)
        hip = avg_pts(l_hip, r_hip)
        knee = avg_pts(l_knee, r_knee)
        ank = avg_pts(l_ank, r_ank)

        # Angles
        knee_ang = angle(hip, knee, ank)
        hip_ang = angle(sh, hip, knee)

        l_elbow = angle(l_sh, l_el, l_wr)
        r_elbow = angle(r_sh, r_el, r_wr)
        elbow_ang = (l_elbow + r_elbow) / 2

        # -------------------- SQUATS --------------------
        if mode == MODE_SQUAT:
            if knee_ang <= KNEE_DOWN:
                squat_stage = "down"

            if knee_ang >= KNEE_UP and squat_stage == "down" and (now - last_squat_rep) > REP_COOLDOWN:
                squat_count += 1
                squat_stage = "up"
                last_squat_rep = now
                say(f"Rep {squat_count}", cooldown=0.1)

                # Goal logic
                if squat_count == rep_goal_squat // 2 and not halfway_spoken["squat"]:
                    say("Halfway there", cooldown=0.1)
                    halfway_spoken["squat"] = True

                if squat_count >= rep_goal_squat:
                    say("Set complete", cooldown=0.1)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    session_log.append([ts, "SQUAT", current_set, squat_count, int(squat_score)])
                    squat_count = 0
                    halfway_spoken["squat"] = False
                    start_rest()

            # form checks
            if squat_stage == "down" and knee_ang > MIN_DEPTH:
                form_ok = False
                feedback.append("Go deeper")
                say("Go deeper", cooldown=2.0)

            if hip_ang < 65:
                form_ok = False
                feedback.append("Less forward lean")
                say("Keep your back straighter", cooldown=3.0)

            if abs(knee[0] - ank[0]) > 130:
                form_ok = False
                feedback.append("Align knee over foot")
                say("Keep your knee aligned", cooldown=3.0)

            squat_score = min(100.0, squat_score + 0.2) if form_ok else max(0.0, squat_score - 0.6)

            cv2.putText(frame, f"Knee: {int(knee_ang)}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"Hip: {int(hip_ang)}", (10, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # -------------------- PUSH-UPS --------------------
        else:
            if elbow_ang <= ELBOW_DOWN:
                pushup_stage = "down"

            if elbow_ang >= ELBOW_UP and pushup_stage == "down" and (now - last_pushup_rep) > REP_COOLDOWN:
                pushup_count += 1
                pushup_stage = "up"
                last_pushup_rep = now
                say(f"Rep {pushup_count}", cooldown=0.1)

                if pushup_count == rep_goal_pushup // 2 and not halfway_spoken["pushup"]:
                    say("Halfway there", cooldown=0.1)
                    halfway_spoken["pushup"] = True

                if pushup_count >= rep_goal_pushup:
                    say("Set complete", cooldown=0.1)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    session_log.append([ts, "PUSHUP", current_set, pushup_count, int(pushup_score)])
                    pushup_count = 0
                    halfway_spoken["pushup"] = False
                    start_rest()

            if pushup_stage == "down" and elbow_ang > 110:
                form_ok = False
                feedback.append("Lower your chest")
                say("Go lower", cooldown=2.5)

            hip_y = hip[1]
            sh_y = sh[1]
            if hip_y < sh_y - 35:
                form_ok = False
                feedback.append("Hips too high")
                say("Lower your hips", cooldown=3.0)
            elif hip_y > sh_y + 80:
                form_ok = False
                feedback.append("Hips sagging")
                say("Tighten core", cooldown=3.0)

            pushup_score = min(100.0, pushup_score + 0.2) if form_ok else max(0.0, pushup_score - 0.6)

            cv2.putText(frame, f"Elbow: {int(elbow_ang)}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # -------------------- UI --------------------
    elapsed = int(time.time() - start_time)
    mode_name = "SQUATS" if mode == MODE_SQUAT else "PUSH-UPS"

    cv2.putText(frame, f"AI Fitness Coach | Mode: {mode_name}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Set: {current_set}/{sets_goal}", (650, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    if mode == MODE_SQUAT:
        cv2.putText(frame, f"Reps: {squat_count}/{rep_goal_squat}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Stage: {squat_stage}", (300, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Score: {int(squat_score)}", (520, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    else:
        cv2.putText(frame, f"Reps: {pushup_count}/{rep_goal_pushup}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Stage: {pushup_stage}", (300, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, f"Score: {int(pushup_score)}", (520, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    cv2.putText(frame, f"Time: {elapsed}s", (650, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    y = 150
    if feedback:
        for msg in feedback[:3]:
            y += 35
            cv2.putText(frame, f"- {msg}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    else:
        cv2.putText(frame, "Good form ✅", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, "Keys: 1=Squat  2=Pushup  R=Reset  Q=Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("AI Fitness Coach", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        mode = MODE_SQUAT
        say("Squats mode", cooldown=0.1)
    elif key == ord('2'):
        mode = MODE_PUSHUP
        say("Push ups mode", cooldown=0.1)
    elif key == ord('r'):
        squat_count, pushup_count = 0, 0
        squat_stage, pushup_stage = "up", "up"
        squat_score, pushup_score = 100.0, 100.0
        current_set = 1
        resting = False
        halfway_spoken = {"squat": False, "pushup": False}
        session_log.clear()
        say("Reset", cooldown=0.1)

cap.release()
cv2.destroyAllWindows()
save_history()
print("Saved workout_history.csv")