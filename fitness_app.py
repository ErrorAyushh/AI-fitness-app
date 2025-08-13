import time
import cv2
import numpy as np
import streamlit as st
import PoseModule as pm  # your poseDetector class

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Pose Gym", page_icon="💪", layout="wide")

if "run" not in st.session_state:
    st.session_state.run = False
if "count" not in st.session_state:
    st.session_state.count = 0
if "stage" not in st.session_state:
    st.session_state.stage = None
if "feedback" not in st.session_state:
    st.session_state.feedback = "Get Ready!"
if "hold_frames" not in st.session_state:
    st.session_state.hold_frames = 0

# ---------------------------
# Helpers
# ---------------------------
def reset_state():
    st.session_state.count = 0
    st.session_state.stage = None
    st.session_state.feedback = "Get Ready!"
    st.session_state.hold_frames = 0

def draw_progress_bar_v(img, x, y, w, h, pct, color=(0,255,0), bg=(200,200,200)):
    pct = int(max(0, min(100, pct)))
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, 2)
    filled = int(h * (pct/100.0))
    cv2.rectangle(img, (x, y+h-filled), (x+w, y+h), color, -1)
    cv2.putText(img, f"{pct}%", (x-5, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def put_hud(img, title, count, feedback, color=(255,255,255)):
    cv2.putText(img, f"{title}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(img, f"Reps: {int(count)}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    cv2.putText(img, f"{feedback}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if "Good" in feedback or "Perfect" in feedback else (0,0,255), 2)

# Angle → percentage helper
def interp(val, rng_in, rng_out):
    return int(np.interp(val, rng_in, rng_out))

# ---------------------------
# Exercise processors (PoseModule-based)
# Each returns the annotated frame
# ---------------------------

def process_pushups(detector, frame):
    # Your original push-up logic with PoseModule, kept same thresholds
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        elbow = detector.findAngle(img, 11, 13, 15)
        shoulder = detector.findAngle(img, 13, 11, 23)
        hip = detector.findAngle(img, 11, 23, 25)

        per = np.interp(elbow, (90, 160), (0, 100))
        bar = np.interp(elbow, (90, 160), (380, 50))

        # form check
        if elbow > 160 and shoulder > 40 and hip > 160:
            form_ok = True
        else:
            form_ok = False

        if form_ok:
            if per == 0:
                if elbow <= 90 and hip > 160:
                    st.session_state.feedback = "Up"
                    if st.session_state.stage != "up":
                        # half rep
                        st.session_state.stage = "up"
                        st.session_state.count += 0.5
                else:
                    st.session_state.feedback = "Fix Form"

            if per == 100:
                if elbow > 160 and shoulder > 40 and hip > 160:
                    st.session_state.feedback = "Down"
                    if st.session_state.stage != "down":
                        st.session_state.stage = "down"
                        st.session_state.count += 0.5
                else:
                    st.session_state.feedback = "Fix Form"
        else:
            st.session_state.feedback = "Fix Form"

        # Draw vertical bar
        cv2.rectangle(img, (580, 50), (600, 380), (0, 255, 0), 3)
        cv2.rectangle(img, (580, int(bar)), (600, 380), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (565, 430), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    put_hud(img, "Push-ups", st.session_state.count, st.session_state.feedback)
    return img

def process_squats(detector, frame):
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # Use LEFT HIP (23), LEFT KNEE (25), LEFT ANKLE (27) in PoseModule index mapping
        # But your code uses Mediapipe IDs: LEFT_HIP=23, LEFT_KNEE=25, LEFT_ANKLE=27 (same indices)
        knee_angle = detector.findAngle(img, 23, 25, 27)

        # rep logic
        if knee_angle > 160:
            st.session_state.stage = "up"
        if knee_angle < 100 and st.session_state.stage == "up":
            st.session_state.stage = "down"
            st.session_state.count += 1
            st.session_state.feedback = "Perfect Depth ✅"
        else:
            # Depth feedback around 90 degrees
            ideal_knee = 90
            form_score = max(0, min(100, 100 - abs(knee_angle - ideal_knee)))
            if form_score > 80:
                st.session_state.feedback = "Perfect Depth ✅"
            elif form_score > 50:
                st.session_state.feedback = "Go Lower ⬇️"
            else:
                st.session_state.feedback = "Too Shallow ⚠️"

        # Depth bar (horizontal)
        ideal_knee = 90
        form_score = max(0, min(100, 100 - abs(knee_angle - ideal_knee)))
        bar_w = int((form_score/100) * 300)
        bar_color = (0,255,0) if form_score>80 else ((0,165,255) if form_score>50 else (0,0,255))
        cv2.rectangle(img, (50, 50), (50+300, 80), (200,200,200), 2)
        cv2.rectangle(img, (50, 50), (50+bar_w, 80), bar_color, -1)
        cv2.putText(img, f"Knee: {int(knee_angle)}°", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)

    put_hud(img, "Squats", st.session_state.count, st.session_state.feedback)
    return img

def process_bicep_curls(detector, frame):
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # LEFT arm: shoulder(11), elbow(13), wrist(15)
        angle = detector.findAngle(img, 11, 13, 15)

        # Counter logic (your thresholds)
        if angle > 160:
            st.session_state.stage = "down"
        if angle < 30 and st.session_state.stage == "down":
            st.session_state.stage = "up"
            st.session_state.count += 1
            st.session_state.feedback = "Good rep!"
        else:
            st.session_state.feedback = "Full curl ⟳"

        # Show angle
        cv2.putText(img, f"Elbow: {int(angle)}°", (420, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    put_hud(img, "Bicep Curls", st.session_state.count, st.session_state.feedback)
    return img

def process_lunges(detector, frame):
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        # Right leg hip(24), knee(26), ankle(28)
        angle = detector.findAngle(img, 24, 26, 28)

        # smooth-ish feedback not strictly necessary in streamlit loop
        if angle > 160:
            if st.session_state.stage != "up":
                st.session_state.stage = "up"
                st.session_state.feedback = "Stand Tall"
        elif angle < 90:
            if st.session_state.stage == "up":
                st.session_state.stage = "down"
                st.session_state.count += 1
                st.session_state.feedback = "Good Lunge!"
        else:
            st.session_state.feedback = "Keep Going"

        # Depth percent bar
        max_a, min_a = 170, 80
        depth_pct = int(max(0, min(100, (max_a - angle) / (max_a - min_a) * 100)))
        draw_progress_bar_v(img, 40, 120, 30, 300, depth_pct)

        cv2.putText(img, f"Knee: {int(angle)}°", (90, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    put_hud(img, "Lunges", st.session_state.count, st.session_state.feedback)
    return img

def process_planks(detector, frame):
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Using indices like your code: 11 (L Shoulder), 23 (L Hip), 27 (L Ankle)
        shoulder_y = lmList[11][2]
        hip_y = lmList[23][2]
        ankle_y = lmList[27][2]

        hip_to_line = abs(hip_y - ((shoulder_y + ankle_y) / 2))

        if hip_to_line < 20:
            st.session_state.feedback = "Good Form ✅"
            st.session_state.hold_frames += 1
        else:
            st.session_state.feedback = "Hips too high/low ⚠️"
            st.session_state.hold_frames = 0

        seconds = st.session_state.hold_frames // 30  # approx 30 FPS
        cv2.putText(img, f"Hold: {seconds} sec", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    put_hud(img, "Planks (Hold Time)", st.session_state.count, st.session_state.feedback)
    return img

def process_jumping_jacks(detector, frame):
    img = detector.findPose(frame, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # Using shoulders and hips as proxies to decide "open" vs "closed"
        # lmList: [id, x, y]; nose is 0
        nose_y = lmList[0][2]
        left_sh_y = lmList[11][2]
        right_sh_y = lmList[12][2]

        # "Hands up": shoulders higher than nose is a loose proxy if wrists not used
        hands_up = (left_sh_y < nose_y) and (right_sh_y < nose_y)

        # legs apart: distance between hips in pixels compared to frame width
        left_hip_x = lmList[23][1]
        right_hip_x = lmList[24][1]
        legs_apart = abs(left_hip_x - right_hip_x) > 80  # heuristic in pixels

        if hands_up and legs_apart:
            if st.session_state.stage != "open":
                st.session_state.stage = "open"
                st.session_state.count += 1
                st.session_state.feedback = "Good! Keep Going"
        else:
            st.session_state.stage = "closed"
            st.session_state.feedback = "Open Arms & Legs"

    put_hud(img, "Jumping Jacks", st.session_state.count, st.session_state.feedback)
    return img

# Map exercise names to processors
EXERCISES = {
    "Push-ups": process_pushups,
    "Squats": process_squats,
    "Bicep Curls": process_bicep_curls,
    "Lunges": process_lunges,
    "Planks": process_planks,
    "Jumping Jacks": process_jumping_jacks,
}

# ---------------------------
# UI
# ---------------------------
st.title("💪 Pose Gym – AI Form & Rep Counter")
st.caption("Local webcam + OpenCV + your PoseModule. Select an exercise and hit Start.")

with st.sidebar:
    exercise = st.selectbox("Choose exercise", list(EXERCISES.keys()), index=0)
    colA, colB = st.columns(2)
    if colA.button("Start"):
        reset_state()
        st.session_state.run = True
    if colB.button("Stop"):
        st.session_state.run = False

    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown("- Ensure good lighting\n- Keep full body in frame\n- Press **Stop** to release the camera")

video_placeholder = st.empty()
metrics = st.empty()

# ---------------------------
# Main loop
# ---------------------------
def run_loop():
    detector = pm.poseDetector()
    cap = cv2.VideoCapture(0)
    # Set a reasonable size (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    processor = EXERCISES[exercise]

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view
        frame = processor(detector, frame)

        # Convert BGR to RGB for streamlit
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb, channels="RGB")

        # Light throttling (aim ~20-30 fps)
        time.sleep(0.03)

    cap.release()

# Run when toggled
if st.session_state.run:
    run_loop()
else:
    st.info("Select an exercise and press **Start** to begin.")
