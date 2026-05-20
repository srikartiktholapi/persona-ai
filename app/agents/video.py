from app.orchestrator.state import AgentState
import cv2
import pandas as pd
import mediapipe as mp
import cv2
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic

from collections import deque

# Add a simple temporal smoothing buffer for scores
SCORE_SMOOTHING_WINDOW = 5

# Initialize global deques for smoothing
posture_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)
expression_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)
eye_contact_buffer = deque(maxlen=SCORE_SMOOTHING_WINDOW)

def process(state: AgentState) -> dict:
    """Live webcam frame mapping to Visual Performance Score with temporal smoothing"""
    features = state.get("recent_video_features", [])
    if not features or "raw_frames" not in features[0]:
        return {}
        
    frames = features[0]["raw_frames"]
    
    posture_vals = []
    expr_vals = []
    eye_vals = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True  # Enables iris landmarks 468-477
    ) as holistic:
        for idx, frame in enumerate(frames):
            if idx % 3 != 0:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            if results.pose_landmarks:
                posture_vals.append(calculate_posture_score(results.pose_landmarks.landmark))
            
            if results.face_landmarks:
                expr_vals.append(calculate_expression_score(results.face_landmarks.landmark))
                eye_vals.append(calculate_eye_contact(results.face_landmarks.landmark))

    posture_mean = pd.Series(posture_vals).mean() if posture_vals else 5.0
    expr_mean = pd.Series(expr_vals).mean() if expr_vals else 5.0
    eye_mean = pd.Series(eye_vals).mean() if eye_vals else 5.0

    # Append to smoothing buffers
    posture_buffer.append(posture_mean)
    expression_buffer.append(expr_mean)
    eye_contact_buffer.append(eye_mean)

    # Compute smoothed scores
    smoothed_posture = sum(posture_buffer) / len(posture_buffer)
    smoothed_expression = sum(expression_buffer) / len(expression_buffer)
    smoothed_eye_contact = sum(eye_contact_buffer) / len(eye_contact_buffer)
    
    body_score = (0.4 * smoothed_posture) + (0.3 * smoothed_eye_contact) + (0.3 * smoothed_expression)
    
    scores = state.get("scores", {})
    scores["visual_performance_score"] = round(float(body_score), 2)
    # Granular sub-scores for detailed feedback
    scores["posture_score"] = round(float(smoothed_posture), 2)
    scores["eye_contact_score"] = round(float(smoothed_eye_contact), 2)
    scores["expression_score"] = round(float(smoothed_expression), 2)
    
    # We dump raw_frames here to avoid saturating state history graph
    return {"recent_video_features": [], "scores": scores}


def calculate_posture_score(landmarks):
    """
    Neutral-baseline posture score (5.0 = average seated person).
    Earns bonuses for good indicators, penalties for bad ones.
    This prevents the 'always 9+' issue of the penalty-only model.
    """
    ls        = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    rs        = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip  = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value]
    nose      = landmarks[mp_holistic.PoseLandmark.NOSE.value]

    score = 5.0  # neutral/average baseline

    # ── 1. Shoulder tilt ───────────────────────────────────────────────────────
    shoulder_slope = abs(ls.y - rs.y)
    if shoulder_slope < 0.02:   score += 1.5   # very level → bonus
    elif shoulder_slope < 0.04: score += 0.5   # slightly tilted → small bonus
    elif shoulder_slope < 0.07: score -= 1.5   # noticeable tilt
    else:                       score -= 3.0   # severe tilt

    # ── 2. Shoulder visibility ─────────────────────────────────────────────────
    vis = (getattr(ls, "visibility", 1.0) + getattr(rs, "visibility", 1.0)) / 2
    if vis > 0.80:   score += 1.0   # shoulders clearly visible → good sign
    elif vis > 0.60: score += 0.0   # moderate
    elif vis > 0.40: score -= 1.5   # barely visible
    else:            score -= 3.0   # not detected = very bad

    # ── 3. Visible torso (shoulder–hip distance) ───────────────────────────────
    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_y      = (left_hip.y + right_hip.y) / 2
    torso_length   = hip_mid_y - shoulder_mid_y   # positive = upright
    if torso_length > 0.18:   score += 1.5   # clear torso → upright
    elif torso_length > 0.10: score += 0.5
    elif torso_length > 0.04: score -= 0.5
    else:                     score -= 2.0   # collapsed / too close

    # ── 4. Head–shoulder clearance (scale-invariant via torso ratio) ───────────
    head_gap = shoulder_mid_y - nose.y   # positive = head above shoulders
    # normalise by torso length to be camera-distance-independent
    ref      = max(torso_length, 0.05)
    ratio    = head_gap / ref
    if ratio > 1.2:    score += 1.0   # head well clear of shoulders
    elif ratio > 0.7:  score += 0.0   # acceptable
    elif ratio > 0.4:  score -= 1.5   # head too close to shoulders
    else:              score -= 3.0   # severe forward hunch

    return max(0.0, min(10.0, score))


def calculate_expression_score(face):
    """
    Expression score from mouth openness + eyebrow raise.

    FIX: MediaPipe y increases DOWNWARD (0=top, 1=bottom).
    Eyebrows are ABOVE eyes → eyebrow.y < eye.y.
    Old code used  max(0, eyebrow.y - eye.y) which is always 0.
    Correct:        max(0, eye.y     - eyebrow.y)  → positive when raised.
    """
    # Mouth openness (lips upper=13, lower=14)
    mouth_openness = abs(face[13].y - face[14].y)

    # Eyebrow raise — CORRECTED sign
    left_eyebrow_raise  = max(0.0, face[159].y - face[52].y)   # eye.y - eyebrow.y
    right_eyebrow_raise = max(0.0, face[386].y - face[282].y)
    avg_eyebrow_raise   = (left_eyebrow_raise + right_eyebrow_raise) / 2

    expression_metric = mouth_openness + avg_eyebrow_raise

    # Neutral resting face has expression_metric ~0.03–0.08; add a baseline
    # so a calm, engaged face doesn't score 0.
    score = max(0.0, min(10.0, expression_metric * 120 + 1.5))
    return score


import numpy as np
import math

def calculate_eye_contact(face):
    """
    Gaze estimation using iris landmarks (requires refine_face_landmarks=True).
    Measures iris deviation from eye-center — avoids the direction-sign bug
    that the ratio approach had with abs() on the denominator.

    Iris centers : left=468, right=473
    Eye corners  : left  outer=33,  inner=133
                   right outer=263, inner=362
    """
    # ── Fallback when iris landmarks not available (< 478 pts) ──────────────
    if len(face) < 478:
        left_eye  = face[33]
        right_eye = face[263]
        nose      = face[1]
        eye_center_x = (left_eye.x + right_eye.x) / 2
        return max(0, min(10, 10 - abs(eye_center_x - nose.x) * 40))

    # ── Iris center-deviation approach ──────────────────────────────────────
    # For each eye: compute midpoint of the two corners → that is the
    # "looking straight" reference.  Measure how far the iris deviates,
    # normalised by the eye width so face size doesn't matter.

    # Left eye (corners 33 & 133, iris 468)
    left_center_x   = (face[33].x + face[133].x) / 2
    left_eye_width  = max(abs(face[33].x - face[133].x), 0.001)
    left_dev        = abs(face[468].x - left_center_x) / left_eye_width

    # Right eye (corners 263 & 362, iris 473)
    right_center_x  = (face[263].x + face[362].x) / 2
    right_eye_width = max(abs(face[263].x - face[362].x), 0.001)
    right_dev       = abs(face[473].x - right_center_x) / right_eye_width

    avg_dev = (left_dev + right_dev) / 2

    # avg_dev ≈ 0.0–0.10 → looking at camera  (score 10–7.5)
    # avg_dev ≈ 0.10–0.25 → minor off-axis    (score 7.5–3.75)
    # avg_dev ≈ 0.25–0.40 → clearly off       (score 3.75–0)
    score = max(0.0, min(10.0, 10.0 - avg_dev * 25.0))
    return score


def analyze_video(video_path):

    posture_vals = []
    expr_vals = []
    eye_vals = []
    head_movement_vals = []

    cap = cv2.VideoCapture(video_path)

    prev_nose_x = None

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True  # Enables iris landmarks 468-477
    ) as holistic:

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # skip frames for speed
            if frame_count % 3 != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = holistic.process(rgb)

            if results.pose_landmarks:

                posture_vals.append(
                    calculate_posture_score(
                        results.pose_landmarks.landmark
                    )
                )

                nose = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.NOSE.value
                ]

                if prev_nose_x is not None:

                    movement = abs(nose.x - prev_nose_x)

                    head_movement_vals.append(
                        max(0, min(10, 10 - movement * 60))
                    )

                prev_nose_x = nose.x

            if results.face_landmarks:

                expr_vals.append(
                    calculate_expression_score(
                        results.face_landmarks.landmark
                    )
                )

                eye_vals.append(
                    calculate_eye_contact(
                        results.face_landmarks.landmark
                    )
                )

            else:

                expr_vals.append(5)
                eye_vals.append(5)

    cap.release()

    posture_mean = pd.Series(posture_vals).mean()
    expr_mean = pd.Series(expr_vals).mean()
    eye_mean = pd.Series(eye_vals).mean()

    if head_movement_vals:
        head_mean = pd.Series(head_movement_vals).mean()
    else:
        head_mean = 5

    body_score = (
        0.30 * posture_mean +
        0.25 * eye_mean +
        0.20 * head_mean +
        0.15 * expr_mean +
        0.10 * posture_mean
    )

    return {
        "posture_score": round(posture_mean, 2),
        "eye_contact_score": round(eye_mean, 2),
        "expression_score": round(expr_mean, 2),
        "head_stability": round(head_mean, 2),
        "body_language_score": round(body_score, 2)
    }
'''import os
import uuid
import requests

def download_video(video_url: str, save_dir="uploads"):
    os.makedirs(save_dir, exist_ok=True)

    video_id = str(uuid.uuid4())
    video_path = os.path.join(save_dir, f"{video_id}.mp4")

    with requests.get(video_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
'''
