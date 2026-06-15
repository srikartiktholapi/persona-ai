from app.orchestrator.state import AgentState
import cv2
import pandas as pd
import mediapipe as mp
import pathlib
import requests

from collections import deque

try:
    mp_holistic = mp.solutions.holistic
except AttributeError:
    mp_holistic = None

try:
    from mediapipe.tasks.python import BaseOptions, vision
except Exception:
    BaseOptions = None
    vision = None

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_DIR = pathlib.Path(__file__).resolve().parents[2] / "models" / "mediapipe"
POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

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

    if mp_holistic is None:
        return {
            "recent_video_features": [],
            "scores": {
                "visual_performance_score": 0.0,
                "posture_score": 0.0,
                "eye_contact_score": 0.0,
                "expression_score": 0.0,
            },
        }
        
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
    ls        = landmarks[POSE_LEFT_SHOULDER]
    rs        = landmarks[POSE_RIGHT_SHOULDER]
    left_hip  = landmarks[POSE_LEFT_HIP]
    right_hip = landmarks[POSE_RIGHT_HIP]
    nose      = landmarks[POSE_NOSE]

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


def _ensure_mediapipe_model(path: pathlib.Path, url: str):
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(path, "wb") as model_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    model_file.write(chunk)


def _create_tasks_landmarkers():
    if BaseOptions is None or vision is None:
        raise RuntimeError("MediaPipe Tasks vision API is unavailable.")

    _ensure_mediapipe_model(POSE_MODEL_PATH, POSE_MODEL_URL)
    _ensure_mediapipe_model(FACE_MODEL_PATH, FACE_MODEL_URL)

    pose_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return (
        vision.PoseLandmarker.create_from_options(pose_options),
        vision.FaceLandmarker.create_from_options(face_options),
    )


def analyze_video_with_mediapipe_tasks(video_path):
    posture_vals = []
    expr_vals = []
    eye_vals = []
    head_movement_vals = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {
            "video_analysis_available": False,
            "video_analysis_status": "Video file could not be opened for MediaPipe visual analysis.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    try:
        pose_landmarker, face_landmarker = _create_tasks_landmarkers()
    except Exception as exc:
        cap.release()
        return {
            "video_analysis_available": False,
            "video_analysis_status": f"MediaPipe Tasks could not start: {exc}",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    prev_nose_x = None
    sampled_frames = 0
    landmark_frames = 0

    try:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:
                continue

            sampled_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_result = pose_landmarker.detect(image)
            face_result = face_landmarker.detect(image)

            if pose_result.pose_landmarks:
                landmark_frames += 1
                pose = pose_result.pose_landmarks[0]
                posture_vals.append(calculate_posture_score(pose))

                nose = pose[POSE_NOSE]
                if prev_nose_x is not None:
                    movement = abs(nose.x - prev_nose_x)
                    head_movement_vals.append(max(0, min(10, 10 - movement * 60)))
                prev_nose_x = nose.x

            if face_result.face_landmarks:
                landmark_frames += 1
                face = face_result.face_landmarks[0]
                expr_vals.append(calculate_expression_score(face))
                eye_vals.append(calculate_eye_contact(face))
            else:
                expr_vals.append(5.0)
                eye_vals.append(5.0)
    finally:
        cap.release()
        pose_landmarker.close()
        face_landmarker.close()

    if landmark_frames == 0:
        return {
            "video_analysis_available": False,
            "video_analysis_status": f"MediaPipe detected no face or body landmarks in {sampled_frames} sampled frames.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    posture_mean = pd.Series(posture_vals).mean() if posture_vals else 5.0
    expr_mean = pd.Series(expr_vals).mean() if expr_vals else 5.0
    eye_mean = pd.Series(eye_vals).mean() if eye_vals else 5.0
    head_mean = pd.Series(head_movement_vals).mean() if head_movement_vals else 5.0

    body_score = (
        0.30 * posture_mean +
        0.25 * eye_mean +
        0.20 * head_mean +
        0.15 * expr_mean +
        0.10 * posture_mean
    )

    return {
        "video_analysis_available": True,
        "video_analysis_status": "Visual analysis completed using MediaPipe Tasks.",
        "video_analysis_method": "mediapipe_tasks",
        "sampled_frames": sampled_frames,
        "landmark_frames": landmark_frames,
        "posture_score": round(posture_mean, 2),
        "eye_contact_score": round(eye_mean, 2),
        "expression_score": round(expr_mean, 2),
        "head_stability": round(head_mean, 2),
        "body_language_score": round(body_score, 2)
    }


def analyze_video_with_opencv_fallback(video_path):
    """
    Fallback visual analysis for environments where MediaPipe solutions are
    unavailable. It uses OpenCV face detection to estimate framing, gaze
    direction, and head stability, so good videos do not collapse to 0.0.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {
            "video_analysis_available": False,
            "video_analysis_status": "Video file could not be opened for visual analysis.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        cap.release()
        return {
            "video_analysis_available": False,
            "video_analysis_status": "OpenCV face detector could not be loaded.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    posture_vals = []
    eye_vals = []
    expression_vals = []
    centers = []
    frame_count = 0
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            continue

        h, w = frame.shape[:2]
        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
        cx = (x + fw / 2) / max(w, 1)
        cy = (y + fh / 2) / max(h, 1)
        face_ratio = fw / max(w, 1)
        centers.append((cx, cy))

        # Camera-facing proxy: a centered face usually means the speaker is
        # facing the camera. This is intentionally conservative.
        horizontal_dev = abs(cx - 0.5)
        eye_vals.append(max(3.0, min(10.0, 10.0 - horizontal_dev * 18.0)))

        # Framing/posture proxy: good score when the face is centered and not
        # extremely close or far from the camera.
        posture = 7.0
        posture -= min(3.0, abs(cy - 0.36) * 10.0)
        if face_ratio < 0.14:
            posture -= 1.5
        elif face_ratio > 0.42:
            posture -= 1.2
        else:
            posture += 0.8
        posture_vals.append(max(3.0, min(10.0, posture)))

        # Haar cascades cannot read expression reliably, so use a neutral
        # baseline rather than punishing the candidate.
        expression_vals.append(6.0)

    cap.release()

    if not centers:
        return {
            "video_analysis_available": False,
            "video_analysis_status": (
                f"No frontal face was detected in {processed_frames} sampled frames."
            ),
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    if len(centers) > 1:
        movement_vals = [
            abs(centers[idx][0] - centers[idx - 1][0]) + abs(centers[idx][1] - centers[idx - 1][1])
            for idx in range(1, len(centers))
        ]
        avg_movement = pd.Series(movement_vals).mean()
        head_mean = max(3.0, min(10.0, 10.0 - avg_movement * 30.0))
    else:
        head_mean = 6.0

    posture_mean = pd.Series(posture_vals).mean()
    eye_mean = pd.Series(eye_vals).mean()
    expr_mean = pd.Series(expression_vals).mean()
    body_score = round(
        0.35 * posture_mean + 0.30 * eye_mean + 0.20 * expr_mean + 0.15 * head_mean,
        2,
    )

    return {
        "video_analysis_available": True,
        "video_analysis_status": "Visual analysis completed using OpenCV fallback.",
        "video_analysis_method": "opencv_fallback",
        "posture_score": round(posture_mean, 2),
        "eye_contact_score": round(eye_mean, 2),
        "expression_score": round(expr_mean, 2),
        "head_stability": round(head_mean, 2),
        "body_language_score": body_score,
    }


def analyze_video(video_path):

    posture_vals = []
    expr_vals = []
    eye_vals = []
    head_movement_vals = []

    cap = cv2.VideoCapture(video_path)

    if mp_holistic is None:
        cap.release()
        return analyze_video_with_mediapipe_tasks(video_path)

    if not cap.isOpened():
        cap.release()
        return {
            "video_analysis_available": False,
            "video_analysis_status": "Video file could not be opened for visual analysis.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    prev_nose_x = None
    landmark_frames = 0

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
                landmark_frames += 1

                posture_vals.append(
                    calculate_posture_score(
                        results.pose_landmarks.landmark
                    )
                )

                nose = results.pose_landmarks.landmark[POSE_NOSE]

                if prev_nose_x is not None:

                    movement = abs(nose.x - prev_nose_x)

                    head_movement_vals.append(
                        max(0, min(10, 10 - movement * 60))
                    )

                prev_nose_x = nose.x

            if results.face_landmarks:
                landmark_frames += 1

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

    if landmark_frames == 0:
        return {
            "video_analysis_available": False,
            "video_analysis_status": "No face or body landmarks were detected in the recording.",
            "posture_score": None,
            "eye_contact_score": None,
            "expression_score": None,
            "head_stability": None,
            "body_language_score": None,
        }

    posture_mean = pd.Series(posture_vals).mean() if posture_vals else 5.0
    expr_mean = pd.Series(expr_vals).mean() if expr_vals else 5.0
    eye_mean = pd.Series(eye_vals).mean() if eye_vals else 5.0

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
        "video_analysis_available": True,
        "video_analysis_status": "Visual analysis completed.",
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
