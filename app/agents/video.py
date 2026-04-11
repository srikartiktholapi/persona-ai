'''from app.orchestrator.state import AgentState

def process(state: AgentState) -> dict:
    """Posture, gaze, gestures, facial engagement"""
    return {}'''
import cv2
import pandas as pd
import mediapipe as mp
import os
import json
import openai

mp_holistic = mp.solutions.holistic


def calculate_posture_score(landmarks):
    ls = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
    rs = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]

    slope = abs(ls.y - rs.y)

    return max(0, min(10, 10 - slope * 120))


def calculate_expression_score(face):
    mouth = abs(face[13].y - face[14].y)
    return max(0, min(10, mouth * 150))


def calculate_eye_contact(face):
    left_eye = face[33]
    right_eye = face[263]
    nose = face[1]

    eye_center = (left_eye.x + right_eye.x) / 2
    deviation = abs(eye_center - nose.x)

    return max(0, min(10, 10 - deviation * 40))


def analyze_video(video_path):

    posture_vals = []
    expr_vals = []
    eye_vals = []
    head_movement_vals = []

    cap = cv2.VideoCapture(video_path)

    prev_nose_x = None

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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
