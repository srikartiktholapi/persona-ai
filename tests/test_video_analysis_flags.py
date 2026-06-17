def test_visual_summary_should_be_disabled_without_face_landmarks():
    from app.agents.video import should_generate_visual_summary

    assert should_generate_visual_summary({
        "video_analysis_available": True,
        "face_landmarks_detected": False,
    }) is False


def test_visual_summary_should_be_enabled_with_face_landmarks():
    from app.agents.video import should_generate_visual_summary

    assert should_generate_visual_summary({
        "video_analysis_available": True,
        "face_landmarks_detected": True,
    }) is True
