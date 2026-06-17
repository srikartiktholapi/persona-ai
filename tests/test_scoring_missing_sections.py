import pytest
from app.pipeline.batch_analyser import compute_overall_score


def test_overall_score_penalizes_when_no_speech_detected():
    # When speech_detected=False, missing audio/text/relevance scores are penalized as 0.0
    overall = compute_overall_score(visual_score=8.0, audio_score=None, text_score=None, rel_score=None, speech_detected=False)
    assert overall == round(8.0 * 0.20 + 0.0 * 0.25 + 0.0 * 0.20 + 0.0 * 0.35, 2)


def test_overall_score_defaults_to_neutral_when_speech_detected():
    # When speech_detected=True, missing scores default to 5.0 (neutral)
    overall = compute_overall_score(visual_score=8.0, audio_score=None, text_score=None, rel_score=None, speech_detected=True)
    assert overall == round(8.0 * 0.20 + 5.0 * 0.25 + 5.0 * 0.20 + 5.0 * 0.35, 2)


def test_overall_score_uses_provided_scores_when_available():
    overall = compute_overall_score(visual_score=8.0, audio_score=6.0, text_score=7.0, rel_score=9.0)
    assert overall == round(8.0 * 0.20 + 6.0 * 0.25 + 7.0 * 0.20 + 9.0 * 0.35, 2)
