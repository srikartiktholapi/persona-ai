def test_build_scenario_context_uses_persona_activity_task():
    from app.pipeline.batch_analyser import build_scenario_context

    context = build_scenario_context(
        persona={"name": "Customer Success Lead"},
        activity={"activity_name": "Discovery call"},
        task={"task_name": "Explain ROI", "description": "Show value to the customer."},
    )

    assert "Customer Success Lead" in context
    assert "Discovery call" in context
    assert "Explain ROI" in context
    assert "Show value to the customer." in context
    assert "persona/activity/task" in context.lower()
