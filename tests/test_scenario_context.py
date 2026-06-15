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


def test_build_analysis_prompt_prefers_selected_scenario_over_manual_question():
    from app.pipeline.batch_analyser import build_analysis_prompt

    prompt = build_analysis_prompt(
        selected_scenario={
            "persona": {"name": "Clueless Job Seeker"},
            "activity": {"activity_name": "E&I Call", "description": "Interview for eligibility and interest."},
            "task": {"task_name": "Do a thorough check on the eligibility and interest of the candidate for the job.", "description": "Assess whether the candidate fits the role."},
        },
        manual_prompt="Ignore the scenario and ask me a generic question.",
    )

    assert "Clueless Job Seeker" in prompt
    assert "E&I Call" in prompt
    assert "Do a thorough check on the eligibility and interest of the candidate for the job." in prompt
    assert "Ignore the scenario and ask me a generic question." not in prompt
