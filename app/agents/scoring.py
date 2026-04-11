'''from app.orchestrator.state import AgentState

def process(state: AgentState) -> dict:
    """Weighted overall score + trend memory"""
    return {}'''
import json
import requests


def score_audio(transcript_text, metrics, openai_key):

    api_url = "https://api.openai.com/v1/chat/completions"

    prompt = f"""
Evaluate the candidate's speech based on the following dimensions:

1. Clarity of Speech  
   - Assess pronunciation, coherence, and ease of understanding.

2. Confidence Level  
   - Evaluate confidence based on fluency, tone, and delivery.
   - Consider hesitation signals while scoring.

3. Communication Effectiveness  
   - Assess how clearly and logically the candidate conveys ideas.
   - Check for completeness, structure, and engagement.

4. Hesitation Analysis (Critical Factor)  
   - Use the provided `filler_score` and `pause_rate` to assess hesitation.
   - Higher values indicate more hesitation and reduced fluency.

IMPORTANT SCORING GUIDELINES:
- If `filler_score` and/or `pause_rate` are high:
  → Reduce confidence and communication scores accordingly.
- If both are low:
  → Increase confidence and communication scores.
- Ensure scores are logically consistent with hesitation levels.
- Do not ignore hesitation signals while evaluating.

Transcript:
{transcript_text}

Audio Metrics:
{metrics}

Return ONLY JSON:
{{
"speech_clarity_score out of 10": 0-10,
"confidence_score out of 10": 0-10,
"communication_score out of 10": 0-10,
"total_audio_score out of 30": 0-30,
"tone_quality":"",
"voice_quality_feedback":"",
"engagement_level":"",
"communication_style":"",
"language_fluency":"",
"professionalism_level":"",
"filler_word_usage":""
}}
"""


    payload = {
        "model":"gpt-4.1",
        "messages":[
            {"role":"system","content":"You are a speech evaluator."},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_object"}
    }

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers, json=payload)

    data = response.json()

    return json.loads(data["choices"][0]["message"]["content"])

def evaluate_answer_relevance(question, transcript, openai_key):

    import requests
    import json

    api_url = "https://api.openai.com/v1/chat/completions"

    prompt = f"""
Evaluate the candidate's introduction based on the following criteria:

Self-Introduction Clarity, Assess whether the candidate introduces themselves clearly, including name, role, and relevant background.
Product Introduction, Check if the candidate effectively introduces a product from their company, including its purpose and value proposition.
Professionalism, Evaluate the candidate's tone, language, and overall behavior to ensure it is formal, professional, and appropriate for a business context.
Language Appropriateness. Identify any use of informal, filler, or unnecessary words, and assess whether the communication is concise and precise.

Provide a detailed evaluation with:

Strengths
Areas of improvement
A final rating (e.g., Poor / Average / Good / Excellent)
Specific suggestions to improve the introduction
Ensure relevance_score is a numeric score out of 50 based on:
- clarity of self introduction
- product explanation
- professionalism
- language quality
Question:
{question}

Candidate Answer:
{transcript}

Return ONLY JSON:

{{
"relevance_score out of 50": 0-50,
"relevance": "relevant / partially_relevant / irrelevant",
"reason": "short explanation"
}}
"""

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You are an interview evaluator."},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers, json=payload)

    data = response.json()

    if "choices" not in data:
        return {"error": data}

    return json.loads(data["choices"][0]["message"]["content"])
def body_language_interpretation(body_metrics, openai_key):

    api_url = "https://api.openai.com/v1/chat/completions"

    prompt = f"""
You are evaluating a candidate's body language based ONLY on the given scores.

Do NOT re-calculate anything. Only interpret the values.

Scores Provided:
- Posture: {body_metrics["posture_score out of 5"]} out of 5
- Eye Contact: {body_metrics["eye_contact_score out of 5"]} out of 5
- Facial Expression: {body_metrics["expression_score out of 10"]} out of 10
- Total Score: {body_metrics["body_total"]} out of 20

Evaluation Rules:

1. Posture:
   - 0-2 → Poor (low confidence, unprofessional)
   - >2-3.5 → Average (slight instability)
   - >3.5-5 → Good (confident and professional)

2. Eye Contact:
   - 0-2 → Poor (lack of engagement)
   - >2-3.5 → Moderate (inconsistent engagement)
   - >3.5-5 → Strong (good engagement)

3. Facial Expression:
   - 0-4 → Poor (blank or disengaged)
   - >4-7 → Average (limited expressiveness)
   - >7-10 → Good (engaging and appropriate)

4. Overall Body Language (based on total score):
   - 0-8 → Poor
   - 9-14 → Average
   - 15-17 → Good
   - 18-20 → Excellent

Instructions:
- Keep output short and recruiter-style
- Do not mention numbers again in explanation
- Be professional and direct
- Give improvement suggestions ONLY if total score < 15

Metrics:
{body_metrics}

Return ONLY JSON:

{{
"posture_analysis": "",
"eye_contact_analysis": "",
"expression_analysis": "",
"overall_body_language": "Poor / Average / Good / Excellent",
"body_language_interpretation": "2-3 line professional summary",
"improvement_suggestions": ""
}}
"""

    payload = {
        "model":"gpt-4.1",
        "messages":[
            {"role":"system","content":"You are an interview evaluator."},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_object"}
    }

    headers = {
        "Authorization": f"Bearer {openai_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, headers=headers, json=payload)

    data = response.json()

    return json.loads(data["choices"][0]["message"]["content"])

def interpret_body_language(body_metrics, events, openai_key):

    import requests
    import json

    api_url = "https://api.openai.com/v1/chat/completions"

    prompt = f"""
You are evaluating a candidate's body language based ONLY on the given scores.

Do NOT re-calculate anything. Only interpret the values.

Scores Provided:
- Posture: {body_metrics["posture_score out of 5"]} out of 5
- Eye Contact: {body_metrics["eye_contact_score out of 5"]} out of 5
- Facial Expression: {body_metrics["expression_score out of 10"]} out of 10
- Total Score: {body_metrics["body_total"]} out of 20

Evaluation Rules:

1. Posture:
   - 0-2 → Poor (low confidence, unprofessional)
   - >2-3.5 → Average (slight instability)
   - >3.5-5 → Good (confident and professional)

2. Eye Contact:
   - 0-2 → Poor (lack of engagement)
   - >2-3.5 → Moderate (inconsistent engagement)
   - >3.5-5 → Strong (good engagement)

3. Facial Expression:
   - 0-4 → Poor (blank or disengaged)
   - >4-7 → Average (limited expressiveness)
   - >7-10 → Good (engaging and appropriate)

4. Overall Body Language (based on total score):
   - 0-8 → Poor
   - 9-14 → Average
   - 15-17 → Good
   - 18-20 → Excellent

Instructions:
- Keep output short and recruiter-style
- Do not mention numbers again in explanation
- Be professional and direct
- Give improvement suggestions ONLY if total score < 15

Metrics:
{body_metrics}

Detected Events:
{events}

Explain briefly what this means in an interview.
Return ONLY JSON:

{{
"posture_analysis": "",
"eye_contact_analysis": "",
"expression_analysis": "",
"overall_body_language": "Poor / Average / Good / Excellent",
"body_language_interpretation": "2-3 line professional summary",
"improvement_suggestions": ""
}}
"""

    payload = {
        "model":"gpt-4.1",
        "messages":[
            {"role":"system","content":"You are an interview evaluator."},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_object"}
    }

    headers = {
        "Authorization":f"Bearer {openai_key}",
        "Content-Type":"application/json"
    }

    r = requests.post(api_url, headers=headers, json=payload)

    data = r.json()

    return json.loads(data["choices"][0]["message"]["content"])
def calculate_body_score(body_metrics):

    posture = body_metrics.get("posture_score", 0)
    eye = body_metrics.get("eye_contact_score", 0)
    expression = body_metrics.get("expression_score", 0)

    posture_score = (posture / 10) * 5
    eye_score = (eye / 10) * 5
    expression_score = (expression / 10) * 10

    total = posture_score + eye_score + expression_score

    return {
        "posture_score out of 5": round(posture_score,2),
        "eye_contact_score out of 5": round(eye_score,2),
        "expression_score out of 10": round(expression_score,2),
        "body_total": round(total,2),
        "body_total_label": f"{round(total,2)} out of 20"
    }
def calculate_audio_total(audio_scores):

    clarity = audio_scores.get("speech_clarity_score", 0)
    confidence = audio_scores.get("confidence_score", 0)
    communication = audio_scores.get("communication_score", 0)

    clarity_score = (clarity / 10) * 10
    confidence_score = (confidence / 10) * 10
    communication_score = (communication / 10) * 10

    total = clarity_score + confidence_score + communication_score

    return {
        "speech_clarity score out of 10": round(clarity_score,1),
         "confidence score  out of 10": round(confidence_score,1),
        "communication score out of 10": round(communication_score,1),
        "audio_total": f"{round(total)}/30"
    }
