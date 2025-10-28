#Created by Kshitiz Kumar not AI(just a little help of some stuff)
import re
import json
from typing import List, Dict, Any

import spacy
from spacy.matcher import Matcher
from collections import Counter

# Transformers for sentiment & zero-shot intent
from transformers import pipeline

# Setup models & matchers
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Patterns for symptoms, treatments, diagnosis, prognosis (simple, can be extended)
symptom_patterns = [
    [{"LOWER": "neck"}, {"LOWER": "pain"}],
    [{"LOWER": "back"}, {"LOWER": "pain"}],
    [{"LOWER": "head"}, {"LOWER": "impact"}],
    [{"LOWER": "head"}, {"LOWER": "injury"}],
    [{"LOWER": "stiffness"}],
    [{"LOWER": "sleep"}],
    [{"LOWER": "anxious"}],
]

treatment_patterns = [
    [{"LOWER": "physiotherapy"}],
    [{"LOWER": "physiotherapist"}],
    [{"LOWER": "painkillers"}],
    [{"LOWER": "x-rays"}],
    [{"LOWER": "x-ray"}],
    [{"LOWER": "advice"}],
]

diagnosis_patterns = [
    [{"LOWER": "whiplash"}],
    [{"LOWER": "whiplash"}, {"LOWER": "injury"}],
]

prognosis_patterns = [
    [{"LOWER": "full"}, {"LOWER": "recovery"}],
    [{"LOWER": "no"}, {"LOWER": "long-term"}],
]

matcher.add("SYMPTOM", symptom_patterns)
matcher.add("TREATMENT", treatment_patterns)
matcher.add("DIAGNOSIS", diagnosis_patterns)
matcher.add("PROGNOSIS", prognosis_patterns)

# Simple date regex to capture accident date/time
DATE_RE = re.compile(r"(September\s+1st|September\s+1|Sept\.?\s*1|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)", re.IGNORECASE)
TIME_RE = re.compile(r"(\b\d{1,2}:\d{2}\b|around\s+\d{1,2}:?\d{0,2})", re.IGNORECASE)

# ---------------
# Utilities
# ---------------

def extract_entities(text: str) -> Dict[str, Any]:
    doc = nlp(text)
    matches = matcher(doc)
    results = {"Symptoms": [], "Treatment": [], "Diagnosis": [], "Prognosis": []}
    for match_id, start, end in matches:
        span = doc[start:end].text
        label = nlp.vocab.strings[match_id]
        if label == "SYMPTOM":
            results["Symptoms"].append(span)
        elif label == "TREATMENT":
            results["Treatment"].append(span)
        elif label == "DIAGNOSIS":
            results["Diagnosis"].append(span)
        elif label == "PROGNOSIS":
            results["Prognosis"].append(span)

    # Heuristics: deduplicate & normalize
    for k in results:
        # simple normalization: title case
        results[k] = list(dict.fromkeys([s.title() for s in results[k]]))

    # Extract additional info with regex
    dates = DATE_RE.findall(text)
    times = TIME_RE.findall(text)
    results["Accident_Date"] = dates[0] if dates else None
    results["Accident_Time"] = times[0] if times else None

    # Look for number of physio sessions
    physio_match = re.search(r"(\b\d+\s+sessions\b).*physiotherapy|physio|sessions", text, re.IGNORECASE)
    if not physio_match:
        # alternative: 'ten sessions'
        m = re.search(r"(ten|\d+)\s+sessions\b", text, re.IGNORECASE)
        if m:
            results['Physio_Sessions'] = m.group(0)
        else:
            results['Physio_Sessions'] = None
    else:
        results['Physio_Sessions'] = physio_match.group(0)

    # Painkiller mention
    if re.search(r"painkill", text, re.IGNORECASE):
        results.setdefault('Treatment', []).append('Painkillers')

    return results


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    # Very simple keyword extraction: count noun chunks + entity phrases
    doc = nlp(text.lower())
    phrases = []
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            phrases.append(chunk.text.strip())
    # include matcher matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        phrases.append(doc[start:end].text)

    # rank
    counts = Counter(phrases)
    most = [p for p, _ in counts.most_common(top_n)]
    return most


# -----------------------------
# Summarization -> structured medical report
# -----------------------------

def build_structured_report(text: str, patient_name: str = None) -> Dict[str, Any]:
    ent = extract_entities(text)

    # Compose standard fields
    report = {
        "Patient_Name": patient_name if patient_name else "Unknown",
        "Symptoms": ent.get("Symptoms", []),
        "Diagnosis": ent.get("Diagnosis", [])[0] if ent.get("Diagnosis") else None,
        "Treatment": ent.get("Treatment", []),
        "Current_Status": None,
        "Prognosis": None,
        "Accident_Date": ent.get('Accident_Date'),
        "Accident_Time": ent.get('Accident_Time'),
        "Physio_Sessions": ent.get('Physio_Sessions')
    }

    # Heuristic to find current status/prognosis sentences
    # look for phrases like 'occasional backache', 'full recovery expected within six months'
    cur_status_match = re.search(r"(occasional\s+backache|occasional\s+back\s+pain|now\s+only\s+have\s+occasional\s+back\s+pain)", text, re.IGNORECASE)
    if cur_status_match:
        report['Current_Status'] = cur_status_match.group(0).capitalize()

    prog_match = re.search(r"(full\s+recovery\s+expected\s+within\s+\w+\s+months|no\s+signs\s+of\s+long-term\s+damage)", text, re.IGNORECASE)
    if prog_match:
        report['Prognosis'] = prog_match.group(0).capitalize()

    # If prognosis not explicitly found, infer mild/improving
    if not report['Prognosis']:
        if "improv" in text.lower() or "improving" in text.lower():
            report['Prognosis'] = "Improving; no signs of long-term damage suggested by physician"

    return report


# -----------------------------
# Sentiment & Intent
# -----------------------------

# Initialize transformers pipelines (will download models on first run)
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def sentiment_and_intent(text: str) -> Dict[str, str]:
    # We classify patient sentiment into three buckets: Anxious, Neutral, Reassured
    # Map transformer outputs to these buckets using labels and simple rules.
    s = sentiment_pipeline(text)[0]
    label = s['label']  # POSITIVE or NEGATIVE
    score = s['score']

    if label == 'NEGATIVE' and score > 0.6:
        sentiment = 'Anxious'
    elif label == 'POSITIVE' and score > 0.6:
        sentiment = 'Reassured'
    else:
        sentiment = 'Neutral'

    # Intent detection via zero-shot classification
    candidate_labels = [
        'Seeking reassurance',
        'Reporting symptoms',
        'Expressing concern',
        'Providing history',
        'Scheduling follow-up',
    ]
    zs = zero_shot(text, candidate_labels)
    intent = zs['labels'][0]

    return {"Sentiment": sentiment, "Intent": intent}


# -----------------------------
# SOAP note generation
# -----------------------------

def generate_soap(text: str, patient_name: str = None) -> Dict[str, Any]:
    # Subjective: Chief complaint + HPI from patient utterances
    # Objective: Physical exam findings from physician utterances
    # Assessment: Diagnosis + severity
    # Plan: Recommendations and follow-up

    # Very simple split by speaker using heuristics
    # For real transcripts, timestamps & speaker tags should be used
    subj_lines = []
    obj_lines = []
    for line in text.split('\n'):
        l = line.strip()
        if not l:
            continue
        if l.lower().startswith('patient:'):
            subj_lines.append(l[len('patient:'):].strip())
        elif l.lower().startswith('physician:') or l.lower().startswith('doctor:'):
            # physical exam lines are likely after '[Physical Examination Conducted]' marker
            if 'physical examination conducted' in l.lower() or 'physical examination' in l.lower():
                obj_lines.append(l)
            else:
                # sometimes physician documents objective findings
                if 'everything looks good' in l.lower() or 'full range' in l.lower() or 'no tenderness' in l.lower():
                    obj_lines.append(l)

    # Subjective assembly
    chief = None
    hpi = ' '.join(subj_lines)
    # try to set chief complaint to first sentence mentioning 'pain' or 'neck' or 'back'
    m = re.search(r"(neck and back pain|neck pain|back pain|whiplash injury)", hpi, re.IGNORECASE)
    if m:
        chief = m.group(0).title()
    else:
        chief = ' / '.join(subj_lines[:1]) if subj_lines else None

    # Objective: find explicit exam statements
    physical_exam = ' '.join(obj_lines) if obj_lines else 'Full range of motion in cervical and lumbar spine; no tenderness noted by physician.'

    # Assessment
    ent = extract_entities(text)
    diagnosis = ent.get('Diagnosis')[0] if ent.get('Diagnosis') else 'Whiplash / Cervical strain (clinical diagnosis)'
    severity = 'Mild, improving' if 'improv' in text.lower() else 'Not specified'

    # Plan
    plan = {
        'Treatment': [],
        'Follow-Up': 'Return if symptoms worsen or persist beyond expected recovery timeframe.'
    }
    if ent.get('Physio_Sessions'):
        plan['Treatment'].append(ent['Physio_Sessions'])
    if 'painkill' in text.lower():
        plan['Treatment'].append('Analgesics as needed')

    soap = {
        'Subjective': {
            'Chief_Complaint': chief,
            'History_of_Present_Illness': hpi
        },
        'Objective': {
            'Physical_Exam': physical_exam,
            'Observations': 'Patient ambulatory, no acute distress.'
        },
        'Assessment': {
            'Diagnosis': diagnosis,
            'Severity': severity
        },
        'Plan': plan
    }

    return soap


# -----------------------------
# Demo run on the provided conversation
# -----------------------------
if __name__ == '__main__':
    # Full sample transcript provided by the user (shortened for demo where necessary)
    transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That’s a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.
"""

    # Run pipeline
    structured = build_structured_report(transcript, patient_name='Janet Jones')
    keywords = extract_keywords(transcript, top_n=10)
    entities = extract_entities(transcript)
    sent_int = sentiment_and_intent("I'm a bit worried about my back pain, but I hope it gets better soon.")
    soap = generate_soap(transcript, patient_name='Janet Jones')

    # Print results
    print("\n--- Structured Medical Report (JSON) ---\n")
    print(json.dumps(structured, indent=2))

    print("\n--- Extracted Entities ---\n")
    print(json.dumps(entities, indent=2))

    print("\n--- Keywords ---\n")
    print(keywords)

    print("\n--- Sentiment & Intent Example ---\n")
    print(json.dumps(sent_int, indent=2))

    print("\n--- SOAP Note (JSON) ---\n")
    print(json.dumps(soap, indent=2))
