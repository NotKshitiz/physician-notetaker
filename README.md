# Physician Notetaker — Medical NLP Pipeline

This project implements a **medical transcription and analysis pipeline** that performs:
1. **Named Entity Recognition (NER)** — Extracts Symptoms, Diagnosis, Treatment, etc.  
2. **Summarization** — Converts raw doctor-patient conversations into structured JSON reports.  
3. **Sentiment & Intent Analysis** — Identifies patient emotions and purpose of statements.  
4. **SOAP Note Generation** — Generates clinically formatted notes (Subjective, Objective, Assessment, Plan).


## Setup Instructions

### Clone or Download the Project
```bash
git clone https://github.com/NotKshitiz/physician-notetaker.git
cd physician-notetaker
```
## Create a Virtual Environment
```bash
python -m venv venv
```
## Clone or Download the Project
```bash
pip install spacy transformers torch scikit-learn tqdm pyyaml
```
## Download the spaCy Model
```bash
python -m spacy download en_core_web_trf
```

# Sample Output
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Back Pain", "Stiffness"],
  "Diagnosis": "Whiplash",
  "Treatment": ["Painkillers", "Physiotherapy"],
  "Prognosis": "Full recovery expected within six months"
}

# Project Questions & Answers
Q1. How would you handle ambiguous or missing medical data?
-> In real-world medical conversations, patients often speak in vague or incomplete ways — for example, saying “it hurts sometimes” without specifying where or how much.
To handle that, I’d design the pipeline to:

Detect uncertainty using patterns like “maybe,” “sometimes,” “I think,” etc.

Flag missing details (like unspecified body parts or durations) for review instead of making assumptions.

Leverage context — for example, if the patient talks about “pain” after mentioning their neck, the model can infer “neck pain.”

Optionally, use confidence scores for extracted entities, so anything below a threshold can be reviewed manually by a physician or QA system.

The idea isn’t to “fill in” missing data but to surface uncertainty clearly for better medical safety.


Q2. What pre-trained NLP models would you use for medical summarization?

I’d combine general-purpose and domain-specific models:

SpaCy for base entity extraction and syntactic parsing.

BioClinicalBERT or PubMedBERT (from Hugging Face) for understanding medical context, terminology, and relationships.

T5 or BART models fine-tuned for summarization to generate readable medical reports.

A hybrid approach — where rule-based filters handle structured extraction and transformers handle summarization — gives both accuracy and natural readability.


Q3. How would you fine-tune BERT for medical sentiment detection?

I’d start with a pre-trained BERT variant, like BioBERT or ClinicalBERT, since they already understand healthcare language. Then I’d fine-tune it on annotated medical dialogue datasets, where patient statements are labeled with emotions or intents such as “anxious,” “hopeful,” or “neutral.”

During fine-tuning:

Use patient-only utterances for sentiment labels.

Apply data augmentation (synonym replacement, tone rephrasing) to make the model robust.

Use cross-entropy loss with weighted classes (since “neutral” may dominate the dataset).

This way, the model learns to detect subtle emotions rather than just positive or negative polarity — which is important in clinical settings.

Q4. What datasets would you use for training a healthcare-specific sentiment model?

There aren’t many perfect off-the-shelf datasets, but I’d combine several sources:

i2b2 Clinical Notes Dataset — contains patient and doctor notes for context understanding.

MIMIC-III / MIMIC-IV — great for extracting emotional tone from real-world medical records.

Private annotated datasets — manually labeled from transcribed doctor-patient dialogues for fine-tuning emotional tone and intent.

Augmented data — generated from synthetic dialogues using LLMs to expand emotion classes safely.

This combination ensures the model understands both clinical structure and human emotion in medical speech.

Q5. How would you train an NLP model to map medical transcripts into SOAP format?

I’d train a sequence-to-sequence model (like T5 or FLAN-T5) to convert raw dialogues into structured SOAP notes.
Steps:

Prepare a dataset where transcripts are paired with their corresponding SOAP notes.

Use prompt-based fine-tuning — e.g., “Convert this transcript into a SOAP note.”

Add section-specific tokens like <SUBJECTIVE>, <ASSESSMENT>, etc., to guide the model structure.

Evaluate outputs using ROUGE scores and clinical validation from experts.

This method allows the model to learn both semantic structure and clinical language flow.

Q6. What rule-based or deep learning techniques would improve the accuracy of SOAP note generation?

A hybrid system would perform best:

Rule-based extraction for predictable patterns like vitals, medication names, and numeric data.

Deep learning models (like BERT or T5) for narrative understanding — turning “my back hurts” into “chief complaint: back pain.”

Post-processing with regex to ensure consistent formatting (e.g., every SOAP note must have the four sections).

Optionally, a validation layer that checks for medical completeness — e.g., no SOAP note should miss an “Assessment.”

This mix ensures both precision and readability, while maintaining clinical reliability.
