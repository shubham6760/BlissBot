import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pre-trained medical language model
med_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')

# Example medical context and user query
medical_context = "Medical information related to a specific topic."
user_query = "Can you provide information about symptoms of COVID-19?"

# Encode the medical context and user query into embeddings
context_embedding = np.array(med_model.encode([medical_context]))
query_embedding = np.array(med_model.encode([user_query]))

# Calculate cosine similarity between the context and user query embeddings
similarity_score = cosine_similarity(context_embedding, query_embedding)[0][0]

medical_disease_embeddings = {
    'cancer': np.array([0.8, 0.7, 0.9]),
    'diabetes': np.array([0.7, 0.5, 0.8]),
    'heart disease': np.array([0.6, 0.4, 0.7]),
    'stroke': np.array([0.5, 0.3, 0.6]),
    'pneumonia': np.array([0.7, 0.6, 0.8]),
    'asthma': np.array([0.6, 0.5, 0.7]),
    'hypertension': np.array([0.7, 0.6, 0.8]),
    'Alzheimer\'s': np.array([0.8, 0.7, 0.9]),
    'COVID-19': np.array([0.9, 0.8, 0.9]),
    'arthritis': np.array([0.6, 0.5, 0.7]),
    'flu': np.array([0.7, 0.6, 0.8]),
    'depression': np.array([0.5, 0.4, 0.6]),
    'anxiety': np.array([0.4, 0.3, 0.5]),
}


doctors = ['surgeon', 'physician', 'specialist']
nurses = ['registered nurse', 'practitioner', 'midwife']
patients = ['inpatient', 'outpatient', 'pediatric patient']
medical_staff = ['paramedic', 'pharmacist', 'medical technician']


X = doctors
Y = nurses
A = patients
B = medical_staff
# Sample medical questions and correct answers
medical_questions = {
    "What is the primary symptom of COVID-19?": "Fever",
    "Which organ does the pancreas belong to?": "Digestive system",
    "Who discovered penicillin?": "Alexander Fleming",
    "What is the normal range for blood pressure?": "120/80 mmHg",
    "What is the main function of the respiratory system?": "Breathing"
}

# Hypothetical responses from the LLM for native and non-native speakers
native_responses = {
    "What is the primary symptom of COVID-19?": "Fever",
    "Which organ does the pancreas belong to?": "Digestive system",
    "Who discovered penicillin?": "Fleming",
    "What is the normal range for blood pressure?": "120/80 mmHg",
    "What is the main function of the respiratory system?": "Breathing"
}

non_native_responses = {
    "What is the primary symptom of COVID-19?": "Cough",
    "Which organ does the pancreas belong to?": "Cardiovascular system",
    "Who discovered penicillin?": "Flemming",
    "What is the normal range for blood pressure?": "100/60 mmHg",
    "What is the main function of the respiratory system?": "Oxygenation"
}
# Calculate the WEAT score
def s(w, X, Y):
    sim_X = np.mean([cosine_similarity(medical_word_embeddings[w].reshape(1, -1), medical_word_embeddings[x].reshape(1, -1)) for x in X])
    sim_Y = np.mean([cosine_similarity(medical_word_embeddings[w].reshape(1, -1), medical_word_embeddings[y].reshape(1, -1)) for y in Y])
    return sim_X - sim_Y

WEAT_score = sum([s(a, doctors, nurses) for a in patients]) - sum([s(b, doctors, nurses) for b in medical_staff])
print(f"WEAT score for medical terms: {WEAT_score}")