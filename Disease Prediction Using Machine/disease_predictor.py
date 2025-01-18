import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# List of symptoms and diseases
l1 = [
    'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
    'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
    'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
    'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
    'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
    'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
    'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',  # Removed foul_smell_of_urine
    'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
    'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
    'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
    'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

# Load datasets
try:
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure that Training.csv and Testing.csv are in the same directory.")
    exit()

# Check if any disease in the dataset is not in the predefined disease list
all_diseases = set(df['prognosis']).union(set(tr['prognosis']))
invalid_diseases = all_diseases.difference(disease)
if invalid_diseases:
    print(f"Warning: The following diseases are not in the predefined list: {invalid_diseases}")
    # Optionally, you can filter out rows with these invalid diseases from the dataset
    df = df[~df['prognosis'].isin(invalid_diseases)]
    tr = tr[~tr['prognosis'].isin(invalid_diseases)]

# Encode the 'prognosis' column
disease_mapping = {d: i for i, d in enumerate(disease)}
df['prognosis'] = df['prognosis'].map(disease_mapping)
tr['prognosis'] = tr['prognosis'].map(disease_mapping)

# Ensure labels are properly encoded as integers
if df['prognosis'].isnull().any() or tr['prognosis'].isnull().any():
    raise ValueError("Some diseases in the dataset do not match the predefined list.")

# Extract features (X) and labels (y)
X = df[l1]
y = df["prognosis"].astype(int)

X_test = tr[l1]
y_test = tr["prognosis"].astype(int)

# Function to predict using a model
def predict_with_model(clf, psymptoms):
    # Convert input symptoms to binary vector
    l2 = [1 if symptom in psymptoms else 0 for symptom in l1]
    inputtest = [l2]  # The model expects a 2D array
    predict = clf.predict(inputtest)
    return list(disease_mapping.keys())[list(disease_mapping.values()).index(predict[0])]

# Main function
def main():
    print("Disease Prediction System")
    print("Please enter up to 5 symptoms (separated by commas):")
    symptoms_input = input("Enter symptoms: ").strip().split(',')
    symptoms_input = [symptom.strip() for symptom in symptoms_input if symptom.strip()]

    if len(symptoms_input) == 0:
        print("No symptoms entered. Exiting.")
        return

    # Decision Tree
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(X, y)
    dt_result = predict_with_model(clf_tree, symptoms_input)
    print(f"Decision Tree Prediction: {dt_result}")

    # Random Forest
    clf_rf = RandomForestClassifier(random_state=42)
    clf_rf.fit(X, y)
    rf_result = predict_with_model(clf_rf, symptoms_input)
    print(f"Random Forest Prediction: {rf_result}")

    # Naive Bayes
    clf_nb = GaussianNB()
    clf_nb.fit(X, y)
    nb_result = predict_with_model(clf_nb, symptoms_input)
    print(f"Naive Bayes Prediction: {nb_result}")

if __name__ == "__main__":
    main()
