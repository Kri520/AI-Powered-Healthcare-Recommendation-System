from pathlib import Path
import difflib
import pickle

import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# ---------------------------------------------------------------- data
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions_df = pd.read_csv("dataset/precautions_df.csv")
workout_df = pd.read_csv("dataset/workout_df.csv")
description_df = pd.read_csv("dataset/description.csv")
medications_df = pd.read_csv("dataset/medications.csv")
diets_df = pd.read_csv("dataset/diets.csv")

# ---------------------------------------------------------------- model
# Preferred model first. Change the order to use Random Forest ("rf.pkl") instead.
MODEL_CANDIDATES = ["model/dtc.pkl", "model/rf.pkl", "model/svc.pkl"]
model = None
for _path in MODEL_CANDIDATES:
    if Path(_path).exists():
        with open(_path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded model: {_path}")
        break
if model is None:
    raise FileNotFoundError(
        "No trained model found in model/. Run the notebook "
        "'medical_recommendation_system_fixed.ipynb' (Run All Cells) first."
    )

# ---------------------------------------------------------------- label maps
# Same order as the training columns / LabelEncoder classes.
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# If train_model.py exported meta.pkl, prefer it: it is guaranteed to match the trained model.
_meta = Path("model/meta.pkl")
if _meta.exists():
    with open(_meta, "rb") as f:
        _m = pickle.load(f)
    symptoms_dict = _m["symptoms_dict"]
    diseases_list = _m["diseases_list"]


def helper(dis):
    desc = description_df[description_df["Disease"] == dis]["Description"]
    desc = " ".join(desc)

    pre = precautions_df[precautions_df["Disease"] == dis][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ].values
    pre = [p for p in pre[0]] if len(pre) else []

    med = list(medications_df[medications_df["Disease"] == dis]["Medication"].values)
    die = list(diets_df[diets_df["Disease"] == dis]["Diet"].values)
    wrk = list(workout_df[workout_df["disease"] == dis]["workout"].values)
    return desc, pre, med, die, wrk


def _key(text):
    """'Skin Rash', 'skin_rash', 'skinrash' -> 'skinrash'"""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def build_lookups():
    sym = {_key(k): k for k in symptoms_dict}
    dis = {_key(v): v for v in diseases_list.values()}
    return sym, dis


SYMPTOM_LOOKUP, DISEASE_LOOKUP = build_lookups()


def normalize(symptom):
    return symptom.strip("[]'\" ").strip()


def get_predicted_value(patient_symptoms):
    """
    Returns (disease or None, unknown_symptoms, disease_names_typed_by_mistake).
    Never raises KeyError for a wrong symptom.
    """
    vec = np.zeros((1, len(symptoms_dict)))
    unknown, diseases_typed = [], []
    for raw in patient_symptoms:
        k = _key(raw)
        if k in SYMPTOM_LOOKUP:
            vec[0, symptoms_dict[SYMPTOM_LOOKUP[k]]] = 1
        elif k in DISEASE_LOOKUP:
            diseases_typed.append(raw)
        else:
            unknown.append(raw)
    if vec.sum() == 0:
        return None, unknown, diseases_typed

    if hasattr(model, "feature_names_in_"):
        vec = pd.DataFrame(vec, columns=list(symptoms_dict))
    return diseases_list[int(model.predict(vec)[0])], unknown, diseases_typed


def suggest(word):
    """Closest valid symptom names for a misspelled word."""
    close = difflib.get_close_matches(_key(word), list(SYMPTOM_LOOKUP), n=3, cutoff=0.6)
    return [SYMPTOM_LOOKUP[c] for c in close]


# ---------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms = (request.form.get("symptoms") or "").strip()
        if not symptoms or symptoms == "Symptoms":
            return render_template(
                "index.html", message="Please enter at least one symptom."
            )

        user_symptoms = [normalize(x) for x in symptoms.split(",") if x.strip()]
        predicted_disease, unknown, diseases_typed = get_predicted_value(user_symptoms)

        problems = []
        if diseases_typed:
            problems.append(
                "You typed a disease name (" + ", ".join(diseases_typed) +
                "). Please enter SYMPTOMS such as: itching, skin rash, continuous sneezing."
            )
        for u in unknown:
            hint = suggest(u)
            problems.append(f"Unknown symptom '{u}'" + (f" - did you mean: {', '.join(hint)}?" if hint else "."))

        if predicted_disease is None:
            return render_template(
                "index.html",
                message=" ".join(problems) or "Please enter at least one valid symptom.",
            )

        dis_des, my_precautions, meds, rec_diet, wrk = helper(predicted_disease)
        return render_template(
            "index.html",
            predicted_disease=predicted_disease,
            dis_des=dis_des,
            my_precautions=my_precautions,
            medications=meds,
            my_diet=rec_diet,
            workout=wrk,
            message=(" ".join(problems) + " (these were ignored)") if problems else None,
        )
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/registration")
def registration():
    return render_template("registration.html")


@app.route("/login")
def login():
    return render_template("login.html")


if __name__ == "__main__":
    app.run(debug=True)
