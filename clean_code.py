
import warnings
import threading
from tkinter import messagebox, Canvas, Frame, BOTH, RIGHT, Y
from pathlib import Path
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
import customtkinter as ctk

# quiet sklearn warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

# ---------------------------
# Symptom list and disease list
# ---------------------------
l1 = ['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

# ---------------------------
# Data files
# ---------------------------
TRAIN_CSV = "Training.csv"
TEST_CSV  = "Testing.csv"

def load_data():
    if not Path(TRAIN_CSV).exists() or not Path(TEST_CSV).exists():
        raise FileNotFoundError(f"Make sure {TRAIN_CSV} and {TEST_CSV} exist.")
    df = pd.read_csv(TRAIN_CSV)
    tr = pd.read_csv(TEST_CSV)
    mapping = {
        'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
        'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
        'Migraine':11,'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,
        'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,
        'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
        'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
        'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
        '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
        'Impetigo':40
    }
    df.replace({'prognosis': mapping}, inplace=True)
    tr.replace({'prognosis': mapping}, inplace=True)
    X = df[l1]
    y = df[["prognosis"]]
    X_test = tr[l1]
    y_test = tr[["prognosis"]]
    return X, y, X_test, y_test

try:
    X, y, X_test, y_test = load_data()
    DATA_LOADED = True
except Exception as e:
    print("Warning: could not load CSVs:", e)
    DATA_LOADED = False
    X = pd.DataFrame(np.zeros((1, len(l1))), columns=l1)
    y = pd.DataFrame([0])
    X_test = X.copy()
    y_test = y.copy()

# ---------------------------
# prediction helpers and models
# ---------------------------
def normalize_prediction(raw):
    if raw is None:
        return None
    try:
        idx = int(raw)
        if 0 <= idx < len(disease):
            return disease[idx]
        return str(raw)
    except Exception:
        pass
    try:
        if isinstance(raw, (list, tuple, np.ndarray)):
            if len(raw) == 0:
                return None
            return normalize_prediction(raw[0])
    except Exception:
        pass
    return str(raw)

def _to_df_vector(symptom_vector):
    try:
        return pd.DataFrame([symptom_vector], columns=l1)
    except Exception:
        return pd.DataFrame([symptom_vector])

def DecisionTree_predict(symptom_vector):
    try:
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y.values.ravel() if hasattr(y, "values") else np.ravel(y))
        try:
            from sklearn.metrics import accuracy_score
            y_pred = clf.predict(X_test)
            print("DecisionTree accuracy:", accuracy_score(y_test, y_pred))
        except Exception:
            pass
        X_input = _to_df_vector(symptom_vector)
        pred = clf.predict(X_input)
        return normalize_prediction(pred[0])
    except Exception as e:
        print("DecisionTree_predict error:", e)
        traceback.print_exc()
        return None

def RandomForest_predict(symptom_vector):
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        clf = clf.fit(X, np.ravel(y))
        try:
            from sklearn.metrics import accuracy_score
            y_pred = clf.predict(X_test)
            print("RandomForest accuracy:", accuracy_score(y_test, y_pred))
        except Exception:
            pass
        X_input = _to_df_vector(symptom_vector)
        pred = clf.predict(X_input)
        return normalize_prediction(pred[0])
    except Exception as e:
        print("RandomForest_predict error:", e)
        traceback.print_exc()
        return None

def NaiveBayes_predict(symptom_vector):
    try:
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb = gnb.fit(X, np.ravel(y))
        try:
            from sklearn.metrics import accuracy_score
            y_pred = gnb.predict(X_test)
            print("NaiveBayes accuracy:", accuracy_score(y_test, y_pred))
        except Exception:
            pass
        X_input = _to_df_vector(symptom_vector)
        pred = gnb.predict(X_input)
        return normalize_prediction(pred[0])
    except Exception as e:
        print("NaiveBayes_predict error:", e)
        traceback.print_exc()
        return None

def demo_predict(symptom_vector):
    idxs = [i for i, v in enumerate(symptom_vector) if v]
    if not idxs:
        return None
    return disease[idxs[0] % len(disease)]

# ---------------------------
# UI: scrollable left + right with history and larger fonts
# ---------------------------
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")

class ScrollableFrame(Frame):
    def __init__(self, master, **kwargs):
        Frame.__init__(self, master, **kwargs)
        self.canvas = Canvas(self, highlightthickness=0)
        self.scroll_frame = Frame(self.canvas)
        self.vsb = ctk.CTkScrollbar(self, orientation="vertical", command=self.yview)
        self.vsb.pack(side=RIGHT, fill=Y)
        self.canvas.pack(side="left", fill=BOTH, expand=True)
        self.canvas.create_window((0,0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def yview(self, *args):
        self.canvas.yview(*args)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        except Exception:
            pass

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # create fonts AFTER root exists to avoid 'Too early to use font' error
        self.HEADER_FONT = ctk.CTkFont(size=18, weight="bold")
        self.SECTION_FONT = ctk.CTkFont(size=14, weight="bold")
        self.LABEL_FONT = ctk.CTkFont(size=12)
        self.ENTRY_FONT = ctk.CTkFont(size=12)
        self.OPTIONMENU_WIDTH = 560

        self.title("Disease Predictor — Nilay")
        self.geometry("1150x740")
        self.minsize(960,640)

        # layout header + body
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color="#075E54")
        header.grid(row=0, column=0, sticky="we")
        header.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(header, text="  Disease Predictor", font=self.HEADER_FONT, text_color="white", anchor="w")
        title.grid(row=0, column=0, sticky="w", padx=12)

        body = ctk.CTkFrame(self, fg_color="#ECE5DD")
        body.grid(row=1, column=0, sticky="nswe", padx=12, pady=12)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1, uniform="a")
        body.grid_columnconfigure(1, weight=1, uniform="a")

        # left: scrollable form
        left_card = ctk.CTkFrame(body, fg_color="white", corner_radius=12)
        left_card.grid(row=0, column=0, sticky="nswe", padx=(12,8), pady=12)
        left_card.grid_rowconfigure(0, weight=1)
        left_card.grid_columnconfigure(0, weight=1)

        scroll_container = ScrollableFrame(left_card)
        scroll_container.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)
        scroll_container.scroll_frame.grid_columnconfigure(0, weight=1)

        # Patient Name
        self.name_var = ctk.StringVar()
        lbl_name = ctk.CTkLabel(scroll_container.scroll_frame, text="Patient Name", anchor="w", font=self.SECTION_FONT, text_color="#222222")
        lbl_name.grid(row=0, column=0, sticky="we", padx=20, pady=(18,6))
        self.name_entry = ctk.CTkEntry(scroll_container.scroll_frame, textvariable=self.name_var, placeholder_text="Enter patient name", font=self.ENTRY_FONT)
        self.name_entry.grid(row=1, column=0, sticky="we", padx=20, pady=(0,12))

        # fixed 5 symptom selectors
        self.sym_vars = []
        for i in range(5):
            bubble = ctk.CTkFrame(scroll_container.scroll_frame, corner_radius=10, fg_color="#F7F7F7")
            bubble.grid(row=2 + i*2, column=0, sticky="we", padx=20, pady=(10,6))
            bubble.grid_columnconfigure(0, weight=1)

            lbl = ctk.CTkLabel(bubble, text=f"Symptom {i+1}", anchor="w", font=self.LABEL_FONT, text_color="#222222")
            lbl.grid(row=0, column=0, sticky="w", padx=12, pady=(8,0))
            var = ctk.StringVar(value="None")
            opt = ctk.CTkOptionMenu(bubble, values=sorted(l1), variable=var, width=self.OPTIONMENU_WIDTH)
            opt.grid(row=1, column=0, sticky="we", padx=12, pady=(6,12))
            self.sym_vars.append(var)

        # right: results + model buttons + clear history
        right_card = ctk.CTkFrame(body, fg_color="white", corner_radius=12)
        right_card.grid(row=0, column=1, sticky="nswe", padx=(8,12), pady=12)
        right_card.grid_rowconfigure(3, weight=1)
        right_card.grid_columnconfigure(0, weight=1)

        result_title = ctk.CTkLabel(right_card, text="Prediction Result", anchor="w", font=self.SECTION_FONT, text_color="#222222")
        result_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12,6))

        # model buttons
        model_frame = ctk.CTkFrame(right_card, fg_color=None)
        model_frame.grid(row=1, column=0, sticky="we", padx=12, pady=(0,8))
        model_frame.grid_columnconfigure((0,1,2,3), weight=1)
        self.btn_dt = ctk.CTkButton(model_frame, text="Decision Tree", command=lambda: self.run_model("DecisionTree"))
        self.btn_rf = ctk.CTkButton(model_frame, text="Random Forest", command=lambda: self.run_model("RandomForest"))
        self.btn_nb = ctk.CTkButton(model_frame, text="Naive Bayes", command=lambda: self.run_model("NaiveBayes"))
        self.btn_clear = ctk.CTkButton(model_frame, text="Clear History", command=self.clear_history)
        self.btn_dt.grid(row=0, column=0, padx=6, sticky="we")
        self.btn_rf.grid(row=0, column=1, padx=6, sticky="we")
        self.btn_nb.grid(row=0, column=2, padx=6, sticky="we")
        self.btn_clear.grid(row=0, column=3, padx=6, sticky="we")

        # result history box (append-capable) - larger font
        self.result_box = ctk.CTkTextbox(right_card, width=1, height=260, font=ctk.CTkFont(size=12))
        self.result_box.grid(row=2, column=0, sticky="nswe", padx=12, pady=(0,8))
        self.result_box.configure(state="disabled")

        info_title = ctk.CTkLabel(right_card, text="Details & Advice (History)", anchor="w", font=self.SECTION_FONT, text_color="#222222")
        info_title.grid(row=3, column=0, sticky="w", padx=12, pady=(6,2))

        # details/history multiline textbox (append-capable) - larger font
        self.info_text = ctk.CTkTextbox(right_card, height=200, font=ctk.CTkFont(size=12))
        self.info_text.grid(row=4, column=0, sticky="nswe", padx=12, pady=(0,12))
        self.info_text.configure(state="disabled")

        # status
        self.status_var = ctk.StringVar(value="Ready")
        status = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w", font=ctk.CTkFont(size=11))
        status.grid(row=2, column=0, sticky="we", padx=12, pady=(0,8))

        self.history = []

    # helper to append text to CTkTextbox (keeps it readonly to user)
    def _append_textbox(self, textbox: ctk.CTkTextbox, text: str, separator: str = "\n\n"):
        try:
            textbox.configure(state="normal")
            current = textbox.get("0.0", "end").strip()
            if current:
                textbox.insert("end", separator)
            textbox.insert("end", text)
            textbox.see("end")
            textbox.configure(state="disabled")
        except Exception as e:
            print("Error appending textbox:", e)
            traceback.print_exc()

    def clear_history(self):
        self.result_box.configure(state="normal")
        self.result_box.delete("0.0", "end")
        self.result_box.configure(state="disabled")
        self.info_text.configure(state="normal")
        self.info_text.delete("0.0", "end")
        self.info_text.configure(state="disabled")
        self.history = []
        messagebox.showinfo("History cleared", "All previous results were cleared.")

    def set_info_text(self, text):
        self.info_text.configure(state="normal")
        self.info_text.delete("0.0", "end")
        self.info_text.insert("0.0", text)
        self.info_text.configure(state="disabled")

    def collect_symptoms(self):
        return [v.get() for v in self.sym_vars if v.get() and v.get() != "None"]

    def disable_controls(self, disabled=True):
        widgets = [self.btn_dt, self.btn_rf, self.btn_nb, self.btn_clear, self.name_entry]
        for w in widgets:
            try:
                w.configure(state="disabled" if disabled else "normal")
            except Exception:
                pass

    def run_model(self, model_key):
        symptoms = self.collect_symptoms()
        if not symptoms:
            messagebox.showinfo("No symptoms", "Please select at least one symptom.")
            return
        vec = [0] * len(l1)
        for s in symptoms:
            if s in l1:
                vec[l1.index(s)] = 1

        self.status_var.set(f"Running {model_key}...")
        self.disable_controls(True)
        t = threading.Thread(target=self._worker, args=(model_key, vec), daemon=True)
        t.start()

    def _worker(self, model_key, vec):
        try:
            if model_key == "DecisionTree":
                result = DecisionTree_predict(vec) if DATA_LOADED else demo_predict(vec)
            elif model_key == "RandomForest":
                result = RandomForest_predict(vec) if DATA_LOADED else demo_predict(vec)
            elif model_key == "NaiveBayes":
                result = NaiveBayes_predict(vec) if DATA_LOADED else demo_predict(vec)
            else:
                result = demo_predict(vec)
        except Exception as e:
            result = f"Error: {e}"
            print("Worker exception:", e)
            traceback.print_exc()

        if result is None:
            result = "No clear prediction"

        self.after(0, lambda: self._append_history_entry(model_key, result))

    def _append_history_entry(self, model_name, prediction):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_text = f"[{ts}] {model_name} → {prediction}"
        details = (
            f"[{ts}] Model: {model_name}\nPrediction: {prediction}\n\n"
            f"Selected symptoms:\n{', '.join(self.collect_symptoms())}\n\n"
            "Recommendation:\n• This is an automated prediction. Consult a clinician for confirmation."
        )
        self.history.append({"time": ts, "model": model_name, "prediction": prediction, "symptoms": self.collect_symptoms()})
        self._append_textbox(self.result_box, result_text)
        self._append_textbox(self.info_text, details)
        self.disable_controls(False)
        self.status_var.set("Ready")

    def reload_data(self):
        # kept for completeness (no left button shown)
        self.status_var.set("Reloading CSVs...")
        self.disable_controls(True)
        try:
            global X, y, X_test, y_test, DATA_LOADED
            X, y, X_test, y_test = load_data()
            DATA_LOADED = True
            messagebox.showinfo("Reload", "CSV files reloaded successfully.")
        except Exception as e:
            DATA_LOADED = False
            messagebox.showwarning("Reload failed", f"Could not reload CSVs: {e}")
        finally:
            self.status_var.set("Ready")
            self.disable_controls(False)

# ---------------------------
# entry point
# ---------------------------
if __name__ == "__main__":
    try:
        app = App()
        app.set_info_text("Select symptoms and tap a model. Results and details will appear in the right panel. Use Clear History to remove entries.")
        app.mainloop()
    except Exception as e:
        print("Fatal UI error:", e)
        traceback.print_exc()
