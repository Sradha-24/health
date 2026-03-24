from flask import Flask,render_template,redirect,session,url_for,request,jsonify
import sqlite3
import pickle
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import joblib
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_cancer_report(data, filename="breast_cancer_report.pdf"):
    doc = SimpleDocTemplate(f"static/{filename}")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Breast Cancer Risk Report", styles['Title']))
    content.append(Spacer(1, 20))

    for key, value in data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 10))

    doc.build(content)

    return filename

def generate_heart_report(data, filename="heart_report.pdf"):
    doc = SimpleDocTemplate(f"static/{filename}")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Heart Disease Risk Report", styles['Title']))
    content.append(Spacer(1, 20))

    for key, value in data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 10))

    doc.build(content)

    return filename

def generate_hypertension_report(data, filename="hypertension_report.pdf"):
    doc = SimpleDocTemplate(f"static/{filename}")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Hypertension Risk Report", styles['Title']))
    content.append(Spacer(1, 20))

    for key, value in data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 10))

    doc.build(content)

    return filename

def generate_diabetes_report(data,filename="diabetes_report.pdf"):

    doc = SimpleDocTemplate(f"static/{filename}")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Diabetes Risk Report", styles['Title']))
    content.append(Spacer(1, 20))

    for key, value in data.items():
        content.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
        content.append(Spacer(1, 10))

    doc.build(content)

    return filename


#Load the models
model=pickle.load(open('models/diabetes_model.pkl','rb'))
model_columns=pickle.load(open('models/model_columns.pkl','rb'))
training_sample=pd.read_csv('models/training_sample.csv')

#load models for hypertension
hyper_model=pickle.load(open('models/hypertension_model.pkl','rb'))
hyper_sample = pd.read_csv('models/hyper_training_sample.csv')


#load models for heartdisease
heart_model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')
features = joblib.load("models/features.pkl")

# Dummy dataset for LIME
df = pd.read_csv("heart_disease_uci.csv")
df.replace('?', np.nan, inplace=True)
df.rename(columns={'thalch': 'thalach'}, inplace=True)
df.drop(['id', 'dataset'], axis=1, inplace=True)

num_cols = ['age','trestbps','chol','thalach','oldpeak','ca']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)

df['target'] = df['num'].apply(lambda x: 1 if int(x) > 0 else 0)
df.drop('num', axis=1, inplace=True)

df['family_history'] = np.random.choice([0.15,0.35,0.65,1.2,2.0], len(df))

df = pd.get_dummies(df, drop_first=True)

X_data = df[features].values
def heart_predict_wrapper(data):
    scaled_data = scaler.transform(data)
    # Handle any potential NaNs from scaling
    scaled_data = np.nan_to_num(scaled_data) 
    return heart_model.predict_proba(scaled_data)

heart_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_data,
    feature_names=features,
    class_names=["No Disease", "Disease"],
    mode="classification"
)

# Initialize LIME Explainer (Do this once after loading the model)
explainer=lime.lime_tabular.LimeTabularExplainer(
    training_data=training_sample.values,
    feature_names=list(training_sample.columns),
    class_names=['Healthy','Diabetic'],
    mode='classification'
)
# Initialize the Hypertension Explainer
hyper_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=hyper_sample.values,
    feature_names=list(hyper_sample.columns),
    class_names=['Negative', 'Positive'], # 0 = Normal, 1 = Hypertensive
    mode='classification'
)

# --- 1. Load the Breast Cancer Model (at the top of app.py) ---
cancer_model = joblib.load('models/cancer_model.pkl')
cancer_features = joblib.load('models/cancer_features.pkl')
cancer_sample = pd.read_csv('models/cancer_training_sample.csv')

cancer_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=cancer_sample.values,
    feature_names=cancer_features,
    class_names=['Benign', 'Malignant'],
    mode='classification'
)

app=Flask(__name__)

app.secret_key='your_secret_key'
def get_connection():
    conn=sqlite3.connect('users.db')
    conn.row_factory=sqlite3.Row
    return conn
    


@app.route('/')
def base():
    return render_template("index.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        conn=get_connection()
        cursor=conn.cursor()

        
        name=request.form['name']
        email=request.form['email']
        password=request.form['password']
        confirm_password=request.form['confirm_password']

    
        if name.strip()=="" or email.strip()=="" or password.strip()=="" or confirm_password.strip()=="":
           return jsonify({"message":"field cannot be empty"}),400
        
        if password != confirm_password:
           return "password do not match"
        
        cursor.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",(name,email,password))
        conn.commit()
        conn.close()
        
        return redirect('/login')
    
    return render_template("register.html")

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        conn=get_connection()
        cursor=conn.cursor()
        username=request.form.get('username')
        password=request.form.get('password')


        cursor.execute("SELECT * FROM users WHERE email=?",(username,))
        row=cursor.fetchone()

        if row is None or row['password'] != password:
            return "Invalid email or password",401
        
        # Store user info in session
        session['user_email'] = row['email']
        session['user_name'] = row['name']
      
        if row['name']=='admin':
            return redirect('/admin')
        
        return redirect('/dashboard')


    return render_template('login.html')

@app.route('/admin')
def admin():
    conn=get_connection()
    cursor=conn.cursor()
    cursor=cursor.execute("SELECT * FROM users")

    rows=cursor.fetchall()
    conn.close()

    return render_template("admin_dashboard.html",users=rows)

@app.route('/predict_diabetes')
def predict_diabetes_page():
    return render_template('diabetes_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_email' not in session:
        return redirect('/login')
    
    try:
        # 1. Capture ALL data from form
        age = float(request.form.get('Age', 0))
        glucose = float(request.form.get('Glucose', 0))
        bp = float(request.form.get('BloodPressure', 0))
        smoking = float(request.form.get('smoking_history', 0))
        pedigree = float(request.form.get('DiabetesPedigreeFunction', 0))
        
        weight=float(request.form.get('weight',0))
        height=float(request.form.get('height',0))

        #calculate BMI
        if height>0:
            height=height/100 #convert cm to m
            bmi=round(weight/(height ** 2), 2)
        else:
            bmi=0
       

        # 2. Create the EXACT 6-feature list for the model
        # The order must match your training_sample.csv exactly
        model_input = [age, bmi, glucose, bp, smoking, pedigree]
        final_features = np.array(model_input).reshape(1, -1)

        # 3. Predict using the 6 features
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0][1]
        
        # 4. Generate LIME Explanation (Using the 6 features)
        exp = explainer.explain_instance(
            data_row=np.array(model_input), 
            predict_fn=model.predict_proba
        )

        # 5. Convert LIME plot to image
        fig = exp.as_pyplot_figure()
        plt.title("Health Factor Analysis", fontsize=14)
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        data_url = base64.b64encode(buf.getvalue()).decode()
        lime_chart = f"data:image/png;base64,{data_url}"

        # 6. UI Formatting
        result = "Positive (Diabetic Risk)" if prediction[0] == 1 else "Negative (Healthy)"
        risk_score = round(float(probability) * 100, 2)
        bar_color = "#d9534f" if risk_score > 50 else "#5cb85c"

        # SAVE TO HISTORY TABLE
        conn = get_connection()
        # We save the Glucose provided by the user
        conn.execute(
            "INSERT INTO history (email, date, glucose, score, result) VALUES (?, ?, ?, ?, ?)",
            (session['user_email'], datetime.now().strftime("%Y-%m-%d %H:%M"), 
             glucose, risk_score, result)
        )
        conn.commit()

        # Fetch updated history
        history = conn.execute(
            "SELECT date, glucose, score, result FROM history WHERE email = ? ORDER BY id DESC", 
            (session['user_email'],)
        ).fetchall()
        conn.close()

        
        report_data = {
            "Patient Name": session['user_name'],

            "Age": age,
            "Weight (kg)": weight,
            "Height (cm)": request.form.get('height'),
            "BMI": bmi,

            "Glucose (mg/dL)": glucose,
            "Blood Pressure (mm Hg)": bp,

            "Smoking History": request.form.get('smoking_history'),
            "Family Diabetes History": pedigree,

            "Prediction": result,
            "Final Risk (%)": risk_score
            }

        report_file = generate_diabetes_report(report_data)

        return render_template('dashboard.html', 
                               name=session['user_name'], 
                               history=history, 
                               prediction=result, 
                               score=risk_score, 
                               color=bar_color, 
                               explanation_chart=lime_chart,
                               report_file=report_file )

    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/dashboard')
def dashboard():
    if 'user_email' not in session:
        return redirect('/login')
    # This now returns the menu selection page
    return render_template("home.html", name=session['user_name'])


@app.route('/diabetes')
def diabetes_module():
    if 'user_email' not in session:
        return redirect('/login')
    
    conn = get_connection()
    history = conn.execute(
        "SELECT date, glucose, score, result FROM history WHERE email = ? ORDER BY id DESC", 
        (session['user_email'],)
    ).fetchall()
    conn.close()
    # This returns your existing dashboard layout but specifically for Diabetes
    return render_template("dashboard.html", name=session['user_name'], history=history)

@app.route('/hypertension')
def hypertension_module():
    if 'user_email' not in session:
        return redirect('/login')
    
    conn=get_connection()
    history=conn.execute("SELECT date,score,result,salt_intake FROM hypertension_history where email=? ORDER BY id DESC",(session['user_email'],)).fetchall()
    conn.close()
    return render_template('hypertension.html',name=session['user_name'],history =history)

@app.route('/predict_hypertension',methods=['POST'])
def predict_hypertension():
    if 'user_email' not in session:
        return redirect('/login')
    try:
        # 1. Collect inputs in the EXACT order your XGBoost model expects
        # Age, Salt_Intake, Stress_Score, BP_History, Sleep_Duration, BMI, Medication, Family_History, Exercise_Level, Smoking_Status
        family_input = int(request.form.get('Family_History'))

        family_map = {
           0: 0.0,
           1: 0.5,
           2: 1.0
        }

        family_risk = family_map.get(family_input, 0)
        
        features = [
            float(request.form.get('Age')),
            float(request.form.get('Salt_Intake')),
            float(request.form.get('Stress_Score')),
            float(request.form.get('BP_History')),
            float(request.form.get('Sleep_Duration')),
            float(request.form.get('BMI')),
            float(request.form.get('Medication')),
            family_risk,
            float(request.form.get('Exercise_Level')),
            float(request.form.get('Smoking_Status'))
        ]

        # Convert to numpy array for the model
        final_features=np.array(features).reshape(1,-1)

        #run prediction
        prediction=hyper_model.predict(final_features)[0]
        probability = hyper_model.predict_proba(final_features)[0][1]

        features_no_fam = features.copy()
        features_no_fam[7] = 0  # index of Family_History

        final_features_no_fam = np.array(features_no_fam).reshape(1, -1)

        prob_without = hyper_model.predict_proba(final_features_no_fam)[0][1]
        # 3. Generate LIME Explanation
        # hyper_explainer should be initialized at the top of your app.py
        exp = hyper_explainer.explain_instance(
            data_row=np.array(features), 
            predict_fn=hyper_model.predict_proba
        )

        # 4. Convert LIME Plot to Base64 String for HTML
        fig = exp.as_pyplot_figure()
        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        explanation_chart = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

        # 5. Format Results
        result_text = "Positive (Hypertensive Risk)" if prediction == 1 else "Negative (Healthy)"
        if probability < 0.7:
            adjusted_prob = probability + (family_risk * 0.25)
        else:
            adjusted_prob = probability + (family_risk * 0.05)
        adjusted_prob = min(adjusted_prob, 1)

        risk_score = round(adjusted_prob * 100, 2)
        risk_without = round(prob_without * 100, 2)
        color = "#d9534f" if risk_score > 50 else "#5cb85c"

        # 6. Save to the NEW hypertension_history table
        conn = get_connection()
        conn.execute('''
            INSERT INTO hypertension_history (email, date, salt_intake, score, result) 
            VALUES (?, ?, ?, ?, ?)
        ''', (
            session['user_email'], 
            datetime.now().strftime("%Y-%m-%d %H:%M"), 
            float(request.form.get('Salt_Intake')), 
            risk_score, 
            result_text
        ))

        history=conn.execute("SELECT date, score, result, salt_intake FROM hypertension_history WHERE email=? ORDER BY id DESC",
        (session['user_email'],)).fetchall()
        conn.commit()
        conn.close()

        difference = risk_score - risk_without

        if difference > 10:
           hereditary_msg = f"Family history significantly increases your hypertension risk by {round(difference,2)}%."
        elif difference > 3:
           hereditary_msg = f"Family history moderately increases your risk by {round(difference,2)}%."
        else:
           hereditary_msg = "Family history has limited impact because other clinical factors dominate the prediction."

        
        report_data = {
         "Patient Name": session['user_name'],
         "Age": request.form.get('Age'),
         "BMI": request.form.get('BMI'),
         "Salt Intake": request.form.get('Salt_Intake'),
         "Stress Score": request.form.get('Stress_Score'),
         "Sleep Duration": request.form.get('Sleep_Duration'),
         "Smoking Status": request.form.get('Smoking_Status'),
         "Exercise Level": request.form.get('Exercise_Level'),
         "Family History": request.form.get('Family_History'),
         "BP History": request.form.get('BP_History'),
         "Medication": request.form.get('Medication'),
         "Final Risk (%)": risk_score,
         "Without Family History (%)": risk_without,
         "Result": result_text
       }

        report_file = generate_hypertension_report(report_data)

        # 7. Send to result.html (reusing your existing results page)
        return render_template('hypertension.html', 
                           name=session['user_name'],
                           history=history,
                           prediction=result_text, 
                           score=risk_score, 
                           color=color, 
                           explanation_chart=explanation_chart,
                           risk_without=risk_without,
                           hereditary_msg=hereditary_msg,  
                           report_file=report_file,
                           show_result=True)
    except Exception as e:
        return f"Error during Hypertension prediction: {str(e)}"

@app.route('/heartdisease')
def heartdisease():
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('heartdisease.html',name=session['user_name'])

def get_family_risk_score(level):
    if level == 2:
        return 0.6   # High hereditary risk (parents/siblings)
    elif level == 1:
        return 0.3   # Medium risk (grandparents etc.)
    else:
        return 0.0   # No risk


@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    # Base numeric input

    family_history_level = int(request.form['family_history'])
    family_risk = get_family_risk_score(family_history_level)

    input_data = {
        'age': float(request.form['age']),
        'trestbps': float(request.form['trestbps']),
        'chol': float(request.form['chol']),
        'thalach': float(request.form['thalach']),
        'oldpeak': float(request.form['oldpeak']),
        'ca': float(request.form['ca']),
        'family_history': family_risk
  
    }

    

    # Create empty feature vector
    input_df = pd.DataFrame(np.zeros((1, len(features))), columns=features)
    input_df = input_df.reindex(columns=features, fill_value=0)


    # Fill numeric
    for key in input_data:
        input_df.at[0, key] = input_data[key]
    
    def set_dummy(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_df.columns:
            input_df.at[0, col_name] = 1

    set_dummy("sex", request.form['sex'])
    set_dummy("cp", request.form['cp'])
    set_dummy("fbs", request.form['fbs'])
    set_dummy("restecg", request.form['restecg'])
    set_dummy("exang", request.form['exang'])
    set_dummy("slope", request.form['slope'])
    set_dummy("thal", request.form['thal'])

    scaled = scaler.transform(input_df)
    scaled = np.nan_to_num(scaled)

    # Predict
    prob_with = heart_model.predict_proba(scaled)[0][1]

    input_df_no_family = input_df.copy()
    input_df_no_family['family_history'] = 0

    scaled_no_family = scaler.transform(input_df_no_family)
    scaled_no_family = np.nan_to_num(scaled_no_family)

    prob_without = heart_model.predict_proba(scaled_no_family)[0][1]

    risk_with = round(prob_with * 100, 2)
    risk_without = round(prob_without * 100, 2)

    risk = risk_with
    print("NaN check:", np.isnan(scaled).any())

    # 2. LIME (Explainer needs un-scaled row + our wrapper)
    exp = heart_explainer.explain_instance(
        data_row=input_df.values[0], # The raw numbers before scaling
        predict_fn=heart_predict_wrapper # Uses the wrapper we made above
    )

    top_features = exp.as_list()[:3]

    explanation = exp.as_list()[:5]

    explanation_text = "Key factors affecting your risk are: "

    for feature, value in top_features:
        explanation_text += f"{feature}, "

    explanation_text = explanation_text.rstrip(", ") + "."

    if family_risk > 0:
        explanation_text += " Family history has a significant influence on the prediction."
    
    difference = risk - risk_without

    if difference > 15:
      family_msg = f"Family history significantly increases heart disease risk by {round(difference,2)}%."
    elif difference > 5:
      family_msg = f"Family history moderately increases risk by {round(difference,2)}%."
    else:
      family_msg = "Family history has minimal impact compared to other clinical factors."

    report_data = {
    "Age": input_data['age'],
    "Resting BP": input_data['trestbps'],
    "Cholesterol": input_data['chol'],
    "Max Heart Rate": input_data['thalach'],
    "Oldpeak": input_data['oldpeak'],
    "Number of Major Vessels": input_data['ca'],
    "Family History Impact": family_msg,
    "Family History": family_history_level,
    "Final Risk (%)": risk,
    "Without Family History (%)": risk_without
}


    report_file = generate_heart_report(report_data)
    return render_template("heartdisease.html", risk=risk,risk_without=risk_without, explanation=explanation,explanation_text=explanation_text,report_file=report_file)
      
@app.route('/breastcancer')
def breastcancer():
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('breast_cancer.html',name=session['user_name'])

# --- 2. Create the Predict Route ---
@app.route('/predict_cancer', methods=['POST'])
def predict_cancer():
    # Gather form data
    age = float(request.form.get('age'))
    menopause = int(request.form.get('menopause'))
    size = float(request.form.get('tumor_size'))
    nodes = int(request.form.get('inv_nodes'))
    metastasis = int(request.form.get('metastasis'))
    history_input= int(request.form.get('history'))
    breast = request.form.get('breast')
    quadrant = request.form.get('quadrant')

    family_map = {
    0: 0.0,
    1: 0.7,
    2: 1.5
    }

    family_risk = family_map.get(history_input, 0)
    # Create empty row with all 12 feature columns
    input_df = pd.DataFrame(np.zeros((1, len(cancer_features))), columns=cancer_features)

    # Fill Numeric Columns
    input_df.at[0, 'Age'] = age
    input_df.at[0, 'Menopause'] = menopause
    input_df.at[0, 'Tumor Size (cm)'] = size
    input_df.at[0, 'Inv-Nodes'] = nodes
    input_df.at[0, 'Metastasis'] = metastasis
    input_df.at[0, 'History'] = family_risk

    # Fill Categorical Columns (One-Hot Encoding)
    if f"Breast_{breast}" in input_df.columns:
        input_df.at[0, f"Breast_{breast}"] = 1
    if f"Breast Quadrant_{quadrant}" in input_df.columns:
        input_df.at[0, f"Breast Quadrant_{quadrant}"] = 1

    # Predict
    prob = cancer_model.predict_proba(input_df.values)[0][1]

    input_df_no_fam = input_df.copy()
    input_df_no_fam.at[0, 'History'] = 0

    prob_without = cancer_model.predict_proba(input_df_no_fam.values)[0][1]
    adjusted_prob = prob + (family_risk * 0.25)
    adjusted_prob = min(adjusted_prob, 1)

    risk = round(adjusted_prob * 100, 2)
    risk_without = round(prob_without * 100, 2)

    # Explanation
    exp = cancer_explainer.explain_instance(input_df.values[0], cancer_model.predict_proba)
    explanation = exp.as_list()[:5]

    difference = risk - risk_without

    if difference > 20:
      hereditary_msg = f"Family history significantly increases breast cancer risk by {round(difference,2)}%."
    elif difference > 8:
      hereditary_msg = f"Family history moderately increases risk by {round(difference,2)}%."
    else:
      hereditary_msg = "Genetic influence is present but other tumor characteristics dominate."
    report_data = {
    "Age": age,
    "Tumor Size": size,
    "Menopause": menopause,
    "Involved Nodes": nodes,
    "Metastasis": metastasis,
    "Family History Impact": hereditary_msg,
    "Final Risk (%)": risk,
    "Without Family History (%)": risk_without
    }
    report_file = generate_cancer_report(report_data)

    return render_template("breast_cancer.html", risk=risk,risk_without=risk_without,hereditary_msg=hereditary_msg, explanation=explanation,report_file=report_file)


if __name__=="__main__":
    app.run(debug=True)
