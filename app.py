import pickle
import os
from flask import request, Flask, render_template, redirect, url_for, send_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ======================================Create app=================================================
app = Flask(__name__)

# ======================================loading models and datasets================================
df = pd.read_csv('Dataset/HR_comma_sep.csv.crdownload')
model = pickle.load(open('Models/model.pkl','rb'))
scaler = pickle.load(open('Models/scaler.pkl','rb'))

# ======================================dashboard functions========================================

# ======================================Dashboard==================================================
# routes===================================================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def home():
    return render_template("index.html")

@app.route('/recruitment')
def job():
    return render_template('recruitment.html')

@app.route('/HR_Dashboard')
def HR_Dashboard():
    return render_template('HR_Dashboard.html')
#=====================prediction function====================================================
def prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    data = {
        'sl_no': [sl_no],
        'gender': [gender],
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'workex': [workex],
        'etest_p': [etest_p],
        'specialisation': [specialisation],
        'mba_p': [mba_p]
    }
    data = pd.DataFrame(data)
    data['gender'] = data['gender'].map({'Male':1,"Female":0})
    data['workex'] = data['workex'].map({"Yes":1,"No":0})
    data['specialisation'] = data['specialisation'].map({"Mkt&HR":1,"Mkt&Fin":0})
    scaled_df = scaler.transform(data)
    result = model.predict(scaled_df).reshape(1, -1)
    return result[0]

#prediction===============================================================
@app.route("/recruitment", methods=['POST','GET'])
def recruitment():
    if request.method == 'POST':
        sl_no = request.form['sl_no']
        gender = request.form['gender']
        ssc_p = request.form['ssc_p']
        hsc_p = request.form['hsc_p']
        degree_p = request.form['degree_p']
        workex = request.form['workex']
        etest_p = request.form['etest_p']
        specialisation = request.form['specialisation']
        mba_p = request.form['mba_p']

        result = prediction(sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)

        if result == 1:
            pred = "Placed"
            rec = "This candidate is highly recommended for your organization."
            return render_template('recruitment.html', result=pred, rec=rec)

        else:
            pred = "Not Placed"
            rec = "Not placed this time. Keep in mind for future roles."
            return render_template('recruitment.html', result=pred, rec=rec)

    return redirect(url_for('index'))

# ========================python main===================================================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
