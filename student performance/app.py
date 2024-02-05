import os
import pandas as pd
from flask import Flask, render_template, request
import pickle

df = pd.read_csv("StudentsPerformance.csv")
app = Flask('project')

def features(gender, race, education, lunch, course):
    test_row = [gender, race, education, lunch, course]
    
    cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']
    
    df_cat = df[cat_features]
  
    df_cat.loc[(len(df_cat.index))] = test_row
    
    df_cat_dummy = pd.get_dummies(df_cat,drop_first=True)
    
    input_values = df_cat_dummy.iloc[-1].values
   
    model = pickle.load(open('model.sav', 'rb'))
    
    pred_values = model.predict([input_values])
    
    return pred_values


@app.route('/login.html', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin":
            return render_template('predictor_form.html')
        else:
            return render_template('login.html', msg="failed")

    else:
        return render_template('login.html')
@app.route('/') 
def show_form():
    #return render_template('predictor_form.html')
    return render_template('login.html')

@app.route('/visualisation', methods=["POST"])
def visualisation():
    return render_template('Report.html')

@app.route('/result', methods=["POST"])
def results():
    form = request.form
    if request.method == 'POST':
       
        gender = request.form['gender']
        race = request.form['race']
        education = request.form['education']
        lunch = request.form['lunch']
        course = request.form['course']
       
        pred_values = features(gender, race, education, lunch, course)
        pred_values = pred_values[0]
        math_score = round(pred_values[0],0)
        reading_score = round(pred_values[1],0)
        writing_score = round(pred_values[2],0)
        
        return render_template('result_form.html', gender=gender, race = race, education=education, 
                               lunch=lunch, course=course ,math_score=math_score, reading_score=reading_score,
                               writing_score=writing_score)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port,debug=True)
