from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
#Machine Learning packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/calculate', methods = ["POST"])
def calculate():
    df = pd.read_csv("csv/COVID.csv")
    df_w = df.Age
    df_x = df.Gender
    df_y = df.status
    df_z = df.confirmation


@app.route('/predict',methods = ["POST"])
def predict():
    df = pd.read_csv("csv/COVID.csv")
    X = df.drop('Status', axis=1)
    y = df['Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if request.method == "POST":
        genderquery = request.form['genderquery']
        agequery = request.form['agequery']
        data = [[genderquery,agequery]]
        data = sc.transform(data)
        rfc = RandomForestClassifier(n_estimators = 200)
        rfc.fit(X_train, y_train)
        my_prediction = rfc.predict(data)
        my_prediction = my_prediction.tolist()
        my_prediction = ''.join(my_prediction)
        print(my_prediction)
        return render_template('results.html',prediction = my_prediction, gender = genderquery.upper())








if __name__ == "__main__":
    app.run(debug=True)
