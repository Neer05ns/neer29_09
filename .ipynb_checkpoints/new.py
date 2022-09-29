from flask import Flask, request,url_for, jsonify #flask class #flask module
import pandas as pd
import numpy as np
import pickle

logistic_model=pickle.load(open("new.pkl","rb"))

app =  Flask(__name__)


#column_list= pickle.load(open("column.pkl", "rb"))

@app.route('/')
def home_page():
    return jsonify("home.html")
    

@app.route('/prediction',methods=["POST"])
def prediction():
    
    Age = int(request.form["Age"])
    EstimatedSalary = int(request.form["EstimatedSalary"])

    data1 = pd.DataFrame({"Age":Age , "EstimatedSalary":EstimatedSalary})
    

    prediction = logistic_model.predict(data1)


    return jsonify("welcome.html",data=prediction)
if __name__ == "__main__":
    app.run()
