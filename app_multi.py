#flask hello world
from flask import Flask,render_template, request, jsonify
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sintomas import categorias, sintomas, traduccion

import openai
openai.api_key = "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"


# Cargar los datos de entrenamiento desde un archivo CSV
print('Read train data...')

x = pd.read_csv("./dataset/Training.csv").drop(columns=["prognosis"])
y= pd.read_csv("./dataset/TrainingY.csv")
print('Read test data...')

xt = pd.read_csv("./dataset/Testing.csv").drop(columns=["prognosis"])

# Crear el clasificador base y el clasificador multi-clase
base_clf = DecisionTreeClassifier()
multi_clf = OneVsRestClassifier(base_clf)
print('Training model...')

# Entrenar el clasificador multi-clase con los datos de entrenamiento
multi_clf.fit(x, y)

print('Model trained!')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',categorias=categorias, traduccion=traduccion)

@app.route("/getAccuracy")
def getAccuracy():
    acc = tree.score(xt, yt)
    print("Acurray on test set: {:.2f}%".format(acc*100))

#predict get request
@app.route("/predict", methods=['POST'])
def predict():
    #return data received in get request
    if request.method == 'POST':
        submitted = {}
        for sintoma in sintomas:
            if request.form.get(sintoma):
                submitted[sintoma] = [1]
            else:
                submitted[sintoma] = [0]

        submitteddf=pd.DataFrame.from_dict(submitted)
        y_pred = multi_clf.predict(submitteddf)
        y_proba = multi_clf.predict_proba(submitteddf)

        #save y_pred and y_proba to csv with the same columns as y
        y_pred_df = pd.DataFrame(y_pred, columns=y.columns)
        y_proba_df = pd.DataFrame(y_proba, columns=y.columns)

        pred_dict_dirty=y_pred_df.to_dict(orient='records')[0]
        proba_dict_dirty=y_proba_df.to_dict(orient='records')[0]

        pred_dict=[]
        proba_dict=[]
        possible_diseases=[]

        for disease in pred_dict_dirty.keys():
            if pred_dict_dirty[disease]==1:
                pred_dict.append(disease)
        
        for disease in proba_dict_dirty.keys():
            if proba_dict_dirty[disease]==1:
                proba_dict.append(disease)

        for disease in proba_dict:

            prompt = f"Dame una corta descripcion sobre la siguiente enfermedad '{disease}' en español"
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens= 1024
            )
            possible_diseases.append([disease,response['choices'][0]['text']])

        return render_template('diseases.html', traduccion=traduccion,sintomas=request.form.to_dict(), possible_diseases=possible_diseases)

    else:
        return jsonify({'result': 'No data received'})

#flask get multi
@app.route("/from_csv", methods=['GET'])
def from_csv():
    #return data received in get request
    if request.method == 'GET':
        return render_template('load_csv.html',categorias=categorias, traduccion=traduccion)

    else:
        return jsonify({'result': 'No data received'})

#route show results from csv
@app.route("/show_results", methods=['POST'])
def show_results():
    #print file received in request
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        dft = pd.read_csv(file)
        dft.describe()
        dft.shape

        dft.columns

        dft['prognosis'].value_counts()

        xt = dft.drop('prognosis', axis = 1)
        yt = dft['prognosis']

        y_pred = multi_clf.predict(xt)

        #ndarray to dataframe
        y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

        #put column name where value is 1
        y_pred_series = y_pred_df.idxmax(axis=1)

        #convert series to dataframe
        y_pred_df = pd.DataFrame(y_pred_series, columns=['prognosis'])

        #put yt in next column inside y_pred_df
        y_pred_df['prognosis_real'] = yt


        return y_pred_df.to_html()

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    