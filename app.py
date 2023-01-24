#flask hello world
from flask import Flask,render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sintomas import categorias, sintomas, traduccion

print('Read train data...')

df = pd.read_csv("./dataset/Training.csv")
df.describe()
df.shape

df.drop('Unnamed: 133', axis=1, inplace=True)
df.columns

df['prognosis'].value_counts()

x = df.drop('prognosis', axis = 1)
y = df['prognosis']

print('Read test data...')
dft = pd.read_csv("./dataset/Testing.csv")
dft.describe()
dft.shape

dft.columns

dft['prognosis'].value_counts()

xt = dft.drop('prognosis', axis = 1)
yt = dft['prognosis']

tree = DecisionTreeClassifier()
print('Training model...')
tree.fit(x, y)
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


        df=pd.DataFrame.from_dict(submitted)

        predicted=tree.predict(df)
        
        return jsonify(predicted[0])
    else:
        return jsonify({'result': 'No data received'})


if __name__ == "__main__":
    app.run(debug=True, port=5000)

    