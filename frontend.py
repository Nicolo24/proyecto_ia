# flask form with no database
from flask import Flask, render_template, request
import os
from sintomas import sintomas, categorias
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html',categorias=categorias)


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        submitted = {}
        for sintoma in sintomas:
            if request.form.get(sintoma):
                submitted[sintoma] = 1
            else:
                submitted[sintoma] = 0

        return submitted
    else:
        return 'Error'


if __name__ == '__main__':
    app.run(debug=True, port=5001)
