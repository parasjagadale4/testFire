import pickle
from flask import Flask, render_template, request
import numpy as np

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def home():
    result = None
    if request.method == "POST":
        try:
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))
            Ws = float(request.form.get('Ws', 0))
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            DC = float(request.form.get('DC', 0))
            ISI = float(request.form.get('ISI', 0))
            BUI = float(request.form.get('BUI', 0))
            Classes = float(request.form.get('Classes', 0))
            Region = float(request.form.get('Region', 0))

            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region]]
            )
            result = ridge_model.predict(new_data_scaled)[0]
        except Exception as e:
            result = f"Error: {e}"

    return render_template('home.html', results=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
