from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelo al iniciar
modelo = joblib.load('modelo/modelo.pkl')

@app.route('/')
def index():
    return render_template('prediccion.html', prediccion=None)

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Extraer datos del formulario
        datos = [
            float(request.form['metros']),
            int(request.form['habitaciones']),
            int(request.form['banos']),
            int(request.form['antiguedad']),
            int(request.form['garaje']),
            int(request.form['jardin'])
        ]
        
        # Predicción
        prediccion = modelo.predict([datos])[0]
        prediccion = round(prediccion, 2)

        return render_template('prediccion.html', prediccion=prediccion)
    
    except Exception as e:
        return f"Error en la predicción: {e}"

if __name__ == '__main__':
    app.run(debug=True)
