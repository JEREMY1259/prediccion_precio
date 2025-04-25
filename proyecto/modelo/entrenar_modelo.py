import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Cargar los datos
df = pd.read_csv('modelo/datos.csv')

# Selección de características
X = df[['metros', 'habitaciones', 'banos', 'antiguedad', 'garaje', 'jardin']]  # Agrega más si necesitas
y = df['precio']

# Entrenamiento del modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(modelo, 'modelo/modelo.pkl')
print("Modelo entrenado y guardado.")
