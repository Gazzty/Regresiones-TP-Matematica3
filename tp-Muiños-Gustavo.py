# En este TP voy a relacionar los datos del valor del Yen Japones contra la cantidad de turistas por mes en el ultimo año
# Estos valores se relacionan ya que se ve un aumento de turismo cuando el valor del yen cae en comparacion al valor del dolar EEUU
# Se van a ver incoherencias (datos que se oponen a lo que quiero demostrar) en 2020 hasta 2022, ya que en esa epoca estaba la pandemia
# NOTA: En los csv encontrados faltan datos entre 2014 y 2019, por lo cual se usaron predicciones para llenar esos datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Cargar los datasets
tourism_data = pd.read_csv('data/tourism-data.csv')
yen_data = pd.read_csv('data/yen-rate.csv')

# Preprocesamiento de datos
tourism_data['Year'] = tourism_data['Year'].astype(int)
yen_data['Record Date'] = pd.to_datetime(yen_data['Record Date'])
yen_data['Year'] = yen_data['Record Date'].dt.year

# Verificar años únicos en cada dataset
print("Años únicos en turismo:", tourism_data['Year'].unique())
print("Años únicos en yen:", yen_data['Year'].unique())

# Filtrar los años comunes
common_years = set(tourism_data['Year']).intersection(set(yen_data['Year']))
print("Años comunes:", common_years)  # Años comunes

# Filtrar datos según los años comunes
tourism_filtered = tourism_data[tourism_data['Year'].isin(common_years)]
yen_grouped = yen_data.groupby('Year')['Exchange Rate'].mean().reset_index()
yen_filtered = yen_grouped[yen_grouped['Year'].isin(common_years)]

# Imprimir los datasets filtrados
print("Datos de turismo filtrados:\n", tourism_filtered)
print("Datos de yen filtrados:\n", yen_filtered)

# Asegurarse de que los datos filtrados tengan las mismas longitudes
if len(tourism_filtered) != len(yen_filtered):
    raise ValueError("Las longitudes de los datos filtrados no coinciden")

# Imputar valores faltantes en los datos de turismo
imputer = SimpleImputer(strategy='mean')
y_tourism = tourism_filtered['Grand Total'].str.replace('.', '').str.replace(',', '').astype(float).to_numpy()
y_tourism_imputed = imputer.fit_transform(y_tourism.reshape(-1, 1)).ravel()

# Imputar valores faltantes en el valor del yen
y_yen = yen_filtered['Exchange Rate'].astype(float).to_numpy()
y_yen_imputed = imputer.fit_transform(y_yen.reshape(-1, 1)).ravel()

# Preparar datos para regresión lineal simple y múltiple
X = yen_filtered['Year'].values.reshape(-1, 1)  # Para la regresión simple
X_multi = np.column_stack((yen_filtered['Year'].values, y_yen_imputed))  # Para la regresión múltiple

# Regresión lineal simple para el yen
model_yen_simple = LinearRegression()
model_yen_simple.fit(X, y_yen_imputed)
yen_pred_simple = model_yen_simple.predict(X)

# Regresión lineal múltiple para el turismo
model_tourism_multi = LinearRegression()
model_tourism_multi.fit(X_multi, y_tourism_imputed)
tourism_pred_multi = model_tourism_multi.predict(X_multi)

# Graficar los resultados
plt.figure(figsize=(12, 6))

# Gráfico de turismo
plt.subplot(1, 2, 1)
plt.plot(tourism_filtered['Year'], y_tourism_imputed, label='Turistas (imputados)', marker='o')
plt.plot(tourism_filtered['Year'], tourism_pred_multi, label='Regresión Lineal Múltiple', linestyle='--', color='blue')
plt.title('Turismo en Japón')
plt.xlabel('Año')
plt.ylabel('Número de Turistas')
plt.legend()

# Gráfico del yen
plt.subplot(1, 2, 2)
plt.plot(yen_filtered['Year'], y_yen_imputed, label='Valor del Yen (imputado)', marker='o', color='orange')
plt.plot(yen_filtered['Year'], yen_pred_simple, label='Regresión Lineal Simple', linestyle='--', color='red')
plt.title('Valor del Yen Japonés')
plt.xlabel('Año')
plt.ylabel('Tipo de Cambio (Yen/USD)')
plt.legend()

plt.tight_layout()
plt.show()
