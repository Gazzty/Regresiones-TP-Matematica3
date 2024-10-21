# En este TP voy a relacionar los datos del valor del Yen Japones contra la cantidad de turistas por mes en el ultimo año
# Estos valores se relacionan ya que se ve un aumento de turismo cuando el valor del yen cae en comparacion al valor del dolar EEUU
# Se van a ver incoherencias (datos que se oponen a lo que quiero demostrar) en 2020 hasta 2022, ya que en esa epoca estaba la pandemia
# NOTA: En los csv encontrados faltan datos entre 2014 y 2019, por lo cual se usaron predicciones para llenar esos datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar los datos
yen_rate_df = pd.read_csv('data/yen-rate.csv')
tourism_df = pd.read_csv('data/tourism-data.csv')

# Limpiar y preparar los datos
tourism_df['Year'] = tourism_df['Year'].astype(int)  # Asegurarse que el año es int
tourism_df['Grand Total'] = tourism_df['Grand Total'].str.replace('.', '').str.replace(',', '.').astype(float)  # Convertir a float

# Filtrar datos para el gráfico
yen_rate_df['Record Date'] = pd.to_datetime(yen_rate_df['Record Date'])
yen_rate_df['Year'] = yen_rate_df['Record Date'].dt.year

# Agrupar la tasa de cambio por año
yen_rate_yearly = yen_rate_df.groupby('Year')['Exchange Rate'].mean().reset_index()

# Combinar los DataFrames
combined_df = pd.merge(tourism_df, yen_rate_yearly, on='Year', how='outer')

# Visualizar los datos
plt.figure(figsize=(12, 6))
plt.plot(combined_df['Year'], combined_df['Grand Total'], marker='o', label='Turistas (Grand Total)')
plt.plot(combined_df['Year'], combined_df['Exchange Rate'], marker='x', label='Tasa de Cambio (Yen)')
plt.xlabel('Año')
plt.ylabel('Cantidad / Tasa de Cambio')
plt.title('Comparación entre la cantidad de turistas y la tasa de cambio del Yen')
plt.legend()
plt.grid()

# Mejorar la legibilidad de los años
plt.xticks(combined_df['Year'], rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()

# Predecir valores faltantes usando regresión lineal
# Solo para el "Grand Total"
train_df = combined_df[combined_df['Grand Total'].notna()]
X = train_df[['Year', 'Exchange Rate']]
y = train_df['Grand Total']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir sobre los datos faltantes
missing_data = combined_df[combined_df['Grand Total'].isna()]
if not missing_data.empty:
    X_missing = missing_data[['Year', 'Exchange Rate']]
    predictions = model.predict(X_missing)
    combined_df.loc[combined_df['Grand Total'].isna(), 'Grand Total'] = predictions

# Mostrar el DataFrame actualizado con las predicciones
print(combined_df)
