# En este TP voy a relacionar los datos del valor del Yen Japones contra la cantidad de turistas por mes en el ultimo año
# Estos valores se relacionan ya que se ve un aumento de turismo cuando el valor del yen cae en comparacion al valor del dolar EEUU
# Se van a ver incoherencias (datos que se oponen a lo que quiero demostrar) en 2020 hasta 2022, ya que en esa epoca estaba la pandemia
# NOTA: En los csv encontrados faltan datos entre 2014 y 2019, por lo cual se usaron predicciones para llenar esos datos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

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
plt.plot(combined_df['Year'], combined_df['Grand Total'], marker='o', label='Turistas (Grand Total)', color='b')
plt.plot(combined_df['Year'], combined_df['Exchange Rate'], marker='x', label='Tasa de Cambio (Yen)', color='r')
plt.xlabel('Año')
plt.ylabel('Cantidad / Tasa de Cambio')
plt.title('Comparación entre la cantidad de turistas y la tasa de cambio del Yen')
plt.legend()
plt.grid()

# Ajustar la legibilidad de los años
plt.xticks(combined_df['Year'], rotation=45, ha='right', fontsize=10)

# Regresión lineal simple (solo para 'Grand Total')
# Eliminar filas con valores nulos
X_simple = combined_df[['Year']].dropna()
y_simple = combined_df['Grand Total'].dropna()

# Asegurarse de que X y y tengan la misma longitud
X_simple = X_simple.loc[y_simple.index]  # Mantener el mismo índice
y_simple = y_simple.reset_index(drop=True)

# Entrenar el modelo de regresión simple
model_simple = LinearRegression()
model_simple.fit(X_simple, y_simple)
pred_simple = model_simple.predict(X_simple)

# Regresión lineal múltiple
train_df = combined_df[combined_df['Grand Total'].notna()]
X_multi = train_df[['Year', 'Exchange Rate']]
y_multi = train_df['Grand Total']
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)
pred_multi = model_multi.predict(combined_df[['Year', 'Exchange Rate']])

# Añadir las líneas de regresión al gráfico
plt.plot(combined_df['Year'], pred_simple, color='blue', linestyle='--', label='Regresión Lineal Simple')
plt.plot(combined_df['Year'], pred_multi, color='orange', linestyle='--', label='Regresión Lineal Múltiple')

plt.tight_layout()
plt.legend()
plt.show()

# Mostrar el DataFrame actualizado con las predicciones
combined_df['Predicted Grand Total'] = np.nan
combined_df.loc[combined_df['Grand Total'].isna(), 'Predicted Grand Total'] = pred_multi[combined_df['Grand Total'].isna()]

print(combined_df)
