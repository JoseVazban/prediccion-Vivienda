import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/incremento_mensual.csv')

df['periodo_mensual'] = pd.to_datetime(df['periodo_mensual'], format='%d/%m/%Y')
ref = pd.to_datetime('01/01/2012')  # Establecer la fecha de referencia
df['dias_transcurridos'] = (df['periodo_mensual'] - ref).dt.days.astype(float)

df_test = df[(df['periodo_mensual'] >= "01/01/2012") & (df['periodo_mensual'] < "01/01/2024")]

dt_features = df_test.drop(['precio_mes', 'incremento_mensual', 'periodo_mensual'], axis=1)
dt_target = df_test['precio_mes']

X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=40)
print(X_train.shape)
print(y_train.shape)
print('='*36)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
_preds_df = pd.DataFrame(dict(observed=y_test, predicted=y_pred))
print('Entrenamiento del Modelo')
print(_preds_df)
print('='*36)

print('Score: {}'.format(model.score(X_test, y_test)))
print('='*36)

'''plt.scatter(X_test, y_test)
plt.plot(X_test, model.predict(X_test), color='Red')
plt.ylabel("Precio promedio")
plt.xlabel("Días transcurridos")
plt.show()
'''
#Predicción

df_2024 = df[(df['periodo_mensual'] >= "01/01/2024") & (df['periodo_mensual'] < "01/01/2025")]

X_new = df_2024.drop(['precio_mes', 'incremento_mensual', 'periodo_mensual'], axis=1)
y_pred_2024 = model.predict(X_new)
predic_2024 = pd.DataFrame({'Predicciones 2024': y_pred_2024})
print(predic_2024)
print('='*36)

#Nuevo Database

new_data = pd.read_csv('data/housing.csv')
last_reference = df['precio_mes'][143]
increase_one_month = (((y_pred_2024[0] * 100) / last_reference) - 100)
increase_six_months = (((y_pred_2024[5] * 100) / last_reference) - 100)
increase_one_year = (((y_pred_2024[11] * 100) / last_reference) - 100)
print(f'En 1 mes, el incrmento será del {increase_one_month:.2f}%')
print(f'En 6 meses, el incrmento será del {increase_six_months:.2f}%')
print(f'En 1 año, el incrmento será del {increase_one_year:.2f}%')
print('='*36)

new_data['01/01/2024'] = ((increase_one_month * new_data['median_house_value'])/100) + new_data['median_house_value']
new_data['01/06/2024'] = ((increase_six_months * new_data['median_house_value'])/100) + new_data['median_house_value']
new_data['01/12/2024'] = ((increase_one_year * new_data['median_house_value'])/100) + new_data['median_house_value']
new_data.to_csv('new_prices.csv')
print(new_data)
print('='*36)


