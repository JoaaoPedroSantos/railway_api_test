# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Dados fictícios
data = pd.DataFrame({
    "tamanho": [50, 60, 100, 120],
    "quartos": [1, 2, 3, 4],
    "idade": [10, 5, 20, 2],
    "preco": [100000, 150000, 300000, 400000]
})

X = data[["tamanho", "quartos", "idade"]]
y = data["preco"]

model = LinearRegression()
model.fit(X, y)

# Salvar o modelo
joblib.dump(model, "modelo_regressao.pkl")
