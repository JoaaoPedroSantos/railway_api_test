# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time

# -------------------
# Configurar logging
# -------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------
# Iniciar app
# -------------------
app = FastAPI(title="API de Previsão de Preço de Imóveis")

# Carregar modelo treinado
model = joblib.load("modelo_regressao.pkl")
logger.info("Modelo carregado com sucesso.")

# -------------------
# Schema de entrada
# -------------------
class ImovelInput(BaseModel):
    tamanho: float
    quartos: int
    idade: int

# -------------------
# Rotas
# -------------------
@app.get("/")
def read_root():
    logger.info("Rota raiz acessada.")
    return {"message": "API de Previsão de Imóveis no ar!"}

@app.post("/predict")
def predict_price(imovel: ImovelInput, request: Request):
    logger.info(f"Requisição recebida de {request.client.host} com dados: {imovel.dict()}")
    start_time = time.time()

    try:
        features = np.array([[imovel.tamanho, imovel.quartos, imovel.idade]])
        preco_predito = model.predict(features)[0]
        elapsed = time.time() - start_time
        logger.info(f"Previsão concluída: {preco_predito:.2f} (tempo: {elapsed:.3f}s)")
        return {"preco_estimado": round(preco_predito, 2)}
    
    except Exception as e:
        logger.exception("Erro ao realizar a previsão.")
        return {"erro": str(e)}
