# 🚀 MODEL SERVER - Predição de Inadimplência

Servidor Flask que expõe o modelo de Machine Learning como uma API REST.

## ⚙️ Requisitos

- Python 3.8+
- Flask
- Flask-CORS (opcional, para CORS)
- Requests (para testes)

```bash
pip install flask flask-cors requests
```

## 🏃 Como Executar

### 1. Certifique-se de que o modelo foi treinado

```bash
python main.py
```

Isto gera `models/random_forest.pkl` e `models/scaler.pkl`

### 2. Inicie o servidor

```bash
python app.py
```

Você verá:
```
╔══════════════════════════════════════════════════════════════════════╗
║          🚀 MODEL SERVER - PREDIÇÃO DE INADIMPLÊNCIA                ║
╚══════════════════════════════════════════════════════════════════════╝

✓ Servidor iniciado com sucesso!

📍 Endpoints disponíveis:
  GET  http://localhost:5001/health
  GET  http://localhost:5001/model/info
  GET  http://localhost:5001/example
  POST http://localhost:5001/predict              (uma amostra)
  POST http://localhost:5001/predict/batch        (múltiplas amostras)
  POST http://localhost:5001/predict/csv          (arquivo CSV)
```

### 3. Teste o servidor

Em outro terminal:

```bash
python test_server.py
```

Ou com `curl`:

```bash
# Health check
curl http://localhost:5001/health

# Informações do modelo
curl http://localhost:5001/model/info

# Exemplos
curl http://localhost:5001/example
```

---

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```

**Resposta:**
```json
{
  "status": "healthy",
  "timestamp": "2026-04-11T14:30:00.000000",
  "model_loaded": true
}
```

---

### 2. Informações do Modelo
```
GET /model/info
```

**Resposta:**
```json
{
  "model": {
    "model_name": "random_forest",
    "model_path": "...",
    "scaler_loaded": true,
    "loaded_at": "2026-04-11T14:30:00.000000",
    "version": "1.0"
  },
  "features": {
    "count": 23,
    "names": ["LIMIT_BAL", "SEX", "EDUCATION", ...]
  },
  "classes": {
    "0": "Bom Pagador",
    "1": "Inadimplente (RISCO)"
  }
}
```

---

### 3. Predição Única
```
POST /predict
```

**Body:**
```json
{
  "features": [
    150000.0,  // LIMIT_BAL
    2,         // SEX
    2,         // EDUCATION
    1,         // MARRIAGE
    35,        // AGE
    -1, -1, -1, -1, -1, -1,  // PAY_0 a PAY_6
    50000, 45000, 40000, 35000, 30000, 25000,  // BILL_AMT1 a BILL_AMT6
    5000, 4500, 4000, 3500, 3000, 2500         // PAY_AMT1 a PAY_AMT6
  ]
}
```

**Resposta:**
```json
{
  "prediction": 0,
  "probability": {
    "class_0_good_payer": 0.85,
    "class_1_default": 0.15
  },
  "confidence": 0.85,
  "risk_level": "Baixo - Bom Pagador",
  "timestamp": "2026-04-11T14:30:00.000000"
}
```

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [150000.0, 2, 2, 1, 35, -1, -1, -1, -1, -1, -1, 50000, 45000, 40000, 35000, 30000, 25000, 5000, 4500, 4000, 3500, 3000, 2500]
  }'
```

---

### 4. Predição em Batch
```
POST /predict/batch
```

**Body:**
```json
{
  "samples": [
    [150000.0, 2, 2, 1, 35, ...],  // Amostra 1
    [200000.0, 1, 1, 1, 45, ...],  // Amostra 2
    [100000.0, 2, 3, 2, 28, ...]   // Amostra 3
  ]
}
```

**Resposta:**
```json
{
  "count": 3,
  "predictions": [0, 1, 0],
  "probabilities": [
    [0.85, 0.15],
    [0.35, 0.65],
    [0.78, 0.22]
  ],
  "confidences": [0.85, 0.65, 0.78],
  "summary": {
    "class_0_bom_pagador": 2,
    "class_1_inadimplente": 1,
    "pct_inadimplencia": 33.33,
    "avg_confidence": 0.7633
  },
  "timestamp": "2026-04-11T14:30:00.000000"
}
```

---

### 5. Predição com CSV
```
POST /predict/csv
```

**File:** `samples.csv` com 23 colunas (uma amostra por linha)

**Resposta:**
```json
{
  "count": 100,
  "summary": {
    "class_0_bom_pagador": 78,
    "class_1_inadimplente": 22,
    "pct_inadimplencia": 22.0
  },
  "output_file": "results/batch_predictions_20260411_143000.csv",
  "timestamp": "2026-04-11T14:30:00.000000"
}
```

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5001/predict/csv \
  -F "file=@samples.csv"
```

---

### 6. Exemplos
```
GET /example
```

Retorna exemplos de requisições para todos os endpoints.

---

## 💡 Exemplos de Uso

### Python + Requests

```python
import requests
import json

BASE_URL = "http://localhost:5001"

# Predição única
response = requests.post(
    f"{BASE_URL}/predict",
    json={"features": [150000.0, 2, 2, 1, 35, ...]}
)

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
```

### cURL

```bash
# Predição
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [150000.0, 2, 2, 1, 35, ...]}'

# Batch com arquivo
curl -X POST http://localhost:5001/predict/csv \
  -F "file=@data.csv"
```

### JavaScript/Node.js

```javascript
const BASE_URL = 'http://localhost:5001';

async function predict(features) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
  });
  
  return response.json();
}

// Uso
const result = await predict([150000.0, 2, 2, 1, 35, ...]);
console.log(result.risk_level);
```

---

## 🔒 Segurança em Produção

Para ambiente de produção, considere:

### 1. Usar WSGI Server (Gunicorn)

```bash
pip install gunicorn

# Iniciar com múltiplos workers
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### 2. Usar Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.seu-dominio.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Adicionar Autenticação (Token)

```python
from functools import wraps

def require_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('X-API-Key')
        if token != 'seu_token_secreto':
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/predict', methods=['POST'])
@require_token
def predict():
    ...
```

### 4. Rate Limiting

```bash
pip install flask-limiter

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per hour")
def predict():
    ...
```

### 5. Enable HTTPS

Use certificados SSL/TLS com nginx ou Let's Encrypt

---

## 📊 Monitoramento

### Logs

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app.logger.info("Predição realizada")
```

### Métricas

Integrar com Prometheus:

```python
from prometheus_client import Counter, Histogram
import time

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_time = Histogram('prediction_time_seconds', 'Prediction time')

@app.route('/predict', methods=['POST'])
def predict():
    with prediction_time.time():
        # ... lógica
        prediction_counter.inc()
```

---

## 🧪 Testes de Carga

Teste a capacidade do servidor com `Apache Bench`:

```bash
# 1000 requisições, concorrência de 10
ab -n 1000 -c 10 http://localhost:5001/health

# Teste POST
ab -n 1000 -c 10 -p payload.json -T application/json \
   http://localhost:5001/predict
```

---

## 🚨 Troubleshooting

### Erro: "Modelo não encontrado"
Certifique-se de executar `python main.py` primeiro para treinar o modelo.

### Erro: "Conexão recusada"
O servidor não está rodando. Inicie com `python app.py`

### Erro: "Port 5001 já está em uso"
Mude a porta em `app.py`:
```python
app.run(host='0.0.0.0', port=5002, debug=False)
```

---

## 📚 Referências

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask-CORS](https://flask-cors.readthedocs.io/)
- [Gunicorn](https://gunicorn.org/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/deployment/)

---

**Última atualização:** Abril 2026
