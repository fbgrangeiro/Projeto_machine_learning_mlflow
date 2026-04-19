"""
Rotas da API Flask
Endpoints para predições
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import request, jsonify
from api.config import load_model_once
from config.config import RESULTS_DIR


def register_routes(app):
    """Registra todas as rotas da aplicação"""
    
    # ====================================================================
    # HEALTH CHECK
    # ====================================================================
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check do servidor"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": True
        }), 200
    
    
    # ====================================================================
    # INFORMAÇÕES DO MODELO
    # ====================================================================
    
    @app.route('/model/info', methods=['GET'])
    def model_info():
        """Retorna metadados do modelo"""
        try:
            _, _, metadata = load_model_once()
            return jsonify({
                "model": metadata,
                "features": {
                    "count": 23,
                    "names": ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                             'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                             'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                             'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                             'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                },
                "classes": {
                    "0": "Bom Pagador",
                    "1": "Inadimplente (RISCO)"
                }
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    
    # ====================================================================
    # PREDIÇÃO ÚNICA
    # ====================================================================
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Realiza predição para uma única amostra"""
        try:
            model, scaler, _ = load_model_once()
            data = request.get_json()
            
            if not data or 'features' not in data:
                return jsonify({"error": "JSON com campo 'features' é obrigatório"}), 400
            
            features = np.array(data['features']).reshape(1, -1)
            
            if features.shape[1] != 23:
                return jsonify({
                    "error": f"Esperado 23 features, recebido {features.shape[1]}"
                }), 400
            
            # Predição
            prediction, probabilities = model.predict(features)
            pred_class = int(prediction[0])
            confidence = float(np.max(probabilities[0]))
            
            # Interpretação do risco
            risk_level = "ALTO - Inadimplente" if pred_class == 1 else "Baixo - Bom Pagador"
            
            return jsonify({
                "prediction": pred_class,
                "probability": {
                    "class_0_good_payer": float(probabilities[0][0]),
                    "class_1_default": float(probabilities[0][1])
                },
                "confidence": confidence,
                "risk_level": risk_level,
                "timestamp": datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    
    # ====================================================================
    # PREDIÇÃO EM BATCH
    # ====================================================================
    
    @app.route('/predict/batch', methods=['POST'])
    def predict_batch():
        """Realiza predições em múltiplas amostras"""
        try:
            model, scaler, _ = load_model_once()
            data = request.get_json()
            
            if not data or 'samples' not in data:
                return jsonify({"error": "JSON com campo 'samples' é obrigatório"}), 400
            
            samples = np.array(data['samples'])
            
            if samples.shape[1] != 23:
                return jsonify({
                    "error": f"Esperado 23 features, recebido {samples.shape[1]}"
                }), 400
            
            # Predições
            predictions, probabilities = model.predict(samples)
            confidences = np.max(probabilities, axis=1)
            
            # Análise
            class_0_count = int((predictions == 0).sum())
            class_1_count = int((predictions == 1).sum())
            avg_confidence = float(confidences.mean())
            
            return jsonify({
                "count": len(predictions),
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "confidences": confidences.tolist(),
                "summary": {
                    "class_0_bom_pagador": class_0_count,
                    "class_1_inadimplente": class_1_count,
                    "pct_inadimplencia": float(class_1_count / len(predictions) * 100),
                    "avg_confidence": avg_confidence
                },
                "timestamp": datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    
    # ====================================================================
    # UPLOAD E PREDIÇÃO (CSV)
    # ====================================================================
    
    @app.route('/predict/csv', methods=['POST'])
    def predict_csv():
        """Realiza predições a partir de arquivo CSV"""
        try:
            model, scaler, _ = load_model_once()
            
            if 'file' not in request.files:
                return jsonify({"error": "Arquivo CSV é obrigatório"}), 400
            
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({"error": "Arquivo não selecionado"}), 400
            
            # Ler CSV
            df = pd.read_csv(file)
            
            if df.shape[1] != 23:
                return jsonify({
                    "error": f"CSV deve ter 23 colunas, tem {df.shape[1]}"
                }), 400
            
            # Predições
            features = df.values
            predictions, probabilities = model.predict(features)
            confidences = np.max(probabilities, axis=1)
            
            # Adicionar resultados ao DataFrame
            df['prediction'] = predictions
            df['prob_class_0'] = probabilities[:, 0]
            df['prob_class_1'] = probabilities[:, 1]
            df['confidence'] = confidences
            df['risk_level'] = df['prediction'].apply(
                lambda x: "ALTO - Inadimplente" if x == 1 else "Baixo - Bom Pagador"
            )
            
            # Salvar resultado
            output_path = RESULTS_DIR / f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_path, index=False)
            
            return jsonify({
                "count": len(df),
                "summary": {
                    "class_0_bom_pagador": int((predictions == 0).sum()),
                    "class_1_inadimplente": int((predictions == 1).sum()),
                    "pct_inadimplencia": float((predictions == 1).sum() / len(predictions) * 100)
                },
                "output_file": str(output_path),
                "timestamp": datetime.now().isoformat()
            }), 200
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    
    # ====================================================================
    # EXEMPLO DE TESTE
    # ====================================================================
    
    @app.route('/example', methods=['GET'])
    def example():
        """Retorna exemplo de requisição"""
        return jsonify({
            "single_prediction_example": {
                "method": "POST",
                "endpoint": "/predict",
                "body": {
                    "features": [
                        150000.0, 2, 2, 1, 35,
                        -1, -1, -1, -1, -1, -1,
                        50000, 45000, 40000, 35000, 30000, 25000,
                        5000, 4500, 4000, 3500, 3000, 2500
                    ]
                }
            },
            "batch_prediction_example": {
                "method": "POST",
                "endpoint": "/predict/batch",
                "body": {
                    "samples": [
                        [150000.0, 2, 2, 1, 35, -1, -1, -1, -1, -1, -1, 50000, 45000, 40000, 35000, 30000, 25000, 5000, 4500, 4000, 3500, 3000, 2500],
                        [200000.0, 1, 1, 1, 45, 0, 0, 0, 0, 0, 0, 80000, 75000, 70000, 65000, 60000, 55000, 8000, 7500, 7000, 6500, 6000, 5500]
                    ]
                }
            },
            "endpoints": {
                "GET /health": "Health check do servidor",
                "GET /model/info": "Informações do modelo",
                "GET /example": "Exemplos de requisições",
                "POST /predict": "Predição para uma amostra",
                "POST /predict/batch": "Predições para múltiplas amostras",
                "POST /predict/csv": "Predições a partir de arquivo CSV"
            }
        }), 200
    
    
    # ====================================================================
    # ERRO 404
    # ====================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "Endpoint não encontrado",
            "hint": "Visite /example para ver os endpoints disponíveis"
        }), 404
