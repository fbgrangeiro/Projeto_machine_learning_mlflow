"""
API Flask para Modelagem Preditiva
Servidor de produção para predições de inadimplência
"""
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask
from api.routes import register_routes
from api.config import setup_config


def create_app(debug=False):
    """Factory para criar a aplicação Flask"""
    
    app = Flask(__name__)
    
    # Configuração
    setup_config(app, debug=debug)
    
    # Registrar rotas
    register_routes(app)
    
    return app


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║          🚀 MODEL SERVER - PREDIÇÃO DE INADIMPLÊNCIA                ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    app = create_app(debug=False)
    
    print("""
    ✓ Servidor iniciado com sucesso!
    
    📍 Endpoints disponíveis:
    
      Health Check:
        GET  http://localhost:5001/health
      
      Informações:
        GET  http://localhost:5001/model/info
        GET  http://localhost:5001/example
      
      Predições:
        POST http://localhost:5001/predict              (uma amostra)
        POST http://localhost:5001/predict/batch        (múltiplas amostras)
        POST http://localhost:5001/predict/csv          (arquivo CSV)
    
    💡 Para testar:
        curl http://localhost:5001/health
        curl http://localhost:5001/example
    
    ⏸️  Para parar: Ctrl+C
    """)
    
    app.run(host='0.0.0.0', port=5001, debug=False)
