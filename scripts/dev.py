"""
Script para iniciar serviços de desenvolvimento
"""
import subprocess
import sys
import time
from pathlib import Path


def print_header(title):
    """Imprime header formatado"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_streamlit():
    """Inicia Streamlit"""
    print_header("🎨 INICIANDO STREAMLIT")
    print("Endpoint: http://localhost:8501\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", 
                   str(Path(__file__).parent.parent / "dashboard" / "app.py")])


def run_flask():
    """Inicia Flask"""
    print_header("🔌 INICIANDO FLASK API")
    print("Endpoint: http://localhost:5001\n")
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "api" / "app.py")])


def run_mlflow():
    """Inicia MLflow"""
    print_header("📊 INICIANDO MLFLOW UI")
    print("Endpoint: http://localhost:5000\n")
    subprocess.run(["mlflow", "ui"])


def run_train():
    """Executa treinamento"""
    print_header("🤖 TREINANDO MODELO")
    subprocess.run([sys.executable, str(Path(__file__).parent.parent / "main.py")])


def show_menu():
    """Mostra menu de opções"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        🚀 PROJETO DE ML - PREDIÇÃO DE INADIMPLÊNCIA        ║
    ╚════════════════════════════════════════════════════════════╝
    
    Opções disponíveis:
    
      1. streamlit      - Iniciar dashboard Streamlit
      2. flask          - Iniciar servidor Flask API
      3. mlflow         - Iniciar UI do MLflow
      4. train          - Treinar modelo (executar main.py)
      5. sair           - Sair do programa
    
    """)


if __name__ == "__main__":
    while True:
        show_menu()
        option = input("Escolha uma opção (1-5): ").strip()
        
        if option == "1":
            run_streamlit()
        elif option == "2":
            run_flask()
        elif option == "3":
            run_mlflow()
        elif option == "4":
            run_train()
        elif option == "5":
            print("\n👋 Até logo!")
            break
        else:
            print("❌ Opção inválida!")
