"""
Script para documentar a arquitetura do projeto
"""
import subprocess
from pathlib import Path


def generate_tree(directory, prefix="", max_depth=3, current_depth=0, ignore_dirs={".git", "__pycache__", ".pytest_cache", "mlruns", ".streamlit"}):
    """Gera árvore de diretórios"""
    if current_depth >= max_depth:
        return ""
    
    entries = []
    try:
        items = sorted(Path(directory).iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return ""
    
    for item in items:
        if item.name.startswith('.') and item.name not in {'.github', '.gitignore'}:
            continue
        if item.name in ignore_dirs:
            continue
        
        is_last = item == items[-1]
        current_prefix = "└── " if is_last else "├── "
        entries.append(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir():
            next_prefix = prefix + ("    " if is_last else "│   ")
            entries.append(generate_tree(item, next_prefix, max_depth, current_depth + 1, ignore_dirs))
    
    return "\n".join(filter(None, entries))


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    tree = generate_tree(project_root)
    
    print("""
╔════════════════════════════════════════════════════════════════════════╗
║              📁 ARQUITETURA DO PROJETO                                 ║
╚════════════════════════════════════════════════════════════════════════╝

Projeto_machine_learning_mlflow/
""")
    print(tree)
