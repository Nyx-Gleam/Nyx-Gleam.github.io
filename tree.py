import os
import argparse
from pathlib import Path

def print_directory_tree(startpath, max_depth=None, exclude_dirs=None, show_size=False, show_hidden=False):
    """
    Muestra la estructura de directorios en forma de árbol
    """
    exclude = set(exclude_dirs) if exclude_dirs else set()
    prefix = "│   "
    space = "    "
    branch = "├── "
    last_branch = "└── "
    
    if show_size:
        print(f"{'Tamaño':<10}  Estructura")
        print("-----------------------")

    for root, dirs, files in os.walk(startpath):
        # Filtrar directorios excluidos y ocultos
        dirs[:] = [d for d in dirs if d not in exclude and (show_hidden or not d.startswith('.'))]
        files = [f for f in files if show_hidden or not f.startswith('.')]

        level = root.replace(startpath, '').count(os.sep)
        if max_depth and level > max_depth:
            continue

        # Calcular indentación
        indent = space * (level - 1) + branch if level > 0 else ""
        dir_name = os.path.basename(root)
        
        # Mostrar tamaño si está habilitado
        size_str = ""
        if show_size and level == 0:
            total_size = sum(f.stat().st_size for f in Path(root).glob('**/*') if f.is_file())
            size_str = f"{total_size / 1024:.1f} KB".ljust(10) + " "
        elif show_size:
            size_str = " " * 11

        # Imprimir directorio actual
        print(f"{size_str}{indent}{dir_name}/")

        # Manejar archivos
        file_indent = space * level + (last_branch if level == 0 else branch)
        for i, file in enumerate(files):
            final_branch = last_branch if i == len(files) - 1 else branch
            file_indent = space * (level) + final_branch
            
            # Calcular tamaño de archivo si está habilitado
            file_size = ""
            if show_size:
                try:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    file_size = f" ({size} bytes)" if size < 1024 else f" ({size/1024:.1f} KB)"
                except OSError:
                    file_size = " (error al obtener tamaño)"
            
            print(f"{' ' * len(size_str)}{file_indent}{file}{file_size}")

def main():
    parser = argparse.ArgumentParser(description="Mostrar estructura de directorios en forma de árbol")
    parser.add_argument("-d", "--depth", type=int, help="Profundidad máxima del árbol")
    parser.add_argument("-e", "--exclude", nargs='+', help="Directorios a excluir (ej. .git __pycache__)")
    parser.add_argument("-s", "--size", action="store_true", help="Mostrar tamaños de archivos/directorios")
    parser.add_argument("-H", "--hidden", action="store_true", help="Incluir archivos y carpetas ocultos")
    
    args = parser.parse_args()
    
    startpath = os.getcwd()
    print(f"\nEstructura de: {startpath}\n")
    print_directory_tree(
        startpath,
        max_depth=args.depth,
        exclude_dirs=args.exclude,
        show_size=args.size,
        show_hidden=args.hidden
    )

if __name__ == "__main__":
    main()