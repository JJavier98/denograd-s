import os

import torch


class Colors:
    """ANSI color helpers for lightweight CLI output."""

    RESET = "\033[0m"
    NEGRITA = "\033[1m"
    SUBRAYADO = "\033[4m"
    ROJO = "\033[91m"
    VERDE = "\033[92m"
    AMARILLO = "\033[93m"
    AZUL = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    CYAN_2 = "\033[36m"
    BLANCO = "\033[30m"
    NEGRO = "\033[37m"


def make_dir(dir_path):
    """Create a directory if needed."""
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return True


def extract_data_from_loader(loader):
    """Materialize a dataloader into full tensors."""
    features = []
    targets = []
    for batch_x, batch_y in loader:
        features.append(batch_x)
        targets.append(batch_y)
    return torch.cat(features), torch.cat(targets)


def print_results_table(results, title="Results", title_color=Colors.CYAN):
    """Pretty-print scalar results in a compact table."""
    print(f"\n{Colors.NEGRITA}{title_color}--- {title} ---{Colors.RESET}")
    print(f"{Colors.SUBRAYADO}{'Model':<20} | {'Value':<15}{Colors.RESET}")
    for model, value in results.items():
        value_str = f"{value:.6f}" if isinstance(value, (float, int)) else str(value)
        print(f"{model:<20} | {value_str:<15}")
    print("-" * 40)