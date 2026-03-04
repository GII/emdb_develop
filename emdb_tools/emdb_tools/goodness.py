import pandas as pd
import ast
import matplotlib.pyplot as plt
import bz2
import argparse
import os
import numpy as np
import sys


# ---------------------------------------------------------------------
# Utilidades de entrada
# ---------------------------------------------------------------------

def open_file(name):
    """Abrir fichero de texto o bz2 sin depender de 'magic'."""
    if name.endswith(".bz2"):
        return bz2.open(name, mode="rt", encoding="utf-8")
    return open(name, encoding="utf-8")


def strtobool(val):
    """Convertir string a booleano."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value {val}")


# ---------------------------------------------------------------------
# Carga y extracción de recompensas
# ---------------------------------------------------------------------

def obtain_rewards_dict(file, selected_goals=None):
    """
    Leer el goodness_0.txt del bartender y devolver:
    - lista de iteraciones
    - diccionario {goal_name: lista de recompensas}
    - lista de cambios sensoriales (0/1)
    """
    f = open_file(file)
    dataset = pd.read_csv(f, delimiter="\t", header=0)

    # Parsear diccionario de recompensas por fila
    rewards_str = dataset["Goal reward list"].to_list()
    rewards = [ast.literal_eval(goal) for goal in rewards_str]

    rewards_dataset = pd.DataFrame(rewards)

    # Si no se especifican goals, usar todas las columnas del diccionario
    if selected_goals is None:
        selected_goals = list(rewards_dataset.columns)

    # Filtrar solo los goals relevantes
    rewards_dataset = rewards_dataset[selected_goals]

    iterations = dataset["Iteration"].to_list()
    changes = dataset["Sensorial changes"].to_list()
    rewards_per_goal = rewards_dataset.to_dict(orient="list")

    return iterations, rewards_per_goal, changes


# ---------------------------------------------------------------------
# Estadísticas por bloques de iteraciones
# ---------------------------------------------------------------------

def generate_grouped_statistics(goal_name, iterations, rewards, changes,
                                block_size, base_file_name):
    """
    Calcular estadísticas por bloques de 'block_size' iteraciones:
    - Avg_reward (%)
    - NoChange_Iterations
    - Rewarded_Iterations
    """
    data = []
    accumulated_reward = 0.0
    no_change_iterations = 0
    rewarded_iterations = 0

    if len(iterations) != len(rewards) or len(rewards) != len(changes):
        raise ValueError("Error. Dimensions do not match")

    for i in range(len(iterations)):
        iteration = int(iterations[i])
        reward = float(rewards[i]) if not pd.isna(rewards[i]) else 0.0
        change = int(changes[i])

        accumulated_reward += reward
        if not change:
            no_change_iterations += 1
        if reward > 0.01:
            rewarded_iterations += 1

        # Cuando se completa un bloque
        if (iteration % block_size == 0) and iteration > 0:
            data.append({
                "Block": iteration // block_size,
                "Iteration": iteration,
                "Avg_reward": accumulated_reward / block_size * 100.0,
                "NoChange_Iterations": no_change_iterations,
                "Rewarded_Iterations": rewarded_iterations,
            })

            accumulated_reward = 0.0
            no_change_iterations = 0
            rewarded_iterations = 0

    df = pd.DataFrame(data)
    file_name = f"{goal_name}_{base_file_name}_blocks{block_size}.csv"
    if os.path.exists(file_name):
        print(f"File already exists: {file_name}")
    else:
        df.to_csv(file_name, sep="\t", index=False)

    data_to_plot = df.to_dict(orient="list")
    return data_to_plot


def generate_accumulated_statistics(all_data, base_file_name, block_size):
    """
    Sumar la Avg_reward de todos los goals para cada bloque.
    """
    # all_data es una lista de dicts {goal_name: data_dict}
    # Tomamos los bloques de la primera entrada como referencia
    first_goal_data = next(iter(all_data[0].values()))
    blocks = first_goal_data["Block"]
    avg_rewards_per_goal = [
        goal_data[list(goal_data.keys())[0]]["Avg_reward"]
        for goal_data in all_data
    ]

    accumulated_avg_rewards = [sum(vals) for vals in zip(*avg_rewards_per_goal)]
    accumulated_statistics = {
        "Block": blocks,
        "Avg_reward": accumulated_avg_rewards,
    }

    df = pd.DataFrame(accumulated_statistics)
    file_name = f"accumulated_reward_{base_file_name}_blocks{block_size}.csv"
    if os.path.exists(file_name):
        print(f"File already exists: {file_name}")
    else:
        df.to_csv(file_name, sep="\t", index=False)

    return accumulated_statistics


# ---------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------

def plot_data(all_data, accumulated_data, block_size,
              figures, show_figures, base_file_name):
    """
    Generar figuras para el paper:
    - Una figura por goal relevante (Avg_reward vs bloque de iteraciones)
    - Figura con todos los goals
    - Figura de recompensa acumulada
    """

    # Eje x en bloques de iteraciones
    blocks = next(iter(all_data[0].values()))["Block"]

    # 3: figuras individuales + all goals + acumulada
    # 2: all goals + acumulada
    # 1: solo acumulada

    # Figuras individuales por goal
    if figures == 3:
        for data_dict in all_data:
            goal_name = list(data_dict.keys())[0]
            data = data_dict[goal_name]

            plt.figure(figsize=(10, 6))
            plt.title(f"{goal_name}: recompensa media por bloque", fontsize=16)
            plt.xlabel("Bloques de iteraciones", fontsize=14)
            plt.ylabel("Recompensa media del goal (%)", fontsize=14)

            plt.plot(blocks, data["Avg_reward"], marker="o", linewidth=2,
                     label=goal_name)

            plt.grid(linewidth=0.5, alpha=0.5)
            plt.legend(loc="best", fontsize=12)
            plt.tight_layout()

            img_name = f"{goal_name}_blocks{block_size}_{base_file_name}.svg"
            if os.path.exists(img_name):
                print(f"File already exists: {img_name}")
            else:
                plt.savefig(img_name)
            if show_figures:
                plt.show()
            plt.close()

    # Figura con todos los goals relevantes
    if figures in (2, 3):
        plt.figure(figsize=(10, 6))
        plt.title("Bartender: recompensas medias por bloque", fontsize=16)
        plt.xlabel("Bloques de iteraciones", fontsize=14)
        plt.ylabel("Recompensa media del goal (%)", fontsize=14)

        for data_dict in all_data:
            goal_name = list(data_dict.keys())[0]
            data = data_dict[goal_name]
            plt.plot(blocks, data["Avg_reward"], linewidth=2, label=goal_name)

        plt.grid(linewidth=0.5, alpha=0.5)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()

        img_name = f"all_goals_blocks{block_size}_{base_file_name}.svg"
        if os.path.exists(img_name):
            print(f"File already exists: {img_name}")
        else:
            plt.savefig(img_name)
        if show_figures:
            plt.show()
        plt.close()

    # Figura de recompensa acumulada (todos los goals)
    if figures in (1, 2, 3):
        plt.figure(figsize=(10, 6))
        plt.title("Bartender: recompensa acumulada por bloque", fontsize=16)
        plt.xlabel("Bloques de iteraciones", fontsize=14)
        plt.ylabel("Recompensa acumulada de goals (%)", fontsize=14)

        plt.plot(accumulated_data["Block"], accumulated_data["Avg_reward"],
                 linewidth=2, color="black", label="Suma de goals")

        plt.grid(linewidth=0.5, alpha=0.5)
        plt.legend(loc="best", fontsize=12)
        plt.tight_layout()

        img_name = f"accumulated_reward_blocks{block_size}_{base_file_name}.svg"
        if os.path.exists(img_name):
            print(f"File already exists: {img_name}")
        else:
            plt.savefig(img_name)
        if show_figures:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estadísticas de goals (bartender) por bloques de iteraciones "
            "a partir de goodness_0.txt"
        )
    )
    parser.add_argument(
        "-f", "--file", required=True,
        help="Fichero goodness_0.txt del experimento del bartender",
    )
    parser.add_argument(
        "-n", "--iterations", type=int, required=True,
        help="Tamaño del bloque de iteraciones usado para las estadísticas",
    )
    parser.add_argument(
        "-fg", "--figures", type=int, default=2,
        help=(
            "3: todas las figuras (por goal + todos + acumulada) / "
            "2: todos los goals + acumulada / "
            "1: solo acumulada / 0: sin figuras"
        ),
    )
    parser.add_argument(
        "-s", "--show_figures", default="false",
        help="true/false: mostrar las figuras tras generarlas",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    file_name = kwargs["file"]
    block_size = kwargs["iterations"]
    figures = kwargs["figures"]
    show_figures = strtobool(kwargs["show_figures"])

    base_file_name, _ = os.path.splitext(file_name)

    # Goals relevantes del bartender
    selected_goals = [
        "serve_the_drink_drive",
        "return_the_glass_drive",
        "novelty_goal",
    ]

    iterations, rewards_per_goal, changes = obtain_rewards_dict(file_name, selected_goals=selected_goals)

    all_data = []
    for goal in rewards_per_goal:
        data_to_plot = generate_grouped_statistics(
            goal,
            iterations,
            rewards_per_goal[goal],
            changes,
            block_size,
            base_file_name,
        )
        data_to_plot_dict = {goal: data_to_plot}
        all_data.append(data_to_plot_dict)

    accumulated_data = generate_accumulated_statistics(
        all_data,
        base_file_name,
        block_size,
    )

    if figures != 0:
        if figures in (1, 2, 3):
            plot_data(
                all_data,
                accumulated_data,
                block_size,
                figures,
                show_figures,
                base_file_name,
            )
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
