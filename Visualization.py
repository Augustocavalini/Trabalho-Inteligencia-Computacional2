from typing import List, Optional
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Modelagem import QCSPInstance, compute_start_times, compute_finish_times


def plot_gantt_by_bay(
    instance: QCSPInstance,
    encoded1: List[int],
    encoded2: List[int],
    title: str = "Gantt",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plota Gantt com linhas por bay e setas de movimentação dos guindastes.
    
    Inclui indicadores de tempo de início e fim das tarefas na base e topo dos retângulos.
    """
    start_times = compute_start_times(instance, encoded1, encoded2)
    finish_times = compute_finish_times(instance, start_times)

    n = len(instance.processing_times)
    q = len(instance.cranes_ready)

    # mapa tarefa -> guindaste
    task_crane = [-1] * n
    for idx, task_id in enumerate(encoded1):
        task_crane[task_id - 1] = encoded2[idx]

    colors = plt.cm.get_cmap("tab20", q)

    fig, ax = plt.subplots(figsize=(14, 6))

    # índices válidos (evita NaN/inf e tempos negativos)
    valid_tasks = [
        i for i in range(n)
        if math.isfinite(start_times[i])
        and math.isfinite(finish_times[i])
        and finish_times[i] >= start_times[i]
    ]

    # retângulos das tarefas por bay
    for task_idx in valid_tasks:
        bay = instance.task_bays[task_idx]
        start = start_times[task_idx]
        finish = finish_times[task_idx]
        crane = task_crane[task_idx]
        color = colors((crane - 1) % q) if crane != -1 else (0.7, 0.7, 0.7, 1.0)

        rect = patches.Rectangle(
            (start, bay - 0.4),
            finish - start,
            0.8,
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )
        ax.add_patch(rect)
        
        # Rótulo da tarefa no centro
        ax.text(
            start + (finish - start) / 2,
            bay,
            f"T{task_idx + 1}",
            ha="center",
            va="center",
            fontsize=8,
            color="#2a2a2a",
            weight="bold",
            clip_on=True,
        )
        
        # Indicador de tempo no topo (início)
        ax.text(
            start,
            bay + 0.5,
            f"{int(start)}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="darkgreen",
            weight="bold",
        )
        
        # Indicador de tempo na base (fim)
        ax.text(
            finish,
            bay - 0.5,
            f"{int(finish)}",
            ha="left",
            va="top",
            fontsize=7,
            color="darkred",
            weight="bold",
        )

    # setas de movimentação por guindaste
    for crane_id in range(1, q + 1):
        tasks = [
            t for t in valid_tasks
            if task_crane[t] == crane_id
        ]
        tasks.sort(key=lambda t: start_times[t])

        if not tasks:
            continue

        # seta da posição inicial até a primeira tarefa
        first = tasks[0]
        ax.annotate(
            "",
            xy=(start_times[first], instance.task_bays[first]),
            xytext=(instance.cranes_ready[crane_id - 1], instance.cranes_init_pos[crane_id - 1]),
            arrowprops=dict(arrowstyle="->", color=colors(crane_id - 1), linewidth=1.2),
            annotation_clip=True,
        )

        for prev, nxt in zip(tasks[:-1], tasks[1:]):
            ax.annotate(
                "",
                xy=(start_times[nxt], instance.task_bays[nxt]),
                xytext=(finish_times[prev], instance.task_bays[prev]),
                arrowprops=dict(arrowstyle="->", color=colors(crane_id - 1), linewidth=1.2),
                annotation_clip=True,
            )

    ax.set_title(title, fontsize=14, weight="bold", pad=20)
    ax.set_xlabel("Tempo", fontsize=11)
    ax.set_ylabel("Bay", fontsize=11)
    ax.set_yticks(range(1, instance.bays + 1))
    ax.set_ylim(0.2, instance.bays + 0.8)
    ax.set_facecolor("#f0f0f0")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if valid_tasks:
        min_start = min(start_times[i] for i in valid_tasks)
        max_finish = max(finish_times[i] for i in valid_tasks)
        pad = max((max_finish - min_start) * 0.05, 1.0)
        ax.set_xlim(min_start - pad, max_finish + pad)
    else:
        ax.set_xlim(0, 1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
