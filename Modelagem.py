from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple


@dataclass
class QCSPInstance:
    processing_times: List[int]
    task_bays: List[int]
    bays: int
    travel_time: int
    safety_margin: int
    precedence: Set[Tuple[int, int]]
    nonsimultaneous: Set[Tuple[int, int]]
    cranes_ready: List[int]
    cranes_init_pos: List[int]


def load_instance(path: Path) -> QCSPInstance:
    with path.open() as f:
        header = f.readline().strip().replace(" ", "").split(",")
        n, b, phi_sz, psi_sz, q, t, s = map(int, header)

        processing = list(map(int, f.readline().split()))
        locations = list(map(int, f.readline().split()))
        ready_times = list(map(int, f.readline().split()))
        init_pos = list(map(int, f.readline().split()))

        if not f.readline().lower().startswith("phi"):
            raise ValueError("Linha 'Phi:' ausente")

        precedence = set()
        for _ in range(phi_sz):
            i, j = map(int, f.readline().replace(",", " ").split())
            precedence.add((i, j))

        if not f.readline().lower().startswith("psi"):
            raise ValueError("Linha 'Psi:' ausente")

        nonsim = set()
        for _ in range(psi_sz):
            i, j = map(int, f.readline().replace(",", " ").split())
            nonsim.add((i, j))

    if len(processing) != n or len(locations) != n:
        raise ValueError("Tamanho de tarefas inconsistente com n")
    if len(ready_times) != q or len(init_pos) != q:
        raise ValueError("Tamanho de guindastes inconsistente com q")

    return QCSPInstance(
        processing_times=processing,
        task_bays=locations,
        bays=b,
        travel_time=t,
        safety_margin=s,
        precedence=precedence,
        nonsimultaneous=nonsim,
        cranes_ready=ready_times,
        cranes_init_pos=init_pos,
    )


def evaluate_schedule(
    instance: QCSPInstance,
    start_times: List[float],
    task_crane: List[int],
):
    """Compute makespan and validate constraints for a schedule.

    Constraints checked:
    - Precedence: predecessors finish before successors start.
    - Non-simultaneous pairs (Psi): intervals do not overlap.
    - Travel/ready times per crane: respect move time from previous bay (or initial bay) and ready time.
    - Safety margin: overlapping tasks on different cranes must keep bay distance >= safety_margin.
    """

    n = len(instance.processing_times)
    if len(start_times) != n or len(task_crane) != n:
        raise ValueError("start_times e task_crane devem ter tamanho n")

    q = len(instance.cranes_ready)
    if any(c < 1 or c > q for c in task_crane):
        raise ValueError("Ids de guindaste devem estar entre 1 e q")

    finish_times = [start_times[i] + instance.processing_times[i] for i in range(n)]
    violations: List[str] = []

    # Precedence
    for i, j in instance.precedence:
        if finish_times[i - 1] > start_times[j - 1]:
            violations.append(f"precedence {i}->{j} violada: fim {finish_times[i-1]} > inicio {start_times[j-1]}")

    # Non-simultaneous
    for i, j in instance.nonsimultaneous:
        overlap = not (finish_times[i - 1] <= start_times[j - 1] or finish_times[j - 1] <= start_times[i - 1])
        if overlap:
            violations.append(f"nonsimultaneous {i},{j} se sobrepoe")

    # Travel and ready per crane
    tasks_by_crane: List[List[int]] = [[] for _ in range(q)]
    for idx, crane in enumerate(task_crane):
        tasks_by_crane[crane - 1].append(idx)
    for crane_idx, tasks in enumerate(tasks_by_crane):
        tasks.sort(key=lambda t: start_times[t])
        last_finish = instance.cranes_ready[crane_idx]
        last_pos = instance.cranes_init_pos[crane_idx]
        for t_idx in tasks:
            move_time = instance.travel_time * abs(instance.task_bays[t_idx] - last_pos)
            earliest = last_finish + move_time
            if start_times[t_idx] < earliest:
                violations.append(
                    f"guindaste {crane_idx+1} tarefa {t_idx+1} inicia {start_times[t_idx]} < pronto {earliest}"
                )
            last_finish = finish_times[t_idx]
            last_pos = instance.task_bays[t_idx]

    # Safety margin (pairwise overlaps across cranes)
    s = instance.safety_margin
    for i in range(n):
        for j in range(i + 1, n):
            if task_crane[i] == task_crane[j]:
                continue
            overlap = not (finish_times[i] <= start_times[j] or finish_times[j] <= start_times[i])
            if not overlap:
                continue
            if abs(instance.task_bays[i] - instance.task_bays[j]) < s:
                violations.append(
                    f"safety margin violada entre {i+1} e {j+1}: bays {instance.task_bays[i]} vs {instance.task_bays[j]}"
                )

    makespan = max(finish_times)
    return {
        "valid": len(violations) == 0,
        "makespan": makespan,
        "violations": violations,
    }


def order_matrix_to_assignment(order_matrix: List[List[int]], n_tasks: int) -> List[int]:
    """Converte matriz de ordem (q x m) em vetor de alocacao de guindaste por tarefa.

    Convenção: cada linha representa um guindaste; em cada linha, valores em ordem
    crescente de execução; use 0 para posições vazias. Retorna task_crane (ids 1..q).
    Lança erro se uma tarefa aparecer mais de uma vez ou se alguma faltar.
    """

    q = len(order_matrix)
    task_crane = [0] * n_tasks

    seen = set()
    for crane_idx, row in enumerate(order_matrix, start=1):
        for val in row:
            if val == 0:
                continue
            if val < 1 or val > n_tasks:
                raise ValueError(f"Tarefa fora de faixa na linha {crane_idx}: {val}")
            if val in seen:
                raise ValueError(f"Tarefa repetida: {val}")
            seen.add(val)
            task_crane[val - 1] = crane_idx

    if len(seen) != n_tasks:
        missing = [i for i in range(1, n_tasks + 1) if i not in seen]
        raise ValueError(f"Tarefas ausentes na matriz: {missing}")

    return task_crane