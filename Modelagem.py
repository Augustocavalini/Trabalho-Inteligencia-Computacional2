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

        processing = list(map(int, f.readline().split(",")))
        locations = list(map(int, f.readline().split(",")))
        ready_times = list(map(int, f.readline().split(",")))
        init_pos = list(map(int, f.readline().split(",")))

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
    order_matrix: List[List[int]],
    start_times: List[float],
):
    """Compute makespan and validate constraints for a schedule.

    Args:
        instance: QCSPInstance com dados do problema
        order_matrix: matriz q x m, linha i = guindaste i, valores = ids de tarefas em ordem
        start_times: tempos de início (tamanho n)

    Constraints checked:
    - Precedence: predecessors finish before successors start.
    - Non-simultaneous pairs (Psi): intervals do not overlap.
    - Travel/ready times per crane: respect move time and ready time for each crane.
    - Safety margin: overlapping tasks on different cranes must keep bay distance >= safety_margin.
    """

    n = len(instance.processing_times)
    q = len(order_matrix)
    
    if len(start_times) != n:
        raise ValueError(f"start_times deve ter tamanho {n}")

    finish_times = [start_times[i] + instance.processing_times[i] for i in range(n)]
    violations: List[str] = []

    # Build task_crane from order_matrix for safety margin checks
    task_crane = [0] * n
    for crane_idx, row in enumerate(order_matrix, start=1):
        for task_id in row:
            if task_id != 0:
                task_crane[task_id - 1] = crane_idx

    # Precedence
    for i, j in instance.precedence:
        if finish_times[i - 1] > start_times[j - 1]:
            violations.append(f"precedence {i}->{j} violada: fim {finish_times[i-1]} > inicio {start_times[j-1]}")

    # Non-simultaneous
    for i, j in instance.nonsimultaneous:
        overlap = not (finish_times[i - 1] <= start_times[j - 1] or finish_times[j - 1] <= start_times[i - 1])
        if overlap:
            violations.append(f"nonsimultaneous {i},{j} se sobrepoe")

    # Travel and ready per crane (usa ordem da matriz diretamente)
    for crane_idx, row in enumerate(order_matrix):
        last_finish = float(instance.cranes_ready[crane_idx])
        last_pos = instance.cranes_init_pos[crane_idx]
        for task_id in row:
            if task_id == 0:
                continue
            t_idx = task_id - 1
            move_time = instance.travel_time * abs(instance.task_bays[t_idx] - last_pos)
            earliest = last_finish + move_time
            if start_times[t_idx] < earliest - 1e-9:  # pequena tolerância numérica
                violations.append(
                    f"guindaste {crane_idx+1} tarefa {task_id} inicia {start_times[t_idx]} < pronto {earliest}"
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

    makespan = max(finish_times) if finish_times else 0
    return {
        "valid": len(violations) == 0,
        "makespan": makespan,
        "violations": violations,
    }


def compute_start_times_from_order_matrix(
    instance: QCSPInstance,
    order_matrix: List[List[int]],
) -> List[float]:
    """Calcula tempos de início a partir da matriz de ordem e restrições de precedência.

    Processa cada guindaste em sequência. Para cada tarefa na ordem:
    - Calcula o tempo de pronto do guindaste (ready time + travel time desde última posição)
    - Garante que precedências são respeitadas (task_i finaliza antes de task_j iniciar)
    
    Valida que todas as tarefas aparecem exatamente uma vez na matriz.
    Retorna vetor start_times (tamanho n, indexado por id_tarefa - 1).
    """
    n = len(instance.processing_times)
    
    # Validar matriz: cada tarefa deve aparecer exatamente uma vez
    seen = set()
    for crane_idx, row in enumerate(order_matrix):
        for task_id in row:
            if task_id == 0:
                continue
            if task_id < 1 or task_id > n:
                raise ValueError(f"Tarefa fora de faixa na linha {crane_idx}: {task_id}")
            if task_id in seen:
                raise ValueError(f"Tarefa repetida na matriz: {task_id}")
            seen.add(task_id)
    
    if len(seen) != n:
        missing = [i for i in range(1, n + 1) if i not in seen]
        raise ValueError(f"Tarefas ausentes na matriz: {missing}")
    
    # Inicializar start_times com -1 para detectar tarefas não agendadas
    start_times: List[float] = [-1.0] * n
    
    # Processar cada guindaste em sequência
    for crane_idx, row in enumerate(order_matrix):
        last_finish = float(instance.cranes_ready[crane_idx])
        last_pos = instance.cranes_init_pos[crane_idx]
        
        for task_id in row:
            if task_id == 0:
                continue
            
            task_idx = task_id - 1
            move_time = instance.travel_time * abs(instance.task_bays[task_idx] - last_pos)
            earliest_from_crane = last_finish + move_time
            
            # Garantir que precedências são respeitadas
            earliest_from_prec = 0.0
            for pred_id, succ_id in instance.precedence:
                if succ_id == task_id and start_times[pred_id - 1] >= 0:
                    pred_finish = start_times[pred_id - 1] + instance.processing_times[pred_id - 1]
                    earliest_from_prec = max(earliest_from_prec, pred_finish)
            
            start_times[task_idx] = max(earliest_from_crane, earliest_from_prec)
            last_finish = start_times[task_idx] + instance.processing_times[task_idx]
            last_pos = instance.task_bays[task_idx]
    
    return start_times