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

def decoding_solution(
    instance: QCSPInstance,
    result_coding_1: List[int], # vetor de ids de tarefas ordenados pelo start_time
    result_coding_2: List[int], # vetor de guindastes atribuídos às tarefas
):
    """Decodifica a solução representada por dois vetores em uma matriz de ordem por guindaste.

    Args:
        instance: QCSPInstance com dados do problema
        result_coding_1: vetor de ids de tarefas ordenados pelo start_time
        result_coding_2: vetor de guindastes atribuídos às tarefas

    Returns:
        order_matrix: matriz q x m, linha i = guindaste i, valores = ids de tarefas em ordem
    """
    n = len(instance.processing_times)
    q = len(instance.cranes_ready)

    if len(result_coding_1) != n or len(result_coding_2) != n:
        raise ValueError(f"result_coding_1 e result_coding_2 devem ter tamanho {n}")

    order_matrix: List[List[int]] = [[] for _ in range(q)]

    for task_idx, task_id in enumerate(result_coding_1):
        crane_id = result_coding_2[task_idx] - 1
        print(f"Assigning task {task_id} (idx: {task_idx}) to crane {crane_id + 1}")

        order_matrix[crane_id].append(task_id)

    return order_matrix

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
    
    # # Validar matriz: cada tarefa deve aparecer exatamente uma vez
    # seen = set()
    # for crane_idx, row in enumerate(order_matrix):
    #     for task_id in row:
    #         if task_id == 0:
    #             continue
    #         if task_id < 1 or task_id > n:
    #             raise ValueError(f"Tarefa fora de faixa na linha {crane_idx}: {task_id}")
    #         if task_id in seen:
    #             raise ValueError(f"Tarefa repetida na matriz: {task_id}")
    #         seen.add(task_id)
    
    # if len(seen) != n:
    #     missing = [i for i in range(1, n + 1) if i not in seen]
    #     raise ValueError(f"Tarefas ausentes na matriz: {missing}")
    
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

def compute_finish_times(
    instance: QCSPInstance,
    start_times: List[float],
) -> List[float]:
    """Calcula tempos de término a partir dos tempos de início e tempos de processamento."""
    return [start + p for start, p in zip(start_times, instance.processing_times)]

def evaluate_schedule(
    instance: QCSPInstance,
    order_matrix: List[List[int]],
    start_times: List[float],
    finish_times: List[float],
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
    # Verifica cruzamento entre cranes como violação
    # Também registra segmentos de movimento por guindaste para verificar cruzamentos e margem em movimento
    move_segments: List[List[Tuple[float, float, int, int]]] = [[] for _ in range(q)]  # por guindaste: (t0, t1, x0, x1)

    for crane_idx, row in enumerate(order_matrix):
        last_finish = float(instance.cranes_ready[crane_idx]) # considera o cranes_ready como entrada do modelo, evolui dentro do for, não para a instância, TALVEZ TENHAMOS QUE ATUALIZAR CRANE READY
        last_pos = instance.cranes_init_pos[crane_idx]

        for task_id in row:
            if task_id == 0:
                continue
            t_idx = task_id - 1

            move_time = instance.travel_time * abs(instance.task_bays[t_idx] - last_pos)
            earliest = last_finish + move_time # apenas o tempo de movimentação até a bay onde a tarefa será realizada

            if start_times[t_idx] < earliest: # task começou antes do tempo de movimentação necessário
                violations.append(
                    f"guindaste {crane_idx+1} tarefa {task_id} inicia {start_times[t_idx]} < pronto {earliest}"
                )

            # segmento de movimento: assume movimento linear imediatamente antes do início
            # se houver espera, o movimento ocorre nos últimos 'move_time' antes de start_times[t_idx]
            t1 = start_times[t_idx]
            t0 = t1 - move_time
            x0 = last_pos
            x1 = instance.task_bays[t_idx]
            if x0 != x1 and t0 < t1:
                move_segments[crane_idx].append((t0, t1, x0, x1))

            last_finish = finish_times[t_idx]
            last_pos = instance.task_bays[t_idx]

            # verificar se entre essa movimentação ele cruzou com algum outro crane ou feriu a margem de segurança

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

    # Verificação de cruzamentos e margem de segurança durante movimentos
    def sign(x: int) -> int:
        return (x > 0) - (x < 0)

    v = 1.0 / float(instance.travel_time) if instance.travel_time != 0 else 0.0
    for a in range(q):
        for b in range(a + 1, q):
            for (t0a, t1a, xa0, xa1) in move_segments[a]:
                for (t0b, t1b, xb0, xb1) in move_segments[b]:
                    # sobreposição temporal de movimento
                    t0 = max(t0a, t0b)
                    t1 = min(t1a, t1b)

                    if not (t0 < t1):
                        continue

                    sa = sign(xa1 - xa0)
                    sb = sign(xb1 - xb0)

                    if sa == 0 or sb == 0:
                        continue

                    # posições ao tempo t: pa(t) = xa0 + v*sa*(t - t0a), pb(t) = xb0 + v*sb*(t - t0b)
                    # detectar cruzamento (mesma posição em algum t no intervalo)
                    crossed = False
                    if v > 0 and sa != sb:
                        denom = v * float(sa - sb)
                        num = (xb0 - xa0) + v * (sa * t0a - sb * t0b)
                        t_cross = num / denom
                        if t0 <= t_cross <= t1:
                            crossed = True
                            violations.append(
                                f"cruzamento de movimento entre guindastes {a+1} e {b+1} em t={t_cross:.3f}"
                            )

                    # verificar margem de segurança mínima durante a sobreposição
                    if v > 0:
                        # distância nas bordas do intervalo
                        pa_t0 = xa0 + v * sa * (t0 - t0a)
                        pb_t0 = xb0 + v * sb * (t0 - t0b)
                        pa_t1 = xa0 + v * sa * (t1 - t0a)
                        pb_t1 = xb0 + v * sb * (t1 - t0b)
                        min_dist = min(abs(pa_t0 - pb_t0), abs(pa_t1 - pb_t1))
                        if crossed:
                            min_dist = 0.0
                        if min_dist < float(instance.safety_margin):
                            violations.append(
                                f"margem de segurança em movimento violada entre guindastes {a+1} e {b+1}: distância mínima {min_dist:.3f} < {instance.safety_margin}"
                            )

    makespan = max(finish_times) if finish_times else 0
    return {
        "valid": len(violations) == 0,
        "makespan": makespan,
        "violations": violations,
    }
