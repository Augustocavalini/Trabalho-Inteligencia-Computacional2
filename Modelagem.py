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


def encode_solution(
    instance: QCSPInstance,
    order_matrix: List[List[int]],
    start_times: List[float],
):
    """Encodes a solution represented by an order matrix into two vectors.

    Args:
        instance: QCSPInstance with problem data
        order_matrix: q x m matrix, row i = crane i, values = task ids in order
        start_times: list of start times for each task
    Returns:
        result_coding_1: vector of task ids ordered by start_time
        result_coding_2: vector of cranes assigned to each task
    """
    n = len(instance.processing_times)
    q = len(order_matrix)
    i=0

    result_coding_1: List[int] = []
    result_coding_2: List[int] = [0] * n  # Initialize with zeros

    # Sort tasks by start_time
    sorted_indices = sorted(range(n), key=lambda i: start_times[i])
    result_coding_1 = [i + 1 for i in sorted_indices]

    # result_coding 2 representa qual crane executa aquela task
    for crane_idx, row in enumerate(order_matrix):
        for task_idx, task_id in enumerate(result_coding_1):
            if task_id in row:
                result_coding_2[task_idx] = crane_idx + 1


    return result_coding_1, result_coding_2


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


def generate_position_maps(instance, result_coding_1, result_coding_2):
    """
    Generate pos_pre_task_map and pos_during_task_map based on crane movements.
    
    Args:
        instance: QCSP instance with crane initial positions
        result_coding_1: Task order [task_id, ...]
        result_coding_2: Crane assignment [crane_id, ...]
    
    Returns:
        pos_pre_task_map: Position before crane moves to task
        pos_during_task_map: Position while executing task
    """
    crane_positions = {i+1: instance.cranes_init_pos[i] for i in range(len(instance.cranes_init_pos))}
    posi_pre_task_map = []
    posi_during_task_map = []
    
    for i in range(len(result_coding_1)):
        task_id = result_coding_1[i]
        crane_id = result_coding_2[i]
        task_bay = instance.task_bays[task_id - 1]
        
        # Position before moving
        posi_pre_task_map.append(crane_positions[crane_id])
        
        # Position during task execution (at task bay)
        posi_during_task_map.append(task_bay)
        
        # Update crane position after task
        crane_positions[crane_id] = task_bay
    
    return posi_pre_task_map, posi_during_task_map

def _min_distance_and_cross(
        s: int,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> Tuple[float, bool]:
        """
        Calcula a distância mínima entre dois segmentos lineares (a e b) no intervalo de
        sobreposição temporal e indica se houve cruzamento (inversão de ordem).

        Velocidades são constantes e iguais a instance.travel_time (com sinal pela direção).
        """
        t0a, t1a, xa0, xa1 = a
        t0b, t1b, xb0, xb1 = b

        t0 = max(t0a, t0b)
        t1 = min(t1a, t1b)
        if not (t0 < t1):  # Se os dois segmentos estão sendo realizados em períodos que não se interceptam, retorna falso
            return float("inf"), False

        # Podem haver cruzamentos entre cranes parados e cranes em movimento. Por isso pode haver v = 0
        va = 0.0 if t1a == t0a else (xa1 - xa0) / (t1a - t0a)  
        vb = 0.0 if t1b == t0b else (xb1 - xb0) / (t1b - t0b)

        # diferença d(t) = pa(t) - pb(t)
        d0 = (xa0 + va * (t0 - t0a)) - (xb0 + vb * (t0 - t0b))
        d1 = (xa0 + va * (t1 - t0a)) - (xb0 + vb * (t1 - t0b))

        # cruzamento se a ordem inverte ou se d=0 em algum instante
        crossed = d0 == 0 or d1 == 0 or (d0 * d1 < 0)

        # mínimo de |d(t)| no intervalo
        if crossed:
            return s + 1, True  # valor arbitrário > safety_margin para indicar cruzamento
         
        
        k = va - vb
        min_abs = min(abs(d0), abs(d1))
        if k != 0:
            t_star = t0 - d0 / k
            if t0 <= t_star <= t1:
                d_star = (xa0 + va * (t_star - t0a)) - (xb0 + vb * (t_star - t0b))
                min_abs = min(min_abs, abs(d_star))

        return min_abs, crossed

def verify_crane_crossing_and_safety_margins_v2(
    instance, 
    result_coding_1, 
    result_coding_2, 
    posi_pre_task_map, 
    posi_during_task_map,
    start_times,
    finish_times
):
    """
    Verifica cruzamento e violação de margem de segurança considerando sobreposição temporal.
    Detecta cruzamentos tanto em posições estáticas quanto durante movimentos simultâneos.
    
    Returns:
        True se houver violação, False caso contrário
    """
    n = len(result_coding_1) # Número de tarefas
    q = len(instance.cranes_ready)  #Número de guindastes

    # Monta segmentos por guindaste: espera (ocioso), movimento e execução
    # Uma lista para cada crane
    # Segmento: (t0, t1, x0, x1)
    # t0 = tempo antes de começar o movimento
    # t1 = tempo após terminar o movimento
    # x0 = posição inicial do segmento
    # x1 = posição final do segmento
    segments: List[List[Tuple[float, float, float, float]]] = [[] for _ in range(q)]

    horizon = max(finish_times) if finish_times else 0.0

    # Estado corrente por guindaste
    current_time = [float(t) for t in instance.cranes_ready]
    current_pos = [float(p) for p in instance.cranes_init_pos]

    for idx in range(n):
        task_id = result_coding_1[idx] - 1
        crane_id = result_coding_2[idx] - 1

        s = start_times[task_id]
        f = finish_times[task_id]
        bay = float(instance.task_bays[task_id])

        pre = current_pos[crane_id]
        move_time = float(instance.travel_time) * abs(bay - pre)
        move_start = s - move_time

        # Espera (ocioso) até iniciar o movimento
        if move_start > current_time[crane_id]:
            segments[crane_id].append((current_time[crane_id], move_start, pre, pre))

        # Movimento
        if move_time > 0:
            segments[crane_id].append((move_start, s, pre, bay))

        # Execução (posição constante)
        if f > s:
            segments[crane_id].append((s, f, bay, bay))

        current_time[crane_id] = f
        current_pos[crane_id] = bay

    # Espera após última tarefa até o horizonte
    for crane_id in range(q):
        if current_time[crane_id] < horizon:
            segments[crane_id].append((current_time[crane_id], horizon, current_pos[crane_id], current_pos[crane_id]))


    # Ordem esperada baseada nas posições iniciais
    init_pos = [float(p) for p in instance.cranes_init_pos]
    s = float(instance.safety_margin)

    for a in range(q):
        for b in range(a + 1, q):
            if init_pos[a] == init_pos[b]:
                # se começam iguais, qualquer aproximação viola a margem
                expected = 0
            elif init_pos[a] < init_pos[b]:
                expected = -1  # a deve ficar à esquerda de b
            else:
                expected = 1   # a deve ficar à direita de b

            for seg_a in segments[a]:
                for seg_b in segments[b]:# Vai verificar se algum dos segmentos entre os cranes gera cruzamento ou violação de margem
                    
                    min_dist, crossed = _min_distance_and_cross(s, seg_a, seg_b)
                    if min_dist == float("inf"):
                        continue

                    if min_dist <= s:
                        print(
                            f"Safety margin violation: cranes {a+1} and {b+1} "
                            f"min distance {min_dist:.3f} < {instance.safety_margin}"
                        )
                        return True

                    if crossed:
                        print(
                            f"Crossing violation: cranes {a+1} and {b+1} trajectories intersect"
                        )
                        return True

                    # Checagem adicional de ordem quando não cruzou exatamente
                    if expected != 0:
                        t0a, t1a, xa0, xa1 = seg_a
                        t0b, t1b, xb0, xb1 = seg_b
                        t0 = max(t0a, t0b)
                        t1 = min(t1a, t1b)
                        if not (t0 < t1):
                            continue
                        va = 0.0 if t1a == t0a else (xa1 - xa0) / (t1a - t0a)
                        vb = 0.0 if t1b == t0b else (xb1 - xb0) / (t1b - t0b)
                        d0 = (xa0 + va * (t0 - t0a)) - (xb0 + vb * (t0 - t0b))
                        if expected < 0 and d0 > -s:
                            print(
                                f"Crossing/order violation: crane {a+1} should stay left of {b+1} "
                                f"(distance {abs(d0):.3f} < {instance.safety_margin})"
                            )
                            return True
                        if expected > 0 and d0 < s:
                            print(
                                f"Crossing/order violation: crane {a+1} should stay right of {b+1} "
                                f"(distance {abs(d0):.3f} < {instance.safety_margin})"
                            )
                            return True

    return False

def verify_precedence_violations(
    instance: QCSPInstance,
    start_times: List[float],
    finish_times: List[float],
):
    """Verifica violações de precedência.

    Args:
        instance: QCSPInstance com dados do problema
        start_times: tempos de início (tamanho n)
        finish_times: tempos de término (tamanho n)

    Returns:
        Lista de strings descrevendo violações encontradas.
    """
    violations: List[str] = []

    for i, j in instance.precedence:
        if finish_times[i - 1] > start_times[j - 1]:
            violations.append(f"precedence {i}->{j} violada: fim {finish_times[i-1]} > inicio {start_times[j-1]}")

    return violations

def verify_nonsimultaneous_violations(
    instance: QCSPInstance,
    start_times: List[float],
    finish_times: List[float],
):
    """Verifica violações de não simultaneidade.

    Args:
        instance: QCSPInstance com dados do problema
        start_times: tempos de início (tamanho n)
        finish_times: tempos de término (tamanho n)

    Returns:
        Lista de strings descrevendo violações encontradas.
    """
    violations: List[str] = []

    for i, j in instance.nonsimultaneous:
        overlap = not (finish_times[i - 1] <= start_times[j - 1] or finish_times[j - 1] <= start_times[i - 1])

        if overlap:
            violations.append(f"nonsimultaneous {i},{j} se sobrepoe")

    return violations

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
