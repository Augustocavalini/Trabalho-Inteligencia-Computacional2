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
    name: str = ""


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
    name = path.stem

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
        name=name,
    )
    
def compute_start_times(
    instance: QCSPInstance,
    encoded1: List[int], # vetor de ids de tarefas ordenados pelo start_time
    encoded2: List[int], # vetor de guindastes atribuídos às tarefas
) -> List[float]:
    """Calcula tempos de início a partir dos vetores codificados.

    Processa cada tarefa na ordem definida por encoded1. Para cada tarefa:
    - Calcula o tempo de pronto do guindaste (ready time + travel time desde última posição)
    - Considera espera por bloqueio de movimento (não cruzamento)
    - Se houver sobreposição com a tarefa anterior em encoded1, força espera até ela terminar
    
    Retorna vetor start_times (tamanho n, indexado por id_tarefa - 1).
    """
    n = len(instance.processing_times)
    q = len(instance.cranes_ready)

    # Estado corrente por guindaste
    last_finish = [float(instance.cranes_ready[i]) for i in range(q)]
    last_pos = [float(instance.cranes_init_pos[i]) for i in range(q)]

    start_times: List[float] = [float("inf")] * n
    for task_idx, task_id in enumerate(encoded1):
        task_id = task_id - 1
        crane_id = encoded2[task_idx] - 1

        move_time = instance.travel_time * abs(instance.task_bays[task_id] - last_pos[crane_id])
        earliest_from_crane = last_finish[crane_id] + move_time
        
        earliest_from_prec = 0.0
        for pred_id, succ_id in instance.precedence:
            if succ_id == task_id + 1 and start_times[pred_id - 1] != float("inf"):
                pred_finish = start_times[pred_id - 1] + instance.processing_times[pred_id - 1]
                earliest_from_prec = max(earliest_from_prec, pred_finish)

        earliest_from_nonsim = 0.0
        for i, j in instance.nonsimultaneous:
            if i == task_id + 1 and start_times[j - 1] != float("inf"):
                j_finish = start_times[j - 1] + instance.processing_times[j - 1]
                earliest_from_nonsim = max(earliest_from_nonsim, j_finish)
            elif j == task_id + 1 and start_times[i - 1] != float("inf"):
                i_finish = start_times[i - 1] + instance.processing_times[i - 1]
                earliest_from_nonsim = max(earliest_from_nonsim, i_finish)

        # Espera por bloqueio: se o caminho cruza posição atual de outro guindaste em execução
        wait_for_block = 0.0
        origin_pos = float(last_pos[crane_id])
        target_pos = float(instance.task_bays[task_id])
        for other_id in range(q):
            if other_id == crane_id:
                continue
            other_pos = float(last_pos[other_id])
            if (origin_pos <= other_pos <= target_pos) or (origin_pos >= other_pos >= target_pos):
                wait_for_block = max(wait_for_block, last_finish[other_id])

        start_times[task_id] = max(
            earliest_from_crane,
            earliest_from_prec,
            earliest_from_nonsim,
            wait_for_block,
        )
        last_finish[crane_id] = start_times[task_id] + instance.processing_times[task_id]
        last_pos[crane_id] = instance.task_bays[task_id]

    return start_times


def compute_crane_completion_times(
    instance: QCSPInstance,
    encoded1: List[int],
    encoded2: List[int],
    start_times: List[float],
) -> List[float]:
    """Calcula o tempo de conclusão de cada guindaste (Y_k)."""
    q = len(instance.cranes_ready)
    crane_completion = [float(instance.cranes_ready[i]) for i in range(q)]
    for idx, task_id in enumerate(encoded1):
        crane_id = encoded2[idx] - 1
        task_idx = task_id - 1
        finish = start_times[task_idx] + instance.processing_times[task_idx]
        if finish > crane_completion[crane_id]:
            crane_completion[crane_id] = finish
    return crane_completion


# def compute_start_times(
#     instance: QCSPInstance,
#     encoded1: List[int], # vetor de ids de tarefas ordenados pelo start_time
#     encoded2: List[int], # vetor de guindastes atribuídos às tarefas)
# ) -> List[float]:
    
#     """Calcula tempos de início a partir dos vetores codificados.

#     Processa cada tarefa na ordem definida por encoded1. Para cada tarefa:
#     - Calcula o tempo de pronto do guindaste (ready time + travel time desde última posição)
#     - Considera espera por bloqueio de movimento (não cruzamento)
#     - Se houver sobreposição com a tarefa anterior em encoded1, força espera até ela terminar
    
#     Retorna vetor start_times (tamanho n, indexado por id_tarefa - 1).
#     """
#     n = len(instance.processing_times)
#     q = len(instance.cranes_ready)

#     # Estado corrente por guindaste
#     last_finish = [float(instance.cranes_ready[i]) for i in range(q)]
#     last_pos = [float(instance.cranes_init_pos[i]) for i in range(q)]

#     last_finish_global = 0.0
#     start_times: List[float] = [float("inf")] * n
    
#     #encoded1 [3, 1, 4, 5, 7, 2, 6, 8, 9, 10]
#     #encoded2 [2, 1, 2, 2, 2, 1, 2, 2, 2, 2]
    
#     for task_idx, task_id in enumerate(encoded1):
#         task_id = task_id - 1
#         crane_id = encoded2[task_idx] - 1

#         move_time = instance.travel_time * abs(instance.task_bays[task_id] - last_pos[crane_id])
#         earliest_from_crane = last_finish[crane_id] + move_time
        # posso ter que esperar outro crane liberar minha movimentação para ir para o local da tarefa

    

def compute_start_times_from_order_matrix(
    instance: QCSPInstance,
    order_matrix: List[List[int]],
) -> List[float]:
    """Calcula tempos de início a partir da matriz de ordem e restrições de precedência.

    Processa cada guindaste em sequência. Para cada tarefa na ordem:
    - Calcula o tempo de pronto do guindaste (ready time + travel time desde última posição)
    
    Retorna vetor start_times (tamanho n, indexado por id_tarefa - 1).
    """
    n = len(instance.processing_times)
    
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
    i = 0

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
        # print(f"Assigning task {task_id} (idx: {task_idx}) to crane {crane_id + 1}")

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
    start_times,
    finish_times
):
    """
    Verifica cruzamento e violação de margem de segurança considerando sobreposição temporal.
    Detecta cruzamentos tanto em posições estáticas quanto durante movimentos simultâneos.
    
    Returns:
        Lista de violações encontradas (strings). Lista vazia indica solução válida.
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

    violations: List[str] = []

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
                        violations.append(
                            f"safety margin: cranes {a+1} and {b+1} min distance {min_dist:.3f} < {instance.safety_margin}"
                        )

                    if crossed:
                        violations.append(
                            f"crossing: cranes {a+1} and {b+1} trajectories intersect"
                        )

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
                            violations.append(
                                f"order: crane {a+1} should stay left of {b+1} (distance {abs(d0):.3f} < {instance.safety_margin})"
                            )
                        if expected > 0 and d0 < s:
                            violations.append(
                                f"order: crane {a+1} should stay right of {b+1} (distance {abs(d0):.3f} < {instance.safety_margin})"
                            )

    return violations

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


def verify_interference_order_violations(
    instance: QCSPInstance,
    encoded1: List[int],
    encoded2: List[int],
    start_times: List[float],
    finish_times: List[float],
):
    """Verifica a restrição de interferência (Eq. 11) por ordenação de guindastes.

    Para tarefas simultâneas i e j com l_i < l_j, o guindaste que atende i
    deve estar à esquerda (ordem) do guindaste que atende j.
    """
    n = len(instance.processing_times)

    # mapeia tarefa -> guindaste
    task_crane = [-1] * n
    for idx, task_id in enumerate(encoded1):
        task_crane[task_id - 1] = encoded2[idx]

    # define ordem dos guindastes por posição inicial
    crane_positions = [(instance.cranes_init_pos[k], k + 1) for k in range(len(instance.cranes_ready))]
    crane_positions.sort(key=lambda x: (x[0], x[1]))
    crane_rank = {crane_id: rank for rank, (_pos, crane_id) in enumerate(crane_positions)}

    violations: List[str] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            li = instance.task_bays[i]
            lj = instance.task_bays[j]
            if li >= lj:
                continue

            overlap = not (finish_times[i] <= start_times[j] or finish_times[j] <= start_times[i])
            if not overlap:
                continue

            ci = task_crane[i]
            cj = task_crane[j]
            if ci == -1 or cj == -1:
                continue

            if crane_rank[ci] > crane_rank[cj]:
                violations.append(
                    f"interference order: task {i+1} (bay {li}) and task {j+1} (bay {lj}) overlap with crane {ci} right of crane {cj}"
                )

    return violations

def evaluate_schedule(
    instance: QCSPInstance,
    alpha_1: float = 1.0,
    alpha_2: float = 0.0,
    order_matrix: List[List[int]] = None,
    encoded1: List[int] = None,
    encoded2: List[int] = None,
):
    """Avalia a qualidade de um agendamento.

    Args:
        instance: QCSPInstance com dados do problema
        order_matrix: matriz de ordem (q x m)
        encoded1: vetor de ids de tarefas ordenados pelo start_time
        encoded2: vetor de guindastes atribuídos às tarefas

    Returns:
        dicionário com métricas de avaliação
    """
    if order_matrix is None:
        if encoded1 is None or encoded2 is None:
            raise ValueError("Forneça order_matrix ou encoded1 e encoded2.")
        order_matrix = decoding_solution(instance, encoded1, encoded2)
        
    if encoded1 is None or encoded2 is None:
        encoded1, encoded2 = encode_solution(
            instance, order_matrix, compute_start_times_from_order_matrix(instance, order_matrix)
        )

    start_times = compute_start_times(instance, encoded1, encoded2)
    finish_times = compute_finish_times(instance, start_times)

    crane_completion = compute_crane_completion_times(instance, encoded1, encoded2, start_times)

    max_makespan = max(crane_completion) if crane_completion else 0.0
    total_completion = sum(crane_completion)

    cost_function = alpha_1 * max_makespan + alpha_2 * total_completion

    precedence_violations = verify_precedence_violations(instance, start_times, finish_times)
    nonsimultaneous_violations = verify_nonsimultaneous_violations(instance, start_times, finish_times)
    interference_violations = verify_interference_order_violations(
        instance, encoded1, encoded2, start_times, finish_times
    )
    crossing_violations = verify_crane_crossing_and_safety_margins_v2(
        instance, encoded1, encoded2, start_times, finish_times
    )

    report = {
        "cost_function": cost_function,
        "max_makespan": max_makespan,
        "total_completion": total_completion,
        "crane_completion_times": crane_completion,
        "precedence_violations": precedence_violations,
        "nonsimultaneous_violations": nonsimultaneous_violations,
        "interference_violations": interference_violations,
        "crossing_violations": crossing_violations,
    }

    return report


def cost_function(
    finish_times,
    alpha_1,
    alpha_2,
    crane_completion_times: List[float] = None,
) -> float:
    """Calcula a função objetivo a partir do makespan e da soma dos Y_k."""

    if crane_completion_times is None:
        max_makespan = max(finish_times) if finish_times else 0.0
        total_completion = sum(finish_times)
    else:
        max_makespan = max(crane_completion_times) if crane_completion_times else 0.0
        total_completion = sum(crane_completion_times)

    return alpha_1 * max_makespan + alpha_2 * total_completion


def feasible(instance: QCSPInstance, encoded1: List[int], encoded2: List[int]) -> bool:
    """Verifica se um agendamento é viável (sem violações)."""

    start_times = compute_start_times(instance, encoded1, encoded2)
    finish_times = compute_finish_times(instance, start_times)

    precedence_violations = verify_precedence_violations(instance, start_times, finish_times)
    nonsimultaneous_violations = verify_nonsimultaneous_violations(instance, start_times, finish_times)
    interference_violations = verify_interference_order_violations(
        instance, encoded1, encoded2, start_times, finish_times
    )
    crossing_violations = verify_crane_crossing_and_safety_margins_v2(
        instance, encoded1, encoded2, start_times, finish_times
    )

    return not (
        precedence_violations
        or nonsimultaneous_violations
        or interference_violations
        or crossing_violations
    ), start_times, finish_times
