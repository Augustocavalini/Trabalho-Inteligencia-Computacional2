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
    pos_pre_task_map = []
    pos_during_task_map = []
    
    for i in range(len(result_coding_1)):
        task_id = result_coding_1[i]
        crane_id = result_coding_2[i]
        task_bay = instance.task_bays[task_id - 1]
        
        # Position before moving
        pos_pre_task_map.append(crane_positions[crane_id])
        
        # Position during task execution (at task bay)
        pos_during_task_map.append(task_bay)
        
        # Update crane position after task
        crane_positions[crane_id] = task_bay
    
    return pos_pre_task_map, pos_during_task_map


# def verify_crane_crossing_and_safety_margins(instance, result_coding_1, result_coding_2, pos_pre_task_map, pos_during_task_map):
#     crane_positions = {i+1: instance.cranes_init_pos[i] for i in range(len(instance.cranes_init_pos))}
    
#     for i in range(len(result_coding_1)): # possível fazer isso pois result_coding_1 está ordenado por start_time
#         task_id = result_coding_1[i]
#         crane_id = result_coding_2[i]
#         pre_pos = pos_pre_task_map[i]
#         during_pos = pos_during_task_map[i]
        
#         # Verificar cruzamento com outros guindastes    
#         for other_crane_id, other_pos in crane_positions.items():
#             if other_crane_id != crane_id:
#                 if (pre_pos < other_pos < during_pos) or (during_pos < other_pos < pre_pos):
#                     # print(f"Cruze detected between crane {crane_id} and crane {other_crane_id} during task {task_id}")
#                     print(f"Crossing violation: crane {crane_id} moving from {pre_pos} to {during_pos} crosses crane {other_crane_id} at position {other_pos}")
#                     print(f"Details: task {task_id}, crane {crane_id} pre_pos {pre_pos}, during_pos {during_pos}, other_crane {other_crane_id}, other_pos {other_pos}")
#                     return True
                
#                 # Verificar margem de segurança
#                 min_distance = abs(during_pos - other_pos)
#                 if min_distance <= instance.safety_margin:
#                     print(f"Safety margin violation: crane {crane_id} at position {during_pos} and crane {other_crane_id} at position {other_pos}, distance {min_distance} < {instance.safety_margin}")
#                     return True
        
#         # Atualizar a posição do guindaste após a tarefa
#         crane_positions[crane_id] = during_pos
    
#     return False





def verify_crane_crossing_and_safety_margins_gabs(instance, result_coding_1, result_coding_2, pos_pre_task_map, pos_during_task_map):
    vet_bay = [0] * instance.bays
    # tempo_atual = start_times[0]
    # MÃO ESTÁ CONSEGUINDO MAPEAR MOVIMENTAÇÃO SIMULTANEA DOS GUINDASTES

    for crane_id, crane_pos in enumerate(instance.cranes_init_pos):
        vet_bay[crane_pos - 1] = crane_id + 1  # Marca a posição inicial dos guindastes

    for i in range(len(result_coding_1)):  # possível fazer isso pois result_coding_1 está ordenado por start_time
        task_id = result_coding_1[i]
        crane_id = result_coding_2[i]
        pre_pos = pos_pre_task_map[i]
        during_pos = pos_during_task_map[i]

        # Atualizar a posição do guindaste após a tarefa
        vet_bay[pre_pos] = 0  # Libera a posição anterior
        vet_bay[during_pos] = crane_id  # Marca a nova posição do guindaste

        # verifico nesse momento se há violação de margem de segurança e de cruzamento
        # pode ser visto caso tenha guindastes em posições que não respeitem a margem de segurança
        # e se um guindaste de id menor apareceu a direita de um com id maior e vice-versa
        for bay_idx in range(instance.bays):
            if vet_bay[bay_idx] != 0:
                for other_bay_idx in range(0, instance.bays):
                    if other_bay_idx != bay_idx:
                        if vet_bay[other_bay_idx] != 0:
                            distance = abs(bay_idx - other_bay_idx)
                            if distance <= instance.safety_margin:
                                print(f"Safety margin violation: crane {vet_bay[bay_idx]} at bay {bay_idx} and crane {vet_bay[other_bay_idx]} at bay {other_bay_idx}, distance {distance} < {instance.safety_margin}")
                                return True
                            if (vet_bay[bay_idx] > vet_bay[other_bay_idx] and bay_idx < other_bay_idx) or (vet_bay[bay_idx] < vet_bay[other_bay_idx] and bay_idx > other_bay_idx):
                                print(f"Crossing violation: crane {vet_bay[bay_idx]} at bay {bay_idx} crossed crane {vet_bay[other_bay_idx]} at bay {other_bay_idx}")
                                return True               

def verify_crane_crossing_and_safety_margins(
    instance, 
    result_coding_1, 
    result_coding_2, 
    pos_pre_task_map, 
    pos_during_task_map,
    start_times,  # ADICIONAR
    finish_times  # ADICIONAR
):
    """
    Verifica cruzamento e violação de margem de segurança considerando sobreposição temporal.
    
    Returns:
        True se houver violação, False caso contrário
    """
    n = len(result_coding_1)
    
    # Para cada par de tarefas
    for i in range(n):
        task_i = result_coding_1[i] - 1
        crane_i = result_coding_2[i]
        start_i = start_times[task_i]
        finish_i = finish_times[task_i]
        bay_i = pos_during_task_map[i]
        
        for j in range(i + 1, n):
            task_j = result_coding_1[j] - 1
            crane_j = result_coding_2[j]
            
            # Apenas verifica se são guindastes diferentes
            if crane_i == crane_j:
                continue
            
            start_j = start_times[task_j]
            finish_j = finish_times[task_j]
            bay_j = pos_during_task_map[j]
            
            # Verifica se há sobreposição temporal
            overlap = not (finish_i <= start_j or finish_j <= start_i)
            
            if overlap:
                distance = abs(bay_i - bay_j)
                
                # Verificar margem de segurança
                if distance < instance.safety_margin:
                    print(f"Safety margin violation: crane {crane_i} (task {task_i+1}) at bay {bay_i} "
                          f"and crane {crane_j} (task {task_j+1}) at bay {bay_j}, "
                          f"distance {distance} < {instance.safety_margin} "
                          f"during overlap [{max(start_i, start_j):.1f}, {min(finish_i, finish_j):.1f}]")
                    return True
                
                # Verificar cruzamento durante movimento
                pre_i = pos_pre_task_map[i]
                pre_j = pos_pre_task_map[j]
                
                # Cruzamento: crane_i vai de pre_i para bay_i e cruza a posição de crane_j
                if (pre_i < bay_j < bay_i) or (bay_i < bay_j < pre_i):
                    print(f"Crossing violation: crane {crane_i} moving from {pre_i} to {bay_i} "
                          f"crosses crane {crane_j} at position {bay_j}")
                    return True
    
    return False

def verify_crane_crossing_and_safety_margins_v2(
    instance, 
    result_coding_1, 
    result_coding_2, 
    pos_pre_task_map, 
    pos_during_task_map,
    start_times,
    finish_times
):
    """
    Verifica cruzamento e violação de margem de segurança considerando sobreposição temporal.
    Detecta cruzamentos tanto em posições estáticas quanto durante movimentos simultâneos.
    
    Returns:
        True se houver violação, False caso contrário
    """
    n = len(result_coding_1)
    
    for i in range(n):
        task_i = result_coding_1[i] - 1
        crane_i = result_coding_2[i]
        start_i = start_times[task_i]
        finish_i = finish_times[task_i]
        bay_i = pos_during_task_map[i]
        pre_i = pos_pre_task_map[i]
        
        # Tempo de movimento do crane_i
        move_time_i = instance.travel_time * abs(bay_i - pre_i)
        move_start_i = start_i - move_time_i
        
        for j in range(i + 1, n):
            task_j = result_coding_1[j] - 1
            crane_j = result_coding_2[j]
            
            if crane_i == crane_j:
                continue
            
            start_j = start_times[task_j]
            finish_j = finish_times[task_j]
            bay_j = pos_during_task_map[j]
            pre_j = pos_pre_task_map[j]
            
            move_time_j = instance.travel_time * abs(bay_j - pre_j)
            move_start_j = start_j - move_time_j
            
            # === 1. VERIFICAR CRUZAMENTO DURANTE MOVIMENTO SIMULTÂNEO ===
            # Overlap temporal dos movimentos
            movement_overlap = not (start_i <= move_start_j or start_j <= move_start_i)
            
            if movement_overlap:
                # Crane_i vai de pre_i para bay_i
                # Crane_j vai de pre_j para bay_j
                # Cruzamento ocorre se as trajetórias se interceptam
                
                # Caso 1: Crane_i cruza a trajetória de crane_j
                if (pre_i < pre_j < bay_i and pre_i < bay_j < bay_i) or \
                   (bay_i < pre_j < pre_i and bay_i < bay_j < pre_i):
                    print(f"Crossing violation during simultaneous movement: "
                          f"crane {crane_i} ({pre_i}→{bay_i}) crosses crane {crane_j} ({pre_j}→{bay_j})")
                    # return True
                
                # Caso 2: Crane_j cruza a trajetória de crane_i
                if (pre_j < pre_i < bay_j and pre_j < bay_i < bay_j) or \
                   (bay_j < pre_i < pre_j and bay_j < bay_i < pre_j):
                    print(f"Crossing violation during simultaneous movement: "
                          f"crane {crane_j} ({pre_j}→{bay_j}) crosses crane {crane_i} ({pre_i}→{bay_i})")
                    # return True
            
            # === 2. VERIFICAR CRUZAMENTO: MOVIMENTO vs POSIÇÃO ESTÁTICA ===
            # Crane_i em movimento cruza crane_j parado
            if move_start_i < finish_j and start_i > start_j:
                if (pre_i < bay_j < bay_i) or (bay_i < bay_j < pre_i):
                    print(f"Crossing violation: crane {crane_i} moving ({pre_i}→{bay_i}) "
                          f"crosses crane {crane_j} stationary at {bay_j}")
                    # return True
            
            # Crane_j em movimento cruza crane_i parado
            if move_start_j < finish_i and start_j > start_i:
                if (pre_j < bay_i < bay_j) or (bay_j < bay_i < pre_j):
                    print(f"Crossing violation: crane {crane_j} moving ({pre_j}→{bay_j}) "
                          f"crosses crane {crane_i} stationary at {bay_i}")
                    # return True
            
            # === 3. VERIFICAR MARGEM DE SEGURANÇA DURANTE EXECUÇÃO ===
            overlap = not (finish_i <= start_j or finish_j <= start_i)
            
            if overlap:
                distance = abs(bay_i - bay_j)
                
                if distance < instance.safety_margin:
                    print(f"Safety margin violation: crane {crane_i} (task {task_i+1}) at bay {bay_i} "
                          f"and crane {crane_j} (task {task_j+1}) at bay {bay_j}, "
                          f"distance {distance} < {instance.safety_margin} "
                          f"during overlap [{max(start_i, start_j):.1f}, {min(finish_i, finish_j):.1f}]")
                    # return True
    
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
