from typing import List, Tuple, Dict, Set, Optional

from Modelagem import QCSPInstance


# ------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------
def _build_predecessor_map(instance: QCSPInstance) -> List[Set[int]]:
    """Mapa de predecessores 0-based para cada tarefa."""
    n = len(instance.processing_times)
    preds: List[Set[int]] = [set() for _ in range(n)]
    for a, b in instance.precedence:
        preds[b - 1].add(a - 1)  # pares (i,j) são 1-based no arquivo
    return preds


def _last_state_of_crane(
    crane_idx: int,
    scheduled: Dict[int, Tuple[int, float, float]],
    instance: QCSPInstance,
) -> Tuple[float, int]:
    """
    Retorna (finish_time, bay) do último job alocado no guindaste `crane_idx` (0-based).
    Se nenhum, usa ready_time e posição inicial.
    """
    last_finish = instance.cranes_ready[crane_idx]
    last_bay = instance.cranes_init_pos[crane_idx]
    for t, (c, _s, f) in scheduled.items():
        if c == crane_idx and f > last_finish:
            last_finish = f
            last_bay = instance.task_bays[t]
    return last_finish, last_bay


def _pre_bay_for_task(
    task: int,
    crane: int,
    scheduled: Dict[int, Tuple[int, float, float]],
    instance: QCSPInstance,
) -> int:
    """
    Encontra a posição anterior (pre_bay) de uma tarefa já agendada
    com base no último job do mesmo guindaste antes do seu start.
    """
    s_task = scheduled[task][1]
    last_finish = instance.cranes_ready[crane]
    last_bay = instance.cranes_init_pos[crane]
    for t, (c, _s, f) in scheduled.items():
        if c == crane and f <= s_task and f >= last_finish:
            last_finish = f
            last_bay = instance.task_bays[t]
    return last_bay


def _segment_crosses_point(x0: int, x1: int, x: int) -> bool:
    return (x0 < x < x1) or (x1 < x < x0)


def _segments_cross(x0: int, x1: int, y0: int, y1: int) -> bool:
    """Cruzamento em 1D ocorre quando a ordem relativa se inverte."""
    return (x0 - y0) * (x1 - y1) < 0


def _overlap(s1: float, f1: float, s2: float, f2: float) -> bool:
    return not (f1 <= s2 or f2 <= s1)


def _earliest_feasible_start(
    instance: QCSPInstance,
    task: int,          # 0-based
    crane: int,         # 0-based
    scheduled: Dict[int, Tuple[int, float, float]],
    preds: List[Set[int]],
) -> Tuple[float, float]:
    """
    Calcula o earliest start/finish para (task, crane) respeitando:
    - prontidão e deslocamento do guindaste
    - precedência
    - pares não simultâneos
    - margem de segurança entre guindastes
    - cruzamento entre guindastes durante deslocamentos
    Retorna (start, finish).
    """
    proc = instance.processing_times[task]
    bay = instance.task_bays[task]

    # disponibilidade do guindaste (ready + deslocamento a partir do último job)
    last_finish, last_bay = _last_state_of_crane(crane, scheduled, instance)
    start = last_finish + instance.travel_time * abs(bay - last_bay)

    # garante precedência: start >= fim de todos os predecessores
    for p in preds[task]:
        if p in scheduled:
            start = max(start, scheduled[p][2])

    # ajusta para não-simultâneos, margem de segurança e cruzamentos
    changed = True
    while changed:
        changed = False
        for other, (c_o, s_o, f_o) in scheduled.items():
            # não simultâneo
            if (task + 1, other + 1) in instance.nonsimultaneous or (other + 1, task + 1) in instance.nonsimultaneous:
                if _overlap(start, start + proc, s_o, f_o):
                    start = f_o
                    changed = True
                    continue

            # margem de segurança com outros guindastes
            if c_o != crane:
                if _overlap(start, start + proc, s_o, f_o):
                    if abs(bay - instance.task_bays[other]) < instance.safety_margin:
                        start = f_o
                        changed = True
                        continue

                # cruzamento: meu deslocamento vs posição estática do outro
                pre_bay = last_bay
                move_start = start - instance.travel_time * abs(bay - pre_bay)
                if _overlap(move_start, start, s_o, f_o):
                    other_bay = instance.task_bays[other]
                    if _segment_crosses_point(pre_bay, bay, other_bay):
                        start = f_o
                        changed = True
                        continue

                # cruzamento: deslocamento simultâneo entre guindastes
                other_pre_bay = _pre_bay_for_task(other, c_o, scheduled, instance)
                other_move_start = s_o - instance.travel_time * abs(instance.task_bays[other] - other_pre_bay)
                if _overlap(move_start, start, other_move_start, s_o):
                    if _segments_cross(pre_bay, bay, other_pre_bay, instance.task_bays[other]):
                        start = max(start, s_o)
                        changed = True
                        continue

                # cruzamento: deslocamento do outro vs minha posição estática
                if _overlap(other_move_start, s_o, start, start + proc):
                    if _segment_crosses_point(other_pre_bay, instance.task_bays[other], bay):
                        start = max(start, s_o)
                        changed = True
                        continue

        # precedência pode ter sido impactada após empurrar
        for p in preds[task]:
            if p in scheduled and start < scheduled[p][2]:
                start = scheduled[p][2]
                changed = True

    finish = start + proc
    return start, finish


def _serial_schedule(
    instance: QCSPInstance,
    priority: str = "est",               # "est" ou "eft"
    preset_schedule: Optional[Dict[int, Tuple[int, float, float]]] = None,
    restrict_tasks: Optional[Set[int]] = None,
) -> Tuple[List[float], List[int], Dict]:
    """
    Agenda em série escolhendo, a cada passo, o par (tarefa, guindaste) que minimiza
    earliest start (EST) ou earliest finish (EFT), respeitando precedências.
    Se `restrict_tasks` for fornecido, agenda apenas esse subconjunto.
    `preset_schedule` permite começar com tarefas já agendadas.
    """
    n = len(instance.processing_times)
    preds = _build_predecessor_map(instance)

    scheduled: Dict[int, Tuple[int, float, float]] = {}
    if preset_schedule:
        scheduled.update(preset_schedule)

    all_tasks: Set[int] = set(range(n))
    target_tasks: Set[int] = restrict_tasks if restrict_tasks is not None else all_tasks
    remaining = [t for t in target_tasks if t not in scheduled]

    while remaining:
        # tarefas elegíveis (predecessores já agendados)
        eligible = []
        for t in remaining:
            if all(p in scheduled for p in preds[t]):
                eligible.append(t)

        if not eligible:
            # deadlock (provavelmente ciclo ou falta de pred agendado)
            break

        best = None  # (criterion, start, finish, task, crane)
        q = len(instance.cranes_ready)
        for t in eligible:
            for c in range(q):
                s, f = _earliest_feasible_start(instance, t, c, scheduled, preds)
                crit = s if priority == "est" else f
                if best is None or crit < best[0] or (abs(crit - best[0]) < 1e-9 and f < best[2]):
                    best = (crit, s, f, t, c)

        _, s, f, t_sel, c_sel = best
        scheduled[t_sel] = (c_sel, s, f)
        remaining = [t for t in remaining if t != t_sel]

    # monta vetores de saída
    start_times = [0.0] * n
    task_crane = [1] * n  # ids 1..q
    for t, (c, s, f) in scheduled.items():
        start_times[t] = s
        task_crane[t] = c + 1  # volta para 1-based

    return start_times, _to_result_matrix(instance, start_times, task_crane)


def _to_result_matrix(
    instance: QCSPInstance,
    start_times: List[float],
    task_crane: List[int],
    ):
    """
    Gera matriz de resultados no formato esperado (lista de listas), 
    onde a linha é um crane que armazena na ordem de execução as tasks feitas por ele.
    Exemplo:
    [[ 3,  9,  2,  1], # C1
     [ 7,  8, 10,  6,  5,  4]] # C2
    """
    
    q = len(instance.cranes_ready)
    n = len(instance.processing_times)
    result = [[] for _ in range(q)]
    tasks_with_start = [(t, start_times[t]) for t in range(n)]
    tasks_with_start.sort(key=lambda x: x[1])  # ordena por start time
    for t, _ in tasks_with_start:
        c = task_crane[t] - 1  # 0-based
        result[c].append(t + 1)  # 1-based
    
    return result

# ------------------------------------------------------------
# Métodos construtivos
# ------------------------------------------------------------
def constructive_est(instance: QCSPInstance):
    """
    Heurística Gulosa por Earliest Start Time (EST).
    Em cada iteração escolhe (tarefa, guindaste) com menor início viável.
    """
    return _serial_schedule(instance, priority="est")


def constructive_eft(instance: QCSPInstance):
    """
    Heurística Gulosa por Earliest Finish Time (EFT).
    Em cada iteração escolhe (tarefa, guindaste) com menor término viável.
    """
    return _serial_schedule(instance, priority="eft")


def constructive_precedence_first(instance: QCSPInstance):
    """
    Estratégia em duas fases:
      1) Agenda apenas as tarefas que participam de precedência (como no método truncado),
         usando EST.
      2) Agenda as tarefas restantes, respeitando a programação fixa da fase 1.
    """
    n = len(instance.processing_times)
    involved = set()
    for a, b in instance.precedence:
        involved.add(a - 1)
        involved.add(b - 1)

    # Fase 1: apenas tarefas com precedência
    s1, c1 = _serial_schedule(instance, priority="est", restrict_tasks=involved)

    # congela fase 1 em preset_schedule
    preset: Dict[int, Tuple[int, float, float]] = {}
    for t in involved:
        preset[t] = (c1[t] - 1, s1[t], s1[t] + instance.processing_times[t])

    # Fase 2: agenda restantes
    remaining = set(range(n)) - involved
    s2, c2 = _serial_schedule(
        instance,
        priority="est",
        preset_schedule=preset,
        restrict_tasks=remaining,
    )

    # combina resultados (preset já está em s2/c2)
    start_times = s2
    task_crane = c2
    return start_times, task_crane


def _constructive_matrix_based(
    instance: QCSPInstance,
    criterion: str = "est",  # "est" or "eft"
):
    """
    Método construtivo baseado em matriz q x n de EST/EFT.

    A cada iteração:
    - Calcula uma matriz (q x n) com EST ou EFT para todas as tasks não atribuídas.
    - Seleciona o par (crane, task) pelo menor valor do critério.
    - Atualiza o agendamento e repete até completar.

    Respeita precedência, não-simultaneidade, margem de segurança e cruzamentos
    via _earliest_feasible_start().
    """
    n = len(instance.processing_times)
    q = len(instance.cranes_ready)
    preds = _build_predecessor_map(instance)

    scheduled: Dict[int, Tuple[int, float, float]] = {}
    remaining: Set[int] = set(range(n))

    while remaining:
        # tarefas elegíveis (predecessores já agendados)
        eligible = [t for t in remaining if all(p in scheduled for p in preds[t])]
        if not eligible:
            break

        # matriz q x n (None para tasks já agendadas ou não elegíveis)
        matrix: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(q)]

        best = None  # (criterion_value, start, finish, task, crane)
        for c in range(q):
            for t in eligible:
                s, f = _earliest_feasible_start(instance, t, c, scheduled, preds)
                value = s if criterion == "est" else f
                matrix[c][t] = value

                if best is None or value < best[0] or (abs(value - best[0]) < 1e-9 and f < best[2]):
                    best = (value, s, f, t, c)

        if best is None:
            break

        _, s_sel, f_sel, t_sel, c_sel = best
        scheduled[t_sel] = (c_sel, s_sel, f_sel)
        remaining.remove(t_sel)

    # monta vetores de saída
    start_times = [0.0] * n
    task_crane = [1] * n  # ids 1..q
    for t, (c, s, _f) in scheduled.items():
        start_times[t] = s
        task_crane[t] = c + 1  # volta para 1-based

    return start_times, _to_result_matrix(instance, start_times, task_crane)


def constructive_matrix_est(instance: QCSPInstance):
    """
    Construtivo por matriz q x n usando EST.
    """
    return _constructive_matrix_based(instance, criterion="est")


def constructive_matrix_eft(instance: QCSPInstance):
    """
    Construtivo por matriz q x n usando EFT.
    """
    return _constructive_matrix_based(instance, criterion="eft")