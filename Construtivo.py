from typing import List, Tuple, Dict, Set, Optional

from Modelagem import QCSPInstance


# ------------------------------------------------------------
# Utilitários
# ------------------------------------------------------------
def _build_predecessor_map(instance: QCSPInstance) -> List[Set[int]]:
    """
    Mapa de predecessores 0-based para cada tarefa.
    Retorna uma lista onde o índice é a tarefa (0-based) e o valor é um conjunto de predecessores (0-based).
    """
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
    - cruzamento entre guindastes durante deslocamentos (paralelismo permitido se sem cruzamentos)
    Retorna (start, finish).
    """
    # ============================================
    # FASE 1: INICIALIZAÇÃO E PARÂMETROS
    # ============================================
    proc = instance.processing_times[task]  # tempo de processamento da tarefa
    bay = instance.task_bays[task]  # baía (posição) onde a tarefa será executada
    
    print(f"\n{'='*80}")
    print(f"[EFS] Calculando earliest feasible start para Task={task+1}, Crane={crane+1}")
    print(f"  Processing Time: {proc}, Task Bay: {bay}")

    # ============================================
    # FASE 2: DISPONIBILIDADE DO GUINDASTE (Ready Time + Travel)
    # ============================================
    # Obtém o último job alocado ao guindaste ou sua posição inicial
    last_finish, last_bay = _last_state_of_crane(crane, scheduled, instance)
    
    # Calcula tempo de deslocamento da última posição até a nova baía
    travel_distance = abs(bay - last_bay)
    travel_time_needed = instance.travel_time * travel_distance
    start = last_finish + travel_time_needed
    
    print(f"\n[Fase 2] Disponibilidade do Guindaste:")
    print(f"  Last finish time: {last_finish:.2f}, Last bay: {last_bay}")
    print(f"  Travel distance: {travel_distance} bays, Travel time: {travel_time_needed:.2f}")
    print(f"  → Initial start (sem restrições): {start:.2f}")

    # ============================================
    # FASE 3: VERIFICAÇÃO DE PRECEDÊNCIA INICIAL
    # ============================================
    # Garante que a tarefa só inicia após todos seus predecessores terminarem
    print(f"\n[Fase 3] Restrições de Precedência:")
    if preds[task]:
        print(f"  Predecessores da tarefa {task+1}: {[p+1 for p in preds[task]]}")
        for p in preds[task]:
            if p in scheduled:
                pred_finish = scheduled[p][2]
                if pred_finish > start:
                    print(f"    → Task {p+1} termina em {pred_finish:.2f} (maior que start={start:.2f})")
                    start = max(start, pred_finish)
                else:
                    print(f"    → Task {p+1} termina em {pred_finish:.2f} (compatível com start={start:.2f})")
    else:
        print(f"  Nenhum predecessor para a tarefa {task+1}")
    print(f"  → Start após precedência: {start:.2f}")

    # ============================================
    # FASE 4: AJUSTES SEQUENCIAIS (sem loop) para conflitos
    # ============================================
    print(f"\n[Fase 4] Verificação de Conflitos:")
    
    # 4.1: Restrição de Não-Simultaneidade
    print(f"\n  [4.1] Verificando não-simultaneidade...")
    for other, (c_o, s_o, f_o) in scheduled.items():
        if (task + 1, other + 1) in instance.nonsimultaneous or (other + 1, task + 1) in instance.nonsimultaneous:
            if _overlap(start, start + proc, s_o, f_o):
                print(f"    CONFLITO: Task {task+1} overlaps Task {other+1} [{s_o:.2f}, {f_o:.2f}]")
                print(f"    → Ajuste: start = {f_o:.2f} (esperar fim da tarefa conflitante)")
                start = f_o
    
    # 4.2: Restrição de Margem de Segurança Espacial + Paralelismo
    print(f"\n  [4.2] Verificando margem de segurança espacial...")
    for other, (c_o, s_o, f_o) in scheduled.items():
        if c_o != crane:  # Outro guindaste
            distance_to_other = abs(bay - instance.task_bays[other])
            if distance_to_other < instance.safety_margin:
                # Apenas conflita se as EXECUÇÕES se sobrepõem temporalmente
                if _overlap(start, start + proc, s_o, f_o):
                    print(f"    CONFLITO: Task {task+1} (bay {bay}) próximo a Task {other+1} (bay {instance.task_bays[other]})")
                    print(f"    Distância: {distance_to_other} < Margem: {instance.safety_margin}")
                    print(f"    → Ajuste: start = {f_o:.2f}")
                    start = f_o
    
    # 4.3: Restrição de Cruzamento durante Movimentações
    print(f"\n  [4.3] Verificando cruzamentos durante movimentações...")
    pre_bay = last_bay
    move_start = start - instance.travel_time * abs(bay - pre_bay)
    
    for other, (c_o, s_o, f_o) in scheduled.items():
        if c_o != crane:  # Outro guindaste
            other_bay = instance.task_bays[other]
            other_pre_bay = _pre_bay_for_task(other, c_o, scheduled, instance)
            other_move_start = s_o - instance.travel_time * abs(other_bay - other_pre_bay)
            
            # Tipo 1: Minha movimentação (move_start, start) vs posição estática do outro durante sua execução
            if _overlap(move_start, start, s_o, f_o):
                if _segment_crosses_point(pre_bay, bay, other_bay):
                    print(f"    CRUZAMENTO 1: Meu deslocamento [{pre_bay}→{bay}] durante [{move_start:.2f},{start:.2f}]")
                    print(f"      cruza Task {other+1} em bay {other_bay} durante execução [{s_o:.2f},{f_o:.2f}]")
                    print(f"      → Ajuste: start = {f_o:.2f} (esperar fim da tarefa no cruzamento)")
                    start = f_o
            
            # Tipo 2: Deslocamentos SIMULTÂNEOS - cruzamento perigoso
            elif _overlap(move_start, start, other_move_start, s_o):
                if _segments_cross(pre_bay, bay, other_pre_bay, other_bay):
                    print(f"    CRUZAMENTO 2: Deslocamentos simultâneos perigosos!")
                    print(f"      Eu: [{pre_bay}→{bay}] durante [{move_start:.2f},{start:.2f}]")
                    print(f"      Outro: [{other_pre_bay}→{other_bay}] durante [{other_move_start:.2f},{s_o:.2f}]")
                    print(f"      → Ajuste: start = {max(start, s_o):.2f} (sequencializar movimentações)")
                    start = max(start, s_o)
            
            # Tipo 3: Movimentação do outro durante minha execução - risco de cruzamento
            elif _overlap(other_move_start, s_o, start, start + proc):
                if _segment_crosses_point(other_pre_bay, other_bay, bay):
                    print(f"    CRUZAMENTO 3: Task {other+1} se desloca [{other_pre_bay}→{other_bay}] durante [{other_move_start:.2f},{s_o:.2f}]")
                    print(f"      cruza minha posição bay {bay} durante minha execução [{start:.2f},{start+proc:.2f}]")
                    print(f"      → Ajuste: start = {max(start, s_o):.2f} (sequencializar movimentações)")
                    start = max(start, s_o)

    # 4.4: Re-validar precedência após todos os ajustes
    print(f"\n  [4.4] Re-validando precedência após ajustes...")
    for p in preds[task]:
        if p in scheduled:
            pred_finish = scheduled[p][2]
            if start < pred_finish:
                print(f"    PRECEDÊNCIA: Predecessor Task {p+1} termina em {pred_finish:.2f} > start {start:.2f}")
                print(f"    → Ajuste: start = {pred_finish:.2f}")
                start = pred_finish

    # ============================================
    # FASE 5: RESULTADO FINAL
    # ============================================
    finish = start + proc
    print(f"\n[Fase 5] RESULTADO FINAL:")
    print(f"  Task {task+1} no Crane {crane+1}:")
    print(f"    Start: {start:.2f}, Finish: {finish:.2f}, Duration: {proc}")
    print(f"{'='*80}\n")
    
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


# **********************************************************************

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


# **********************************************************************

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
    # ============================================
    # ETAPA 1: INICIALIZAÇÃO
    # ============================================
    # Extrai informações básicas da instância
    n = len(instance.processing_times)  # número total de tarefas
    q = len(instance.cranes_ready)  # número de guindastes disponíveis
    
    # Constrói mapa de precedências
    preds = _build_predecessor_map(instance)  # preds[t] = conjunto de predecessores da tarefa t
    
    # Estado de controle principal:
    scheduled: Dict[int, Tuple[int, float, float]] = {}  # Dict[task_id] = (crane_id, start_time, finish_time)
    remaining: Set[int] = set(range(n))  # Conjunto de tarefas ainda não agendadas
    
    # ============================================
    # ETAPA 2: LOOP ITERATIVO DE CONSTRUÇÃO
    # ============================================
    while remaining:
        # SUBESTAPA 2.1: Identificar tarefas elegíveis
        # Uma tarefa é elegível se todos os seus predecessores já foram agendados
        eligible = [t for t in remaining if all(p in scheduled for p in preds[t])]
        
        if not eligible:
            # Nenhuma tarefa elegível encontrada → possível ciclo ou precedência incompleta
            print(f"[AVISO] Nenhuma tarefa elegível encontrada. Tarefas restantes: {remaining}")
            break

        # SUBESTAPA 2.2: Construir matriz de critério (q x n)
        # Cada célula [c][t] armazena EST ou EFT para alocar tarefa t ao guindaste c
        # Cells não preenchidas permanecem None
        matrix: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(q)]
        
        # Rastreador para melhor alocação encontrada nesta iteração
        best = None  # Tupla: (criterion_value, start, finish, task, crane)

        print(f"\n[ITERANDO] Tarefas elegíveis: {eligible}\n")

        # SUBESTAPA 2.3: Calcular valores para todas as combinações (guindaste, tarefa elegível)
        for c in range(q):
            for t in eligible:
                # Calcular earliest start (s) e earliest finish (f) para tarefa t no guindaste c
                # Esta função respeita: precedência, não-simultaneidade, margem de segurança, cruzamentos
                print(f"[CÁLCULO] Calculando para Task {t+1} no Crane {c+1}")
                s, f = _earliest_feasible_start(instance, t, c, scheduled, preds)
                print(f"[RESULTADO] Task {t+1} no Crane {c+1}: start={s:.2f}, finish={f:.2f}")
                
                # Aplicar o critério de seleção (EST = start time, EFT = finish time)
                value = s if criterion == "est" else f
                
                # Armazenar na matriz
                matrix[c][t] = value
                
                # SUBESTAPA 2.4: Atualizar melhor candidato
                # Critério de escolha: menor value
                # Em caso de empate: preferir menor finish time (f) como desempate
                if best is None or value < best[0] or (abs(value - best[0]) < 1e-9 and f < best[2]):
                    best = (value, s, f, t, c)

        # SUBESTAPA 2.5: Verificar se encontrou alguma alocação válida
        if best is None:
            print(f"[AVISO] Nenhuma alocação válida encontrada. Tarefas restantes: {remaining}")
            break

        # ============================================
        # ETAPA 3: ATUALIZAR ESTADO DA CONSTRUÇÃO
        # ============================================
        # Desempacotar melhor solução encontrada
        _, s_sel, f_sel, t_sel, c_sel = best
        
        # Adicionar tarefa ao agendamento
        scheduled[t_sel] = (c_sel, s_sel, f_sel)
        
        # Remover tarefa do conjunto de pendentes
        remaining.remove(t_sel)
        
        # [DEBUG] Imprimir progresso da iteração
        print(f"\n[ITERAÇÃO] Task {t_sel+1} alocada ao Crane {c_sel+1}: start={s_sel:.2f}, finish={f_sel:.2f}\n")

    # ============================================
    # ETAPA 4: FORMATAR RESULTADO
    # ============================================
    # Inicializar vetores de saída
    start_times = [0.0] * n  # Tempo de início para cada tarefa
    task_crane = [1] * n  # Guindaste alocado para cada tarefa (1-based)
    
    # Preencher vetores a partir do dicionário agendado
    for t, (c, s, _f) in scheduled.items():
        start_times[t] = s  # Tempo de início (0-based)
        task_crane[t] = c + 1  # ID do guindaste (converter para 1-based)

    # ============================================
    # ETAPA 5: CONVERTER PARA FORMATO DE RESULTADO
    # ============================================
    # Gerar matriz de resultado no formato esperado (lista de tarefas por guindaste)
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