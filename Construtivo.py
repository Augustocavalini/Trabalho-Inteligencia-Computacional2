from typing import List, Tuple, Dict, Set, Optional
import random

from Modelagem import (
    QCSPInstance,
    compute_start_times_from_order_matrix,
    compute_finish_times,
    generate_position_maps,
    verify_crane_crossing_and_safety_margins_v2,
)


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

def _partial_precedence_ok(
    instance: QCSPInstance,
    start_times: List[float],
    finish_times: List[float],
    scheduled: Set[int],
) -> bool:
    """
    Verifica precedência apenas entre tarefas já agendadas (0-based).
    Baseado em verify_precedence_violations, mas restringe ao conjunto scheduled.
    """
    for i, j in instance.precedence:
        ti = i - 1
        tj = j - 1
        if ti in scheduled and tj in scheduled:
            if finish_times[ti] > start_times[tj]:
                return False
    return True


def _encode_partial_solution(
    instance: QCSPInstance,
    order_matrix: List[List[int]],
    start_times: List[float],
    scheduled: Set[int],
) -> Tuple[List[int], List[int]]:
    """
    Gera result_coding_1 e result_coding_2 apenas para tarefas já agendadas.
    """
    task_crane = [-1] * len(instance.processing_times)
    for crane_idx, row in enumerate(order_matrix):
        for task_id in row:
            if task_id != 0:
                task_crane[task_id - 1] = crane_idx + 1

    tasks_sorted = sorted(list(scheduled), key=lambda t: start_times[t])
    result_coding_1 = [t + 1 for t in tasks_sorted]
    result_coding_2 = [task_crane[t] for t in tasks_sorted]
    return result_coding_1, result_coding_2


# ------------------------------------------------------------
# Método construtivos
# ------------------------------------------------------------

def constructive_randomized_greedy(
    instance: QCSPInstance,
    alpha: float = 0.2,
    seed: Optional[int] = None,
    criterion: str = "eft",
) -> List[List[int]]:
    """
    Construtivo guloso randomizado com controle de aleatoriedade (alpha).

    Em cada iteração:
    - Gera candidatos (task, crane) elegíveis por precedência.
    - Avalia viabilidade (precedência parcial + crossing/margem via verificador).
    - Seleciona aleatoriamente dentro da RCL (restricted candidate list).

    Retorna matriz de ordem por guindaste.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha deve estar entre 0 e 1")
    if criterion not in {"est", "eft"}:
        raise ValueError("criterion deve ser 'est' ou 'eft'")

    rng = random.Random(seed)

    n = len(instance.processing_times)
    q = len(instance.cranes_ready)
    preds = _build_predecessor_map(instance)

    order_matrix: List[List[int]] = [[] for _ in range(q)]
    scheduled: Set[int] = set()

    while len(scheduled) < n:
        eligible = [t for t in range(n) if t not in scheduled and all(p in scheduled for p in preds[t])]
        if not eligible:
            raise ValueError("Nenhuma tarefa elegível encontrada (possível ciclo de precedência).")

        candidates: List[Tuple[float, int, int, List[float]]] = []  # (cost, task, crane, start_times)

        for t in eligible:
            for c in range(q):
                temp_order = [list(row) for row in order_matrix]
                temp_order[c].append(t + 1)

                start_times = compute_start_times_from_order_matrix(instance, temp_order)
                finish_times = compute_finish_times(instance, start_times)

                # Verificação parcial de precedência
                if not _partial_precedence_ok(instance, start_times, finish_times, scheduled | {t}):
                    continue

                # Verificação de crossing/margem usando solução parcial
                result_coding_1, result_coding_2 = _encode_partial_solution(
                    instance, temp_order, start_times, scheduled | {t}
                )
                pos_pre, pos_during = generate_position_maps(instance, result_coding_1, result_coding_2)

                if verify_crane_crossing_and_safety_margins_v2(
                    instance,
                    result_coding_1,
                    result_coding_2,
                    pos_pre,
                    pos_during,
                    start_times,
                    finish_times,
                ):
                    continue

                cost = start_times[t] if criterion == "est" else finish_times[t]
                candidates.append((cost, t, c, start_times))

        if not candidates:
            raise ValueError("Não foi possível encontrar alocação viável sem cruzamento/margem.")

        costs = [c[0] for c in candidates]
        c_min = min(costs)
        c_max = max(costs)
        threshold = c_min + alpha * (c_max - c_min)

        rcl = [cand for cand in candidates if cand[0] <= threshold]
        cost, t_sel, c_sel, _ = rng.choice(rcl)

        order_matrix[c_sel].append(t_sel + 1)
        scheduled.add(t_sel)

    return order_matrix