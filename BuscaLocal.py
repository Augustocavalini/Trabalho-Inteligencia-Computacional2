from typing import List, Tuple, Optional
import random

from Modelagem import (
	QCSPInstance,
	compute_start_times_from_order_matrix,
	compute_finish_times,
	evaluate_schedule,
)
from Construtivo import constructive_randomized_greedy


def _copy_matrix(order_matrix: List[List[int]]) -> List[List[int]]:
	return [list(row) for row in order_matrix]


def _flatten(order_matrix: List[List[int]]) -> List[int]:
	return [t for row in order_matrix for t in row]


def _swap_within_crane(order_matrix: List[List[int]], crane_idx: int, i: int, j: int) -> List[List[int]]:
	new_mat = _copy_matrix(order_matrix)
	new_mat[crane_idx][i], new_mat[crane_idx][j] = new_mat[crane_idx][j], new_mat[crane_idx][i]
	return new_mat


def _move_between_cranes(
	order_matrix: List[List[int]],
	from_crane: int,
	to_crane: int,
	task_pos: int,
	insert_pos: int,
) -> List[List[int]]:
	new_mat = _copy_matrix(order_matrix)
	task_id = new_mat[from_crane].pop(task_pos)
	new_mat[to_crane].insert(insert_pos, task_id)
	return new_mat


def _evaluate(instance: QCSPInstance, order_matrix: List[List[int]]) -> Tuple[float, float, dict]:
	start_times = compute_start_times_from_order_matrix(instance, order_matrix)
	finish_times = compute_finish_times(instance, start_times)
	report = evaluate_schedule(instance, order_matrix, start_times, finish_times)
	makespan = report["makespan"]
	return makespan, start_times, report


def guided_local_search(
	instance: QCSPInstance,
	initial_order: Optional[List[List[int]]] = None,
	max_iters: int = 1000,
	neighborhood_size: int = 60,
	lambda_penalty: float = 0.1,
	seed: Optional[int] = None,
	alpha: float = 0.2,
	criterion: str = "eft",
) -> List[List[int]]:
	"""
	Guided Local Search (GLS) para QCSP.

	- Usa apenas vizinhos factíveis (respeitando restrições via evaluate_schedule).
	- Penaliza tarefas com maior utilidade para escapar de ótimos locais.

	Args:
		instance: instância QCSP.
		initial_order: matriz inicial (se None, usa constructive_randomized_greedy).
		max_iters: número máximo de iterações.
		neighborhood_size: número de vizinhos amostrados por iteração.
		lambda_penalty: peso das penalidades no custo aumentado.
		seed: semente para aleatoriedade.
		alpha: aleatoriedade do construtivo se initial_order=None.
		criterion: "est" ou "eft" para o construtivo.

	Returns:
		Matriz de ordem por guindaste.
	"""
	rng = random.Random(seed)

	if initial_order is None:
		initial_order = constructive_randomized_greedy(
			instance, alpha=alpha, seed=seed, criterion=criterion
		)

	best = _copy_matrix(initial_order)
	best_makespan, _, best_report = _evaluate(instance, best)
	if not best_report["valid"]:
		raise ValueError("Solução inicial inválida para GLS.")

	current = _copy_matrix(best)
	current_makespan = best_makespan

	n_tasks = len(instance.processing_times)
	penalties = [0] * n_tasks

	for _ in range(max_iters):
		candidates: List[Tuple[float, List[List[int]], float]] = []  # (aug_cost, matrix, makespan)

		for _k in range(neighborhood_size):
			cand = _copy_matrix(current)
			if rng.random() < 0.5:
				# swap dentro de um guindaste
				c = rng.randrange(len(cand))
				if len(cand[c]) >= 2:
					i, j = rng.sample(range(len(cand[c])), 2)
					cand = _swap_within_crane(cand, c, i, j)
			else:
				# mover tarefa entre guindastes
				c_from = rng.randrange(len(cand))
				if len(cand[c_from]) == 0:
					continue
				c_to = rng.randrange(len(cand))
				if c_to == c_from:
					continue
				pos_from = rng.randrange(len(cand[c_from]))
				pos_to = rng.randrange(len(cand[c_to]) + 1)
				cand = _move_between_cranes(cand, c_from, c_to, pos_from, pos_to)

			makespan, _, report = _evaluate(instance, cand)
			if not report["valid"]:
				continue

			penalty_sum = sum(penalties[t - 1] for t in _flatten(cand))
			aug_cost = makespan + lambda_penalty * penalty_sum
			candidates.append((aug_cost, cand, makespan))

		if not candidates:
			break

		candidates.sort(key=lambda x: x[0])
		current = candidates[0][1]
		current_makespan = candidates[0][2]

		if current_makespan < best_makespan:
			best = _copy_matrix(current)
			best_makespan = current_makespan

		# Atualiza penalidades (GLS): maior utilidade = makespan/(1+penalty)
		flat = _flatten(current)
		utilities = [
			(current_makespan / (1 + penalties[t - 1]), t)
			for t in flat
		]
		max_u = max(u for u, _ in utilities)
		for u, t in utilities:
			if abs(u - max_u) < 1e-9:
				penalties[t - 1] += 1

	return best
