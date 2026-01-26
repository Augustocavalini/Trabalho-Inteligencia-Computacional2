from typing import List, Tuple, Optional, Dict, Callable, Iterable
import random

from Modelagem import (
	QCSPInstance,
	cost_function,
	feasible
)
from Construtivo import constructive_randomized_greedy


def _copy_encoded(encoded1: List[int], encoded2: List[int]) -> Tuple[List[int], List[int]]:
	return list(encoded1), list(encoded2)

# movimentos de vizinhança
def _move_swap_order(
	e1: List[int],
	e2: List[int],
	rng: random.Random,
) -> None:
	"""Swap na ordem mantendo o par (task, crane) nas mesmas posições."""
	if len(e1) >= 2:
		i, j = rng.sample(range(len(e1)), 2)
		e1[i], e1[j] = e1[j], e1[i]
		e2[i], e2[j] = e2[j], e2[i]


def _move_change_crane(
	e1: List[int],
	e2: List[int],
	q: int,
	rng: random.Random,
) -> None:
	"""Muda o guindaste de uma tarefa mantendo a posição na ordem."""
	if not e1:
		return
	i = rng.randrange(len(e1))
	current_crane = e2[i]
	other_cranes = [c for c in range(1, q + 1) if c != current_crane]
	if other_cranes:
		e2[i] = rng.choice(other_cranes)


def _move_swap_cranes(
	e1: List[int],
	e2: List[int],
	rng: random.Random,
) -> None:
	"""Troca guindastes entre duas tarefas na ordem."""
	if len(e2) >= 2:
		i, j = rng.sample(range(len(e2)), 2)
		e2[i], e2[j] = e2[j], e2[i]

# features - guided local search
# feature 1: pares adjacentes na sequência (encoded1)
# feature 2: atribuição tarefa->guindaste (encoded2)
# feature 3: tempo ocioso por guindaste entre tarefas

Feature = Tuple[Tuple, float]


def _feature_adjacent_pairs(encoded1: List[int]) -> List[Feature]:
	features: List[Feature] = []
	for i in range(len(encoded1) - 1):
		a = encoded1[i]
		b = encoded1[i + 1]
		features.append((("adj", a, b), 1.0))
	return features


def _feature_task_crane(encoded1: List[int], encoded2: List[int]) -> List[Feature]:
	features: List[Feature] = []
	for i, task in enumerate(encoded1):
		crane = encoded2[i]
		features.append((("assign", task, crane), 1.0))
	return features


def _feature_idle_time(
	instance: QCSPInstance,
	encoded1: List[int],
	encoded2: List[int],
	start_times: List[float],
    finish_times: List[float],
) -> List[Feature]:
	features: List[Feature] = []
	q = len(instance.cranes_ready)

	last_finish = [float(instance.cranes_ready[i]) for i in range(q)]
	last_pos = [float(instance.cranes_init_pos[i]) for i in range(q)]

	for i, task_id in enumerate(encoded1):
		task_idx = task_id - 1
		crane = encoded2[i] - 1
		start = start_times[task_idx]
		travel = instance.travel_time * abs(instance.task_bays[task_idx] - last_pos[crane])
		idle = max(0.0, start - (last_finish[crane] + travel))
		if idle > 0:
			features.append((("idle", crane + 1, task_id), idle))

		last_finish[crane] = finish_times[task_idx]
		last_pos[crane] = instance.task_bays[task_idx]

	return features

	

# GLS
def guided_local_search_encoded(
	instance: QCSPInstance,
	initial_encoded: Optional[Tuple[List[int], List[int]]] = None,
	max_iters: int = 1000,
	neighborhood_size: int = 80,
	lambda_penalty: float = 0.1,
	seed: Optional[int] = None,
	alpha: float = 0.2,
	criterion: str = "eft",
	alpha_1_cost: float = 1.0,
	alpha_2_cost: float = 0.0,
	feature_list: Optional[List[Callable[..., List[Feature]]]] = None, # funções de feature
	debug: bool = False,
	debug_every: int = 50,
) -> Tuple[List[int], List[int]]:
	"""
	Guided Local Search para QCSP usando representação encoded.

	Args:
		instance: instância QCSP.
		initial_encoded: (encoded1, encoded2). Se None, usa construtivo randomizado.
		max_iters: número máximo de iterações.
		neighborhood_size: número de vizinhos amostrados por iteração.
		lambda_penalty: peso das penalidades no custo aumentado.
		seed: semente para aleatoriedade.
		alpha: aleatoriedade do construtivo se initial_encoded=None.
		criterion: "est" ou "eft" para o construtivo.

	Returns:
		(encoded1, encoded2)
	"""
	rng = random.Random(seed)

	if debug:
		print(f"[GLS] start | max_iters={max_iters} neighborhood_size={neighborhood_size} lambda={lambda_penalty}")

	q = len(instance.cranes_ready)

	if feature_list is None:
		feature_list = [
			_feature_adjacent_pairs,
			_feature_task_crane,
			_feature_idle_time,
		]

	penalties: Dict[Tuple, int] = {}

	if initial_encoded is None:
		encoded1, encoded2 = constructive_randomized_greedy(
			instance, alpha=alpha, seed=seed, criterion=criterion
		)
		if debug:
			print("[GLS] initial solution from constructive_randomized_greedy")
	else:
		encoded1, encoded2 = initial_encoded
		if debug:
			print("[GLS] initial solution provided")

	is_feasible, start_times, finish_times = feasible(instance, encoded1, encoded2)

	if not is_feasible:
		raise ValueError("Solução inicial inválida para GLS.")


	best_e1, best_e2 = _copy_encoded(encoded1, encoded2)
	best_makespan = cost_function(finish_times, alpha_1_cost, alpha_2_cost)

    
	current_e1, current_e2 = _copy_encoded(best_e1, best_e2)
	current_makespan = best_makespan

	for it in range(max_iters):
		candidates: List[Tuple[float, List[int], List[int], float]] = []  # (aug_cost, e1, e2, makespan)

		for _k in range(neighborhood_size):
			e1, e2 = _copy_encoded(current_e1, current_e2)

			move_type = rng.random()
			if move_type < 0.5:
				# Swap na ordem (encoded1 e encoded2 nas mesmas posições)
				_move_swap_order(e1, e2, rng)
			elif move_type < 0.8:
				# Muda crane de uma tarefa (mantém posição na ordem)
				_move_change_crane(e1, e2, q, rng)
			else:
				# Troca cranes entre duas tarefas
				_move_swap_cranes(e1, e2, rng)

			is_feasible_it, start_times_it, finish_times_it = feasible(instance, e1, e2)

			if not is_feasible_it:
				continue

			# calcula custo real
			ms = cost_function(finish_times_it, alpha_1_cost, alpha_2_cost)
            

			# calcula somatório das penalidades das features
			penalty_sum = 0.0
			feature_values: List[Feature] = []
			for feature_func in feature_list:
				if feature_func is _feature_adjacent_pairs:
					feature_values.extend(feature_func(e1))
				elif feature_func is _feature_task_crane:
					feature_values.extend(feature_func(e1, e2))
				else:
					feature_values.extend(feature_func(instance, e1, e2, start_times_it, finish_times_it))

			for key, value in feature_values:
				penalty_sum += penalties.get(key, 0) * value

			# calcula função de custo aumentada
			aug_cost = ms + lambda_penalty * penalty_sum

			# armazena candidato com a função de custo aumentada e a função de custo real
			candidates.append((aug_cost, e1, e2, ms))

		if not candidates:
			if debug:
				print(f"[GLS] no candidates at iter={it}")
			break

		# ordena os candidatos pelo custo aumentado
		candidates.sort(key=lambda x: x[0])
		_, current_e1, current_e2, current_makespan = candidates[0]

		# Atualiza melhor solução encontrada usando função de custo real
		if current_makespan < best_makespan:
			best_makespan = current_makespan
			best_e1, best_e2 = _copy_encoded(current_e1, current_e2)
			if debug:
				print(f"[GLS] new best at iter={it}: {best_makespan:.3f}")

		# Atualiza penalidades (GLS) com base nas features da solução corrente
		is_feasible_cur, start_times_cur, finish_times_cur = feasible(instance, current_e1, current_e2)
		if not is_feasible_cur:
			continue

		current_features: List[Feature] = []
		for feature_func in feature_list:
			if feature_func is _feature_adjacent_pairs:
				current_features.extend(feature_func(current_e1))
			elif feature_func is _feature_task_crane:
				current_features.extend(feature_func(current_e1, current_e2))
			else:
				current_features.extend(
					feature_func(instance, current_e1, current_e2, start_times_cur, finish_times_cur)
				)

		utilities: List[Tuple[float, Tuple]] = []
		for key, value in current_features:
			pen = penalties.get(key, 0)
			utilities.append((value / (1 + pen), key))

		if utilities:
			max_u = max(u for u, _ in utilities)
			for u, key in utilities:
				if abs(u - max_u) < 1e-9:
					penalties[key] = penalties.get(key, 0) + 1

		if debug and (it + 1) % debug_every == 0:
			print(f"[GLS] iter={it+1} current={current_makespan:.3f} best={best_makespan:.3f} penalties={len(penalties)}")

	return best_e1, best_e2
