from typing import List, Tuple, Optional, Dict, Callable, Iterable
import random

from RandomSeed import resolve_seed

from Modelagem import (
	QCSPInstance,
	cost_function,
	compute_crane_completion_times,
	feasible
)
from Construtivo import constructive_randomized_greedy


def _copy_encoded(encoded1: List[int], encoded2: List[int]) -> Tuple[List[int], List[int]]:
	return list(encoded1), list(encoded2)

# movimentos de vizinhança
def _move_swap_order_and_crane(
	e1: List[int],
	e2: List[int],
	rng: random.Random,
) -> None:
	"""
	Troca a ordem de duas tarefas na sequência e troca também seus guindastes.
	"""
	if len(e1) >= 2:
		i, j = rng.sample(range(len(e1)), 2)
		e1[i], e1[j] = e1[j], e1[i]
		e2[i], e2[j] = e2[j], e2[i]

def _insert_task_other_crane_best(
	instance: QCSPInstance,
	e1: List[int],
	e2: List[int],
	q: int,
	rng: random.Random,
	alpha_1: float = 1.0,
	alpha_2: float = 0.0,
) -> None:
	"""
	Retira uma task aleatória de sua posição e crane e a insere em outra posição de um crane sorteado
	onde melhore a função objetivo. Só aplica a mudança se houver melhoria.
	Usa sorteia um crane aleatório para limitar o espaço de busca e melhorar performance.
	"""
	if len(e1) < 2:
		return
	
	# Calcular custo atual
	is_feasible_cur, start_times_cur, finish_times_cur = feasible(instance, e1, e2)
	if not is_feasible_cur:
		return
	
	crane_completion_cur = compute_crane_completion_times(instance, e1, e2, start_times_cur)
	cost_current = cost_function(finish_times_cur, alpha_1, alpha_2, crane_completion_cur)
	
	# Selecionar uma task aleatória
	task_pos = rng.randint(0, len(e1) - 1)
	task_id = e1[task_pos]
	task_crane = e2[task_pos]
	
	# Remover task da posição atual
	e1_temp = e1[:task_pos] + e1[task_pos + 1:]
	e2_temp = e2[:task_pos] + e2[task_pos + 1:]
	
	best_cost = cost_current
	best_pos = task_pos
	best_crane = task_crane
	
	# Sortear um crane diferente do atual
	other_cranes = [c for c in range(1, q + 1) if c != task_crane]
	
	if not other_cranes:
		return  # Não há outros cranes disponíveis
	
	selected_crane = rng.choice(other_cranes)
	
	# Testar todas as posições naquele crane sorteado
	for insert_pos in range(len(e1_temp) + 1):
		# Criar candidato
		e1_candidate = e1_temp[:insert_pos] + [task_id] + e1_temp[insert_pos:]
		e2_candidate = e2_temp[:insert_pos] + [selected_crane] + e2_temp[insert_pos:]
		
		# Verificar viabilidade
		is_feasible_cand, start_times_cand, finish_times_cand = feasible(instance, e1_candidate, e2_candidate)
		
		if not is_feasible_cand:
			continue
		
		# Calcular custo
		crane_completion_cand = compute_crane_completion_times(instance, e1_candidate, e2_candidate, start_times_cand)
		cost_candidate = cost_function(finish_times_cand, alpha_1, alpha_2, crane_completion_cand)
		
		# Guardar melhor candidato
		if cost_candidate < best_cost:
			best_cost = cost_candidate
			best_pos = insert_pos
			best_crane = selected_crane
	
	# Aplicar mudança apenas se houver melhoria
	if best_cost < cost_current:
		e1.clear()
		e2.clear()
		e1_final = e1_temp[:best_pos] + [task_id] + e1_temp[best_pos:]
		e2_final = e2_temp[:best_pos] + [best_crane] + e2_temp[best_pos:]
		e1.extend(e1_final)
		e2.extend(e2_final)

def _move_swap_cranes(
	e1: List[int],
	e2: List[int],
	rng: random.Random,
) -> None:
	"""
	Troca guindastes entre duas tarefas na ordem.
	"""
	if len(e2) >= 2:
		i, j = rng.sample(range(len(e2)), 2)
		e2[i], e2[j] = e2[j], e2[i]

def swap_order_same_crane(
	e1: List[int],
	e2: List[int],
	q: int,
	rng: random.Random,
) -> None:
	"""
	Troca a ordem de duas tarefas atribuídas ao mesmo guindaste (crane_id).
	"""
	crane_id = rng.randint(1, q)
	indices = [i for i, c in enumerate(e2) if c == crane_id]
	if len(indices) >= 2:
		i, j = rng.sample(indices, 2)
		e1[i], e1[j] = e1[j], e1[i]

def _move_elevator_schedule(
	instance: QCSPInstance,
	e1: List[int],
	e2: List[int],
	rng: random.Random,
) -> None:
	"""
	Reorganiza as tasks de um crane seguindo estratégia de elevador:
	começa pela task mais próxima da posição inicial do crane,
	depois vai até o ponto mais distante atendendo demandas no caminho,
	e volta na direção oposta.
	"""
	if not e1 or not e2:
		return
	
	q = len(instance.cranes_ready)
	crane_id = rng.randint(1, q)
	
	# Encontrar índices das tasks atribuídas ao crane_id
	crane_indices = [i for i, c in enumerate(e2) if c == crane_id]
	
	if len(crane_indices) < 2:
		return  # Não há como reorganizar com menos de 2 tasks
	
	# Posição inicial do crane
	init_pos = instance.cranes_init_pos[crane_id - 1]
	
	# Bays das tasks do crane
	task_ids_in_crane = [e1[i] for i in crane_indices]
	bays_in_crane = [instance.task_bays[task_id - 1] for task_id in task_ids_in_crane]
	
	# Encontrar a task mais próxima da posição inicial
	closest_idx = min(range(len(bays_in_crane)), key=lambda i: abs(bays_in_crane[i] - init_pos))
	closest_bay = bays_in_crane[closest_idx]
	closest_crane_idx = crane_indices[closest_idx]
	
	# Separar tasks em dois grupos: do lado do ponto mais próximo
	tasks_and_bays = list(zip(crane_indices, bays_in_crane))
	
	# Encontrar bay mais distante
	max_bay = max(bays_in_crane)
	min_bay = min(bays_in_crane)
	
	# Decidir direção inicial: vai para o lado mais distante primeiro
	dist_max = abs(max_bay - init_pos)
	dist_min = abs(min_bay - init_pos)
	
	# Criar ordem: primeiro a task mais próxima, depois estratégia de elevador
	first_task = [closest_crane_idx]
	remaining_indices = [idx for idx in crane_indices if idx != closest_crane_idx]
	remaining_bays = [bays_in_crane[crane_indices.index(idx)] for idx in remaining_indices]
	
	if dist_max >= dist_min:
		# Vai para max_bay primeiro, depois volta para min_bay
		# Ordena remaining_indices: primeiro de closest_bay até max_bay em ordem crescente,
		# depois de min_bay até closest_bay em ordem crescente reversa
		def sort_key(idx):
			bay = bays_in_crane[crane_indices.index(idx)]
			# Tarefas entre closest_bay e max_bay primeiro (em ordem crescente)
			if (closest_bay <= bay <= max_bay) or (max_bay <= bay <= closest_bay):
				return (0, bay)
			# Tarefas entre min_bay e closest_bay depois (em ordem decrescente)
			else:
				return (1, -bay)
		
		sorted_remaining = sorted(remaining_indices, key=sort_key)
		sorted_indices = first_task + sorted_remaining
	else:
		# Vai para min_bay primeiro, depois volta para max_bay
		def sort_key(idx):
			bay = bays_in_crane[crane_indices.index(idx)]
			# Tarefas entre closest_bay e min_bay primeiro (em ordem decrescente)
			if (min_bay <= bay <= closest_bay) or (closest_bay <= bay <= min_bay):
				return (0, -bay)
			# Tarefas entre max_bay e closest_bay depois (em ordem crescente)
			else:
				return (1, bay)
		
		sorted_remaining = sorted(remaining_indices, key=sort_key)
		sorted_indices = first_task + sorted_remaining
	
	# Aplicar a nova ordem
	for new_pos, old_pos in enumerate(sorted_indices):
		if crane_indices[new_pos] != old_pos:
			e1[crane_indices[new_pos]], e1[old_pos] = e1[old_pos], e1[crane_indices[new_pos]]
			e2[crane_indices[new_pos]], e2[old_pos] = e2[old_pos], e2[crane_indices[new_pos]]


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
		features.append((("adj", a, b), 50)) # custo igual a média dos tempos de processamento
	return features


def _feature_task_crane(encoded1: List[int], encoded2: List[int]) -> List[Feature]:
	features: List[Feature] = []
	for i, task in enumerate(encoded1):
		crane = encoded2[i]
		features.append((("assign", task, crane), 50.0))
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
	max_iters: int = 2000,
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
	repair_if_infeasible: bool = True,
	allow_infeasible_initial: bool = True,
	stagnation_limit: int = 200,
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
	rng = random.Random(resolve_seed(seed))

	if debug:
		print(f"[GLS] start | max_iters={max_iters} neighborhood_size={neighborhood_size} lambda={lambda_penalty} stagnation_limit={stagnation_limit}")

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
		if repair_if_infeasible:
			if debug:
				print("[GLS] initial solution infeasible; trying strict constructive repair")
			try:
				encoded1, encoded2 = constructive_randomized_greedy(
					instance,
					alpha=alpha,
					seed=seed,
					criterion=criterion,
					debug=debug,
					allow_infeasible_fallback=False,
				)
				is_feasible, start_times, finish_times = feasible(instance, encoded1, encoded2)
			except Exception:
				is_feasible = False

		if not is_feasible and not allow_infeasible_initial:
			raise ValueError("Solução inicial inválida para GLS.")
		if not is_feasible and debug:
			print("[GLS] proceeding with infeasible initial solution")


	best_e1, best_e2 = _copy_encoded(encoded1, encoded2)
	crane_completion = compute_crane_completion_times(instance, best_e1, best_e2, start_times)
	best_makespan = cost_function(finish_times, alpha_1_cost, alpha_2_cost, crane_completion)

    
	current_e1, current_e2 = _copy_encoded(best_e1, best_e2)
	current_makespan = best_makespan
	
	# Contador de estagnação (iterações sem melhoria)
	stagnation_counter = 0

	for it in range(max_iters):
		# Verificar condição de parada por estagnação
		if stagnation_counter >= stagnation_limit:
			if debug:
				print(f"[GLS] stagnation limit reached at iter={it}")
			break
		
		candidates: List[Tuple[float, List[int], List[int], float]] = []  # (aug_cost, e1, e2, makespan)

		for _k in range(neighborhood_size):
			e1, e2 = _copy_encoded(current_e1, current_e2)

			move_type = rng.random()
			if move_type < 0.20:
				# Swap na ordem entre tarefas do mesmo guindaste
				swap_order_same_crane(e1, e2, q, rng)
			elif move_type < 0.40:
				# Elevator schedule: organiza tasks de um crane em padrão de elevador
				_move_elevator_schedule(instance, e1, e2, rng)
			elif move_type < 0.60:
				# Inserir task em outro crane na melhor posição
				_insert_task_other_crane_best(instance, e1, e2, q, rng, alpha_1_cost, alpha_2_cost)
			elif move_type < 0.80:
				# Swap na ordem (encoded1 e encoded2 nas mesmas posições)
				_move_swap_order_and_crane(e1, e2, rng)
			else:
				# Troca cranes entre duas tarefas
				_move_swap_cranes(e1, e2, rng)

			is_feasible_it, start_times_it, finish_times_it = feasible(instance, e1, e2)

			if not is_feasible_it:
				continue

			# calcula custo real
			crane_completion_it = compute_crane_completion_times(instance, e1, e2, start_times_it)
			ms = cost_function(finish_times_it, alpha_1_cost, alpha_2_cost, crane_completion_it)
            
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
			stagnation_counter = 0  # Reinicia contador de estagnação ao encontrar melhor solução
			if debug:
				print(f"[GLS] new best at iter={it}: {best_makespan:.3f}")
		else:
			stagnation_counter += 1  # Incrementa contador se não encontrou melhoria

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
