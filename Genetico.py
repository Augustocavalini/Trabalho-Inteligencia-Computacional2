from typing import List, Tuple, Optional
import random

from Modelagem import QCSPInstance, feasible, cost_function, compute_crane_completion_times
from Construtivo import constructive_randomized_greedy



Encoded = Tuple[List[int], List[int]]


def _copy_encoded(e: Encoded) -> Encoded:
	return list(e[0]), list(e[1])


def _encoded_to_hashable(e: Encoded) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
	"""
	Converte uma solução encoded para uma forma hashable (tuplas imutáveis) 
	para comparação e detecção de duplicatas.
	"""
	return (tuple(e[0]), tuple(e[1]))


def _fitness(
	instance: QCSPInstance,
	encoded1: List[int],
	encoded2: List[int],
	alpha_1: float,
	alpha_2: float,
	penalty_infeasible: float,
) -> float:
	is_feasible, start_times, finish = feasible(instance, encoded1, encoded2)
	crane_completion = compute_crane_completion_times(instance, encoded1, encoded2, start_times)
	base = cost_function(finish, alpha_1, alpha_2, crane_completion)
	if not is_feasible:
		return base + penalty_infeasible
	return base


def _tournament_select(
	pop: List[Encoded],
	fitnesses: List[float],
	rng: random.Random,
	k: int = 3,
) -> Encoded:
	"""
	Seleção por torneio. Seleciona k indivíduos aleatórios e retorna o melhor entre eles.
	"""
	idxs = rng.sample(range(len(pop)), k=min(k, len(pop))) # seleciona k índices aleatórios da população
	best = min(idxs, key=lambda i: fitnesses[i])
	return _copy_encoded(pop[best])


def _tournament_select_two_different(
	population: List[Encoded],
	fitnesses: List[float],
	rng: random.Random,
	k: int = 3,
) -> Tuple[Encoded, Encoded]:
	"""
	Seleciona dois pais diferentes usando torneio.
	Garante que os índices dos pais sejam diferentes na população,
	portanto sempre retorna soluções diferentes.
	"""
	if len(population) < 2:
		# Se população tem menos de 2 indivíduos, retorna o mesmo indivíduo duas vezes
		return _copy_encoded(population[0]), _copy_encoded(population[0])
	
	# Selecionar índice do primeiro pai
	idxs1 = rng.sample(range(len(population)), k=min(k, len(population)))
	best_idx1 = min(idxs1, key=lambda i: fitnesses[i])
	
	# Selecionar índice do segundo pai garantindo que seja diferente do primeiro
	remaining_indices = [i for i in range(len(population)) if i != best_idx1]
	idxs2 = rng.sample(remaining_indices, k=min(k, len(remaining_indices)))
	best_idx2 = min(idxs2, key=lambda i: fitnesses[i])
	
	return _copy_encoded(population[best_idx1]), _copy_encoded(population[best_idx2])


def _order_crossover_circ(
	parent1: Encoded,
	parent2: Encoded,
	rng: random.Random,
) -> Encoded:
	"""
	Define o crossover de ordem circular (OX-CIRC): copia segmento do pai 1 e 
	preenche o restante com a ordem do pai 2 a partir do final da janela de crossover.
	"""
	p1, c1 = parent1
	p2, c2 = parent2
	n = len(p1)

	i, j = sorted(rng.sample(range(n), 2)) # gera aleatoriamente dois pontos de corte para representar a janela de crossover
	child_encoded1 = [-1] * n
	child_encoded2 = [-1] * n
    
	# copia segmento
	child_encoded1[i:j] = p1[i:j] # copia janela do encode 1 do pai 1 para o encode 1 do filho
	child_encoded2[i:j] = c1[i:j] # copia janela do encode 2 do pai 1 para o encode 2 do filho

	# preenche restante com ordem do outro pai
	pos = j
	for k in range(n):
		gene = p2[(j + k) % n]
		# if gene in child_encoded1: # para o encoded_1 tem que haver garantia de não repetição, já para o encoded_2 não
		if gene in child_encoded1[i:j]:
			continue
		child_encoded1[pos % n] = gene

		# herda crane do mesmo índice do p2
		crane_gene = c2[(j + k) % n]
		child_encoded2[pos % n] = crane_gene
		pos += 1

	return child_encoded1, child_encoded2

def _order_crossover_left_right(
	parent1: Encoded,
	parent2: Encoded,
	rng: random.Random,
) -> Encoded:
	"""
	Define o crossover de ordem left-right (OX-LR): copia segmento do pai 1 e 
	preenche o restante com a ordem do pai 2 da esquerda para a direita.
	"""
	p1, c1 = parent1
	p2, c2 = parent2
	n = len(p1)

	if n < 2:
		return _copy_encoded(parent1)

	i, j = sorted(rng.sample(range(n), 2))
	child_encoded1 = [-1] * n
	child_encoded2 = [-1] * n

	# copia segmento
	child_encoded1[i:j] = p1[i:j]
	child_encoded2[i:j] = c1[i:j]

	# preenche da esquerda para direita em um único loop
	# segment_set = set(p1[i:j])
	pos = 0
	for idx in range(n):
		gene = p2[idx]
		# if gene in segment_set:
		if gene in child_encoded1[i:j]:
			continue
		while i <= pos < j:
			pos = j
		child_encoded1[pos] = gene
		child_encoded2[pos] = c2[idx]
		pos += 1

	return child_encoded1, child_encoded2



def _mutate_swap_order(e1: List[int], e2: List[int], rng: random.Random) -> None:
	"""
	Troca a ordem de duas tarefas aleatórias seja qual for o guindaste.
	"""
	if len(e1) >= 2:
		i, j = rng.sample(range(len(e1)), 2)
		e1[i], e1[j] = e1[j], e1[i]
		e2[i], e2[j] = e2[j], e2[i]

def _mutate_change_crane(e2: List[int], q: int, rng: random.Random) -> None:
	"""
	Altera o guindaste atribuído a uma tarefa aleatória.
	"""
	if not e2:
		return
	i = rng.randrange(len(e2))
	current = e2[i]
	options = [c for c in range(1, q + 1) if c != current]
	if options:
		e2[i] = rng.choice(options)

def _mutate_swap_order_same_crane(e1: List[int], e2: List[int], q: int, rng: random.Random) -> None:
	"""
	Troca a ordem de duas tarefas atribuídas ao mesmo guindaste (crane_id).
	"""

	crane_id = rng.randint(1, q)
	indices = [i for i, c in enumerate(e2) if c == crane_id]
	if len(indices) >= 2:
		i, j = rng.sample(indices, 2)
		e1[i], e1[j] = e1[j], e1[i]


# Mutações Nominais (afetam apenas uma task)
def _mutate_insert_random_position(e1: List[int], e2: List[int], rng: random.Random) -> None:
	"""
	Remove uma tarefa aleatória e a reinsere em uma posição também aleatória.
	Afeta apenas uma tarefa.
	"""
	if len(e1) < 2:
		return
	
	# Selecionar uma tarefa aleatória
	task_pos = rng.randint(0, len(e1) - 1)
	task_id = e1[task_pos]
	task_crane = e2[task_pos]
	
	# Remover a tarefa
	e1.pop(task_pos)
	e2.pop(task_pos)
	
	# Reinseri em posição aleatória
	new_pos = rng.randint(0, len(e1))
	e1.insert(new_pos, task_id)
	e2.insert(new_pos, task_crane)


def _mutate_swap_adjacent_tasks(e1: List[int], e2: List[int], rng: random.Random) -> None:
	"""
	Troca duas tarefas adjacentes na sequência.
	Afeta apenas duas tarefas.
	"""
	if len(e1) < 2:
		return
	
	# Selecionar uma posição aleatória para a primeira tarefa
	i = rng.randint(0, len(e1) - 2)
	j = i + 1
	
	# Trocar as duas tarefas adjacentes
	e1[i], e1[j] = e1[j], e1[i]
	e2[i], e2[j] = e2[j], e2[i]


# Mutações Globais (afetam múltiplas tasks)
def _mutate_reverse_segment(e1: List[int], e2: List[int], rng: random.Random) -> None:
	"""
	Inverte um segmento aleatório da sequência.
	Afeta múltiplas tarefas globalmente.
	"""
	if len(e1) < 3:
		return
	
	# Selecionar dois pontos aleatórios para definir o segmento
	i, j = sorted(rng.sample(range(len(e1)), 2))
	
	# Inverter o segmento
	e1[i:j+1] = reversed(e1[i:j+1])
	e2[i:j+1] = reversed(e2[i:j+1])


def _mutate_reassign_cranes_segment(e1: List[int], e2: List[int], q: int, rng: random.Random) -> None:
	"""
	Seleciona um segmento da sequência e reatribui novos cranes aleatórios para todas as tarefas do segmento.
	Afeta múltiplas tarefas globalmente.
	"""
	if len(e1) < 2:
		return
	
	# Selecionar um segmento aleatório
	i, j = sorted(rng.sample(range(len(e1)), 2))
	
	# Reatribuir novos cranes para todas as tarefas no segmento
	for idx in range(i, j + 1):
		e2[idx] = rng.randint(1, q)


def _apply_mutation(child: Encoded, instance: QCSPInstance, rng: random.Random) -> None:
	"""
	Aplica uma mutação ao indivíduo, escolhendo entre mutações nominais e globais.
	50% de chance: mutação nominal (afeta 1-2 tarefas)
	50% de chance: mutação global (afeta múltiplas tarefas)
	
	Mutações Nominais (5 opções):
	- _mutate_insert_random_position
	- _mutate_swap_adjacent_tasks
	- _mutate_swap_order
	- _mutate_change_crane
	- _mutate_swap_order_same_crane
	
	Mutações Globais (2 opções):
	- _mutate_reverse_segment
	- _mutate_reassign_cranes_segment
	"""
	e1, e2 = child
	q = len(instance.cranes_ready)
	
	mutation_type = rng.random()
	
	if mutation_type < 0.5:  # Mutações Nominais
		nominal_choice = rng.random()
		if nominal_choice < 0.2:
			_mutate_insert_random_position(e1, e2, rng)
		elif nominal_choice < 0.4:
			_mutate_swap_adjacent_tasks(e1, e2, rng)
		elif nominal_choice < 0.6:
			_mutate_swap_order(e1, e2, rng)
		elif nominal_choice < 0.8:
			_mutate_change_crane(e2, q, rng)
		else:
			_mutate_swap_order_same_crane(e1, e2, q, rng)
	else:  # Mutações Globais
		global_choice = rng.random()
		if global_choice < 0.5:
			_mutate_reverse_segment(e1, e2, rng)
		else:
			_mutate_reassign_cranes_segment(e1, e2, q, rng)



def genetic_algorithm_encoded(
	instance: QCSPInstance,
	pop_size: int = 30,
	generations: int = 200,
	crossover_rate: float = 0.7,
	mutation_rate: float = 0.3,
	seed: Optional[int] = None,
	alpha_1: float = 1.0,
	alpha_2: float = 0.0,
	penalty_infeasible: float = 1e6,
	debug: bool = False,
	debug_every: int = 25,
) -> Encoded:
	"""
	Algoritmo Genético para QCSP usando encoded solution.

	Returns:
		(encoded1, encoded2)
	"""
	rng = random.Random(seed)

	n = len(instance.processing_times)
	q = len(instance.cranes_ready)

	def _random_encoded() -> Encoded:
		order = list(range(1, n + 1))
		rng.shuffle(order)
		cranes = [rng.randint(1, q) for _ in range(n)]
		return order, cranes

	if debug:
		print(
			f"[GA] start | pop={pop_size} generations={generations} crossover={crossover_rate} mutation={mutation_rate}"
		)

	# Inicialização
	# 3 soluções via construtivo + restante aleatório
	print("Gerando população inicial...")
	constructive_count = min(3, pop_size)
	population: List[Encoded] = [
		constructive_randomized_greedy(instance, alpha=0.2, criterion="eft", debug=debug)
		for _ in range(constructive_count)
	]
	population.extend(_random_encoded() for _ in range(pop_size - constructive_count))
	
	# Garantir que a população inicial contém apenas soluções diferentes
	seen = set()
	unique_pop: List[Encoded] = []
	for sol in population:
		hashable = _encoded_to_hashable(sol)
		if hashable not in seen:
			seen.add(hashable)
			unique_pop.append(sol)
	
	# Se houver duplicatas, gerar novas soluções até atingir o tamanho desejado
	while len(unique_pop) < pop_size:
		sol = _random_encoded()
		hashable = _encoded_to_hashable(sol)
		if hashable not in seen:
			seen.add(hashable)
			unique_pop.append(sol)
	
	population = unique_pop
	if debug:
		print("[GA] initial population generated - all unique solutions")

    # Verifica dentro da população inicial se há soluções inviáveis. Se uma solução for inviável, aplica uma penalidade alta ao seu fitness, com o objetivo de desencorajar sua seleção para reprodução.
	for gen in range(generations):
		fitnesses = [
			_fitness(instance, e1, e2, alpha_1, alpha_2, penalty_infeasible)
			for (e1, e2) in population
		]

		new_pop: List[Encoded] = []
		# elitismo simples. Adiciona o melhor indivíduo da geração anterior
		best_idx = min(range(len(population)), key=lambda i: fitnesses[i]) # encontra o índice do melhor indivíduo na população atual, com base nos valores de fitness calculados.
		new_pop.append(_copy_encoded(population[best_idx])) # adiciona uma cópia do melhor indivíduo à nova população.

		# Rastrear soluções já adicionadas à nova população para evitar duplicatas
		seen_in_new_pop = set()
		seen_in_new_pop.add(_encoded_to_hashable(new_pop[0]))

		while len(new_pop) < pop_size: 
			# enquanto o tamanho da nova população for menor que o tamanho desejado da população, 
			# realiza seleção, crossover e mutação para gerar novos indivíduos.

			# Seleciona dois pais diferentes garantindo índices diferentes na população
			p1, p2 = _tournament_select_two_different(population, fitnesses, rng)

			# crossover. Pode gera um filho a partir dos dois pais selecionados ou copiar diretamente o pai 1
			if rng.random() < crossover_rate: # aplica crossover com certa probabilidade
				child = _order_crossover_left_right(p1, p2, rng)
			else: # sem crossover, apenas copia o pai 1
				child = _copy_encoded(p1) 

			# mutações. Pode ocorrer tanto em um filho vindo de crossover quanto em um filho copiado diretamente do pai
			if rng.random() < mutation_rate:
				_apply_mutation(child, instance, rng)

			# Verificar se a solução já existe na nova população
			child_hashable = _encoded_to_hashable(child)
			if child_hashable not in seen_in_new_pop:
				seen_in_new_pop.add(child_hashable)
				# adiciona filho à nova população
				# pode vir de um crossover ou de uma cópia do pai 1
				# pode ou não ter sofrido mutação
				new_pop.append(child)
			else:
				# Se for duplicata, incrementar tentativas de gerar nova solução
				if debug:
					print("[GA][loop] solução duplicada descartada, gerando nova")

		population = new_pop

		if debug and (gen + 1) % debug_every == 0:
			best_fit = min(fitnesses) if fitnesses else float("inf")
			print(f"[GA] gen={gen+1} best_fit={best_fit:.3f}")

	# retorna melhor
	fitnesses = [
		_fitness(instance, e1, e2, alpha_1, alpha_2, penalty_infeasible)
		for (e1, e2) in population
	]
	best_idx = min(range(len(population)), key=lambda i: fitnesses[i])

	encoded_1, encoded_2 = _copy_encoded(population[best_idx])
	if debug:
		final_fit = _fitness(instance, encoded_1, encoded_2, alpha_1, alpha_2, penalty_infeasible)
		print(f"[GA] end | best_fit={final_fit:.3f}")
	return encoded_1, encoded_2