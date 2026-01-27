from typing import Optional
import random

# Seed global para todas as aleatorizações do projeto.
GLOBAL_SEED: int = 67


def set_global_seed(seed: int) -> None:
	"""Atualiza a seed global usada como padrão."""
	global GLOBAL_SEED
	GLOBAL_SEED = int(seed)


def resolve_seed(seed: Optional[int]) -> int:
	"""Retorna a seed informada ou a seed global padrão."""
	return GLOBAL_SEED if seed is None else int(seed)


def create_rng(seed: Optional[int] = None) -> random.Random:
	"""Cria um gerador random.Random usando a seed global quando seed=None."""
	return random.Random(resolve_seed(seed))
