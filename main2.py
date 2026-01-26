from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
from Modelagem import *
from Construtivo import *
from BuscaLocal import guided_local_search_encoded
from Genetico import genetic_algorithm_encoded

# Load all QCSP instances
instances_dir = Path("instances")
instances = [load_instance(p) for p in sorted(instances_dir.glob("*.txt"))]


debug = True

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
results: List[Dict[str, Any]] = []

def dbg(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

# dbg(f"Instance loaded successfully!")
# dbg(f"Tasks: {len(instance.processing_times)}")
# dbg(f"Bays: {instance.bays}")
# dbg(f"Cranes: {len(instance.cranes_ready)}")
# dbg()

# após decoding, temos:
resultado = [[1, 2], # C1
             [9, 6, 4, 10, 3, 5, 8, 7]] # C2


instance = instances[0]  # Exemplo: pegar a primeira instância
print(f"Solving instance: {instance.name}")
# start_times, resultado = constructive_est(instance)
criterion = "est"
# resultado = constructive_randomized_greedy(instance, alpha = 0.2, criterion=criterion)

start_times = compute_start_times_from_order_matrix(instance, resultado)
finish_times = compute_finish_times(instance, start_times)

encoded1, encoded2 = encode_solution(instance, resultado, start_times)

dbg("Start times and finish times:")
dbg(start_times)
dbg(finish_times)
dbg()

# codificado1, codificado2 = encode_solution(instance, resultado, start_times)

dbg("Encoded solution (coding):")
dbg((encoded1, encoded2))
dbg()

start_times_encoded = compute_start_times(instance, encoded1, encoded2)
finish_times_encoded = compute_finish_times(instance, start_times_encoded)

dbg("Start times and finish times from encoded solution:")
dbg(start_times_encoded)
dbg(finish_times_encoded)
dbg()

# posi_pre_task_map, posi_during_task_map = generate_position_maps(instance, encoded1, encoded2)

# print("Position maps:")
# print((posi_pre_task_map, posi_during_task_map))
# print()

crossing_violations = verify_crane_crossing_and_safety_margins_v2(
    instance, encoded1, encoded2, start_times, compute_finish_times(instance, start_times)
)
if crossing_violations:
    dbg(crossing_violations)
    dbg("Invalid solution due to crane crossing.\n")
# else:
#     print("No crane crossing detected. Solution is valid.\n")

precedence_breaks_detected = verify_precedence_violations(instance, start_times, finish_times)
if precedence_breaks_detected:
    dbg("Invalid solution due to precedence constraint violations.\n")
# else:
#     print("No precedence constraint violations detected. Solution is valid.\n")

simultaneous_tasks_detected = verify_nonsimultaneous_violations(instance, start_times, start_times)
if simultaneous_tasks_detected:
    dbg("Invalid solution due to simultaneous tasks constraint violations.\n")
# else:
#     print("No simultaneous tasks constraint violations detected. Solution is valid.\n")




[0,0,0,100,0,0,0,0,0,0]
[0,0,0,400,0,0,0,0,0,0]

[]