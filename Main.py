from pathlib import Path
from Modelagem import *
from Construtivo import *
from BuscaLocal import guided_local_search_encoded
from Genetico import genetic_algorithm_encoded

# Load the QCSP instance
instance_path = Path("instances/QCSP_n10_b10_c200_f50_uni_d100_g0_q2_t1_s1_001.txt")
instance = load_instance(instance_path)

debug = True

def dbg(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

dbg(f"Instance loaded successfully!")
dbg(f"Tasks: {len(instance.processing_times)}")
dbg(f"Bays: {instance.bays}")
dbg(f"Cranes: {len(instance.cranes_ready)}")
dbg()

# após decoding, temos:
# resultado = [[1, 8, 2, 7], # C1
#              [6, 3, 4, 5, 9, 10]] # C2

# start_times, resultado = constructive_est(instance)
criterion = "est"
resultado = constructive_randomized_greedy(instance, alpha = 0.2, criterion=criterion)
encoded1, encoded2 = resultado

dbg(f"Constructive {criterion} result (order matrix):")
dbg(resultado)

order_matrix = decoding_solution(instance, encoded1, encoded2)
start_times = compute_start_times_from_order_matrix(instance, order_matrix)
finish_times = compute_finish_times(instance, start_times)

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



res_encoded1_GLS,res_encoded2_GLS = guided_local_search_encoded(instance, resultado, debug=debug)
result_matrix_GLS = decoding_solution(instance, res_encoded1_GLS, res_encoded2_GLS)
res_start_times_GLS = compute_start_times(instance, res_encoded1_GLS, res_encoded2_GLS)
res_finish_times_GLS = compute_finish_times(instance, res_start_times_GLS)
dbg("Resulting order matrix after GLS:")
dbg(result_matrix_GLS)
dbg()
dbg("Encoded solution after GLS:")
dbg((res_encoded1_GLS, res_encoded2_GLS))
dbg()
dbg("Start times and finish times after GLS:")
dbg(res_start_times_GLS)
dbg(res_finish_times_GLS)
dbg()


res_encoded1_GA,res_encoded2_GA = genetic_algorithm_encoded(instance, debug=debug)
result_matrix_GA = decoding_solution(instance, res_encoded1_GA, res_encoded2_GA)
res_start_times_GA = compute_start_times(instance, res_encoded1_GA, res_encoded2_GA)
res_finish_times_GA = compute_finish_times(instance, res_start_times_GA)
dbg("Resulting order matrix after GA:")
dbg(result_matrix_GA)
dbg()
dbg("Encoded solution after GA:")
dbg((res_encoded1_GA, res_encoded2_GA))
dbg()
dbg("Start times and finish times after GA:")
dbg(res_start_times_GA)
dbg(res_finish_times_GA)
