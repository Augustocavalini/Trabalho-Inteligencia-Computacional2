from pathlib import Path
from Modelagem import *
from Construtivo import *

# Load the QCSP instance
instance_path = Path("instances/QCSP_n10_b10_c200_f50_uni_d100_g0_q2_t1_s1_001.txt")
instance = load_instance(instance_path)

print(f"Instance loaded successfully!")
print(f"Tasks: {len(instance.processing_times)}")
print(f"Bays: {instance.bays}")
print(f"Cranes: {len(instance.cranes_ready)}")
print()

# após decoding, temos:
# resultado = [[1, 8, 2, 7], # C1
#              [6, 3, 4, 5, 9, 10]] # C2

# start_times, resultado = constructive_est(instance)
criterion = "est"
resultado = constructive_randomized_greedy(instance, alpha = 0.2, criterion=criterion)

print(f"Constructive {criterion} result (order matrix):")
print(resultado)

start_times = compute_start_times_from_order_matrix(instance, resultado)
finish_times = compute_finish_times(instance, start_times)

print("Start times and finish times:")
print(start_times)
print(finish_times)
print()

codificado1, codificado2 = encode_solution(instance, resultado, start_times)

print("Encoded solution (coding):")
print((codificado1, codificado2))
print()

posi_pre_task_map, posi_during_task_map = generate_position_maps(instance, codificado1, codificado2)

print("Position maps:")
print((posi_pre_task_map, posi_during_task_map))
print()

crossing_detected2 = verify_crane_crossing_and_safety_margins_v2(instance, codificado1, codificado2, posi_pre_task_map, posi_during_task_map, start_times, compute_finish_times(instance, start_times))
if crossing_detected2:
    print("Invalid solution due to crane crossing.\n")
else:
    print("No crane crossing detected. Solution is valid.\n")

# precedence_breaks_detected = verify_precedence_violations(instance, resultado1, start_times1)
# if precedence_breaks_detected:
#     print("Invalid solution due to precedence constraint violations.\n")
# else:
#     print("No precedence constraint violations detected. Solution is valid.\n")

# simultaneous_tasks_detected = verify_nonsimultaneous_violations(instance, resultado1, start_times1)
# if simultaneous_tasks_detected:
#     print("Invalid solution due to simultaneous tasks constraint violations.\n")
# else:
#     print("No simultaneous tasks constraint violations detected. Solution is valid.\n")


