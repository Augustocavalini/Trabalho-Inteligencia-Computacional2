from pathlib import Path
from Modelagem import load_instance, evaluate_schedule, compute_start_times_from_order_matrix, compute_finish_times, decoding_solution

# Load the QCSP instance
instance_path = Path("instances/QCSP_n10_b10_c400_f50_uni_d100_g0_q2_t1_s1_001.txt")
instance = load_instance(instance_path)

print(f"Instance loaded successfully!")
print(f"Tasks: {len(instance.processing_times)}")
print(f"Bays: {instance.bays}")
print(f"Cranes: {len(instance.cranes_ready)}")

result_coding_1 = [1, 3, 4, 5, 6, 7, 2, 8, 9, 10] # ordenado pelo start_time
result_coding_2 = [1, 2, 2, 2, 2, 2, 1, 2, 2, 2] # crane que fez cada task

pos_pre_task_map = [1, 3, 3, 4, 4, 6, 1, 7, 8, 10] # posição do crane antes de se movimentar para a task, mapeados pelo result_coding_1 e 2
pos_during_task_map = [1, 3, 4, 4, 6, 7, 2, 8, 10, 10] # posição do crane durante a execução da task, mapeados pelo result_coding_1 e 2
# Deve ser consideradas as posicoes iniciais dos guindastes
# essa é uma verificação possível para o coding do resultado, quando a metaheurística genética for implementada. 

# após decoding, temos:
# resultado = [[1, 2], # C1
#              [3, 4, 5, 6, 7, 8, 9,  10]] # C2
resultado = decoding_solution(instance, result_coding_1, result_coding_2)
print("Decoded solution (order matrix):")
print(resultado)

# start_times  = [0, 263, 0, 16, 154, 172, 178, 579, 965, 1164]
# finish_times = [262, 643, 15, 154, 170, 177, 578, 963, 1164, 1365]

start_times = compute_start_times_from_order_matrix(instance, resultado)
finish_times = compute_finish_times(instance, start_times)

print(start_times)
print(finish_times)

# evaluate_schedule(instance, resultado, start_times, finish_times)
