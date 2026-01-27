from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict, Any

import pandas as pd
from Modelagem import *
from Construtivo import *
from BuscaLocal import guided_local_search_encoded
from Genetico import genetic_algorithm_encoded
from Visualization import plot_gantt_by_bay

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


for instance in instances[0:2]:
    print(f"Solving instance: {instance.name}")
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



    gls_start = time.perf_counter()
    res_encoded1_GLS, res_encoded2_GLS = guided_local_search_encoded(instance, resultado, debug=debug)
    gls_time = time.perf_counter() - gls_start
    result_matrix_GLS = decoding_solution(instance, res_encoded1_GLS, res_encoded2_GLS)
    # res_start_times_GLS = compute_start_times(instance, res_encoded1_GLS, res_encoded2_GLS)
    # res_finish_times_GLS = compute_finish_times(instance, res_start_times_GLS)
    # print("Resulting order matrix after GLS:")
    # print(result_matrix_GLS)
    # print()
    # print("Encoded solution after GLS:")
    # print((res_encoded1_GLS, res_encoded2_GLS))
    # print()
    # print("Start times and finish times after GLS:")
    # print(res_start_times_GLS)
    # print(res_finish_times_GLS)
    # print()
    result = evaluate_schedule(instance, encoded1=res_encoded1_GLS, encoded2=res_encoded2_GLS)
    custo = result["cost_function"]
    print(f"Cost after GLS: {custo}")

    plot_gantt_by_bay(
        instance,
        res_encoded1_GLS,
        res_encoded2_GLS,
        title=f"Gantt GLS - {instance.name}",
        save_path=Path("results") / f"gantt_{instance.name}_GLS.png",
    )


    ga_start = time.perf_counter()
    res_encoded1_GA, res_encoded2_GA = genetic_algorithm_encoded(instance, debug=debug)
    ga_time = time.perf_counter() - ga_start
    result_matrix_GA = decoding_solution(instance, res_encoded1_GA, res_encoded2_GA)
    # res_start_times_GA = compute_start_times(instance, res_encoded1_GA, res_encoded2_GA)
    # res_finish_times_GA = compute_finish_times(instance, res_start_times_GA)
    # print("Resulting order matrix after GA:")
    # print(result_matrix_GA)
    # print()
    # print("Encoded solution after GA:")
    # print((res_encoded1_GA, res_encoded2_GA))
    # print()
    # print("Start times and finish times after GA:")
    # print(res_start_times_GA)
    # print(res_finish_times_GA)
    result = evaluate_schedule(instance, encoded1=res_encoded1_GA, encoded2=res_encoded2_GA)
    custo = result["cost_function"]
    print(f"Cost after GA: {custo}")

    plot_gantt_by_bay(
        instance,
        res_encoded1_GA,
        res_encoded2_GA,
        title=f"Gantt GA - {instance.name}",
        save_path=Path("results") / f"gantt_{instance.name}_GA.png",
    )

    results.append(
        {
            "run_id": run_id,
            "instance": instance.name,
            "gls_cost": evaluate_schedule(instance, encoded1=res_encoded1_GLS, encoded2=res_encoded2_GLS)["cost_function"],
            "ga_cost": evaluate_schedule(instance, encoded1=res_encoded1_GA, encoded2=res_encoded2_GA)["cost_function"],
            "gls_time_s": gls_time,
            "ga_time_s": ga_time,
        }
    )

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"results_{run_id}.xlsx"
pd.DataFrame(results).to_excel(output_path, index=False)
print(f"Resultados salvos em: {output_path}")