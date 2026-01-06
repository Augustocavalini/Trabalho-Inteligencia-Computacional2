from pathlib import Path
from Modelagem import load_instance

# Load the QCSP instance
instance_path = Path("instances/QCSP_n10_b10_c400_f50_uni_d100_g0_q2_t1_s1_001.txt")
instance = load_instance(instance_path)

print(f"Instance loaded successfully!")
print(f"Tasks: {len(instance.processing_times)}")
print(f"Bays: {instance.bays}")
print(f"Cranes: {len(instance.cranes_ready)}")