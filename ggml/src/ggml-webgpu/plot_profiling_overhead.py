import matplotlib.pyplot as plt
import numpy as np

# Data from your table
master_cpu = np.array([2007.79,1709.87,1662.81,1583.79,1814.26,1574.71])
master_cpu_gpu = np.array([3784.33,2107.13,2182.68,2050.66,2994.26,2502.39])

branch_cpu = np.array([1640.96,1346.75,1692.78,1454.69,1727.39,1368.00])
branch_cpu_gpu = np.array([1667.28,1492.46,1545.30,1603.37,1628.10,1517.78])

# Compute overhead
master_overhead = master_cpu_gpu - master_cpu
branch_overhead = branch_cpu_gpu - branch_cpu

labels = ["run1","run2","run3","run4","run5","run6"]
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()

ax.bar(x - width/2, master_overhead, width, label="master")
ax.bar(x + width/2, branch_overhead, width, label="my branch")

ax.set_ylabel("GPU profiling overhead (ms)")
ax.set_title("GPU Profiling Overhead Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()