import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# FILL IN YOUR DATA HERE
# ============================================================

x = [1, 4, 8, 16, 32, 64]

serial_roadNet     = 0.304541   # from ./dj_serial roadNet-CA
serial_liveJournal = 1.34   # from ./dj_serial soc-LiveJournal1

roadNet     = [0.43, 0.73, 0.80, 0.75, 0.78, 0.91]   # replace with dj_parallel roadNet times
liveJournal = [13.95, 10.10, 9.73, 9.32, 9.75, 9.89]  # replace with your numbers

# ============================================================
# computed — no need to change below
# ============================================================

ideal_x            = [x[0], x[-1]]
ideal_roadNet      = [roadNet[0], roadNet[0] / x[-1]]
ideal_liveJournal  = [liveJournal[0], liveJournal[0] / x[-1]]

plt.figure()
plt.xlabel('Thread Count')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Parallel SSSP (delta-stepping) Time vs Number of Threads')

plt.xscale("log", base=2)
plt.yscale("log", base=2)

plt.axhline(y=serial_roadNet,
            color='purple', linestyle='-', linewidth=1.5,
            label='Serial Dijkstra roadNet-CA')
plt.axhline(y=serial_liveJournal,
            color='red', linestyle='-', linewidth=1.5,
            label='Serial Dijkstra soc-LiveJournal1')

plt.plot(x, roadNet,
         linestyle='-', color='blue', marker='s', markersize=4,
         label='roadNet-CA')
plt.plot(ideal_x, ideal_roadNet,
         color='blue', linestyle='--',
         label='Ideal roadNet-CA')

plt.plot(x, liveJournal,
         linestyle='-', color='orange', marker='s', markersize=4,
         label='soc-LiveJournal1')
plt.plot(ideal_x, ideal_liveJournal,
         color='orange', linestyle='--',
         label='Ideal soc-LiveJournal1')

plt.legend()
plt.savefig('sssp_line.svg', bbox_inches='tight')
plt.savefig('sssp_line.png', dpi=180, bbox_inches='tight')
print("saved sssp_line.svg and sssp_line.png")