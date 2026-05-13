import matplotlib.pyplot as plt
import numpy as np



plt.figure()
plt.xlabel('Processor Count')
plt.ylabel('Simulation Time (seconds)')
plt.title('2D Parallel BFS Time vs Number of Processors')

x = [1, 2, 4, 8, 16, 32, 64]

serial_roadNet = 0.109851
serial_liveJournal = 0.526331


roadNet = [7.49731, 6.12282625, 4.65577375, 3.4169075, 2.93155375, 3.24183, 2.4989425]
liveJournal = [1.53426, 1.36278, 1.02048, 0.645681, 0.256492, 0.785436, 0.697652]

ideal_x = [1, 64]
ideal_roadNet = [7.49731, 7.49731/64]
ideal_liveJournal = [1.53426, 1.53426/64]

plt.xscale("log", base=2)
plt.yscale("log", base=2)


plt.plot([1], [serial_roadNet], marker='*', markersize = 12, linestyle='None',
         color='purple', label='Serial roadNet-CA')

plt.plot([1], [serial_liveJournal], marker='*', markersize = 12, linestyle='None',
         color='red', label='Serial soc-LiveJournal1')
plt.plot(x, roadNet, linestyle='-', color = "blue", marker='s', markersize=4, label = "roadNet-CA")
plt.plot(ideal_x, ideal_roadNet, color = "blue", linestyle="--", label = "Ideal roadNet-CA")
plt.plot(x, liveJournal, color = "orange", marker='s', markersize=4, label = "soc-LiveJournal1")
plt.plot(ideal_x, ideal_liveJournal, color = "orange", linestyle="--", label = "Ideal soc-LiveJournal1")
plt.ylim(bottom=5e-3)
plt.legend()

out_fname = "2d_bfs_plot.svg"
plt.savefig(out_fname, bbox_inches='tight')
