import matplotlib.pyplot as plt
import numpy as np



plt.figure()
plt.xlabel('Processor Count')
plt.ylabel('Simulation Time (seconds)')
plt.title('2D Parallel BFS Time vs Number of Processors (Unbalanced)')

x = [1, 2, 4, 8, 16, 32, 64]

serial_roadNet = 0.109851
serial_liveJournal = 0.526331

roadNet = [4.76532, 5.0038, 3.44691, 2.98737, 2.54703, 3.1779, 2.35281]
liveJournal = [0.656489, 0.616164, 0.426637, 0.422192, 0.321941, 0.494534, 0.497369]

ideal_x = [1, 64]
ideal_roadNet = [4.76532, 4.76532/64]
ideal_liveJournal = [0.656489, 0.656489/64]

plt.xscale("log", base=10)
plt.yscale("log", base=10)


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
