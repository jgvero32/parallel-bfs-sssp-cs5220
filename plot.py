import matplotlib.pyplot as plt
import numpy as np



plt.figure()
plt.xlabel('Processor Count')
plt.ylabel('Simulation Time (seconds)')
plt.title('1D Parallel BFS Time vs Number of Processors')

x = [1, 2, 4, 8, 16, 32, 64]

serial_roadNet = 0.109851
serial_liveJournal = 0.526331

roadNet = [0.0852474, 0.0591012, 0.0564173, 0.046106, 0.0424121, 0.071909, 0.0810325]
liveJournal = [0.766427, 0.610269, 0.478562, 0.321123, 0.256492, 0.605042, 0.514343]

ideal_x = [1, 64]
ideal_roadNet = [0.0852474, 0.0852474/64]
ideal_liveJournal = [0.766427, 0.766427/64]

plt.xscale("log", base=2)
plt.yscale("log", base=2)


plt.axhline(y=0.109851,
            color='purple',
            linestyle='-',
            linewidth=1.5,
            label='Serial roadNet-CA')
plt.axhline(y=0.526331,
            color='red',
            linestyle='-',
            linewidth=1.5,
            label='Serial soc-LiveJournal1')
plt.plot(x, roadNet, linestyle='-', color = "blue", marker='s', markersize=4, label = "roadNet-CA")
plt.plot(ideal_x, ideal_roadNet, color = "blue", linestyle="--", label = "Ideal roadNet-CA")
plt.plot(x, liveJournal, color = "orange", marker='s', markersize=4, label = "soc-LiveJournal1")
plt.plot(ideal_x, ideal_liveJournal, color = "orange", linestyle="--", label = "Ideal soc-LiveJournal1")
plt.legend()

out_fname = "plot.svg"
plt.savefig(out_fname, bbox_inches='tight')
