import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('Processor Count')
plt.ylabel('Simulation Time (seconds)')
plt.title('1D Parallel BFS Time vs Number of Processors (Balanced)')

# Data for both basic and load-balanced BFS are here
# Switch (SW) shows where to swap roadNet and roadNet_balanced

x = [1, 2, 4, 8, 16, 32, 64]

serial_roadNet = 0.109
serial_liveJournal = 0.524

roadNet = [0.0852474, 0.0591012, 0.0564173, 0.046106, 0.0424121, 0.071909, 0.0810325]
roadNet_balanced = [0.0910, 0.0662, 0.0525, 0.0393, 0.0409, 0.0736, 0.0810]
liveJournal = [0.766427, 0.610269, 0.478562, 0.321123, 0.256492, 0.605042, 0.514343]
liveJournal_balanced = [0.639, 0.471, 0.334, 0.224, 0.163, 0.527, 0.462]

ideal_x = [1, 64]
# SW
ideal_roadNet = [roadNet_balanced[0], roadNet_balanced[0]/64]
# SW
ideal_liveJournal = [liveJournal_balanced[0], liveJournal_balanced[0]/64]

plt.xscale("log", base=2)
plt.yscale("log", base=10)


plt.plot([1], [serial_roadNet], marker='*', markersize = 12, linestyle='None',
         color='purple', label='Serial roadNet-CA')
plt.plot([1], [serial_liveJournal], marker='*', markersize = 12, linestyle='None',
         color='red', label='Serial soc-LiveJournal1')

# SW
plt.plot(x, roadNet_balanced, linestyle='-', color = "blue", marker='s', markersize=4, label = "roadNet-CA")
plt.plot(ideal_x, ideal_roadNet, color = "blue", linestyle="--", label = "Ideal roadNet-CA")
# SW
plt.plot(x, liveJournal_balanced, color = "orange", marker='s', markersize=4, label = "soc-LiveJournal1")
plt.plot(ideal_x, ideal_liveJournal, color = "orange", linestyle="--", label = "Ideal soc-LiveJournal1")
plt.legend()

out_fname = "1d_bfs_plot.svg"
plt.savefig(out_fname, bbox_inches='tight')
