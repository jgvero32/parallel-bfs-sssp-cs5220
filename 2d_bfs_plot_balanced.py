import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('Processor Count')
plt.ylabel('Simulation Time (seconds)')
plt.title('2D Parallel BFS Time vs Number of Processors (Balanced)')

# Data for both basic and load-balanced BFS are here
# Switch (SW) shows where to swap roadNet and roadNet_balanced

x = [1, 2, 4, 8, 16, 32, 64]

serial_roadNet = 0.109
serial_liveJournal = 0.524

roadNet = [4.76532, 5.0038, 3.44691, 2.98737, 2.54703, 3.1779, 2.35281]
roadNet_balanced = [4.02209, 6.13254, 3.56732, 3.59116, 2.4388, 3.37274, 2.22202]
liveJournal = [0.656489, 0.616164, 0.426637, 0.422192, 0.321941, 0.494534, 0.497369]
liveJournal_balanced = [0.631548, 0.716758, 0.319335, 0.359383, 0.336126, 0.52704, 0.57269]

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

out_fname = "2d_bfs_plot_balanced.svg"
plt.savefig(out_fname, bbox_inches='tight')
