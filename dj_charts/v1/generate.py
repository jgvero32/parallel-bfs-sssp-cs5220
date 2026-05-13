import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
df = pd.read_csv(csv_file)

serial_time = df[df['type'] == 'serial']['sssp_time_s'].values[0]
par = df[df['type'] == 'parallel'].copy()
d25 = par[par['delta'] == 25].sort_values('threads')
d50 = par[par['delta'] == 50].sort_values('threads')
threads_25 = d25['threads'].values
threads_50 = d50['threads'].values

BLUE         = '#378ADD'
ORANGE       = '#D85A30'
GREY         = '#B4B2A9'
LIGHT_BLUE   = '#B5D4F4'

def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7')
    ax.spines['bottom'].set_color('#D3D1C7')
    ax.yaxis.grid(True, color='#E8E6DF', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#444441')
    ax.yaxis.label.set_color('#444441')
    ax.xaxis.label.set_color('#444441')
    ax.title.set_color('#2C2C2A')

# chart 1 — speedup
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.plot([1, 256], [1, 256], '--', color=GREY, linewidth=1.2, label='Ideal')
ax.plot(threads_25, d25['sssp_speedup'], 'o-', color=BLUE,   linewidth=2, markersize=5, label='Delta=25')
ax.plot(threads_50, d50['sssp_speedup'], 's-', color=ORANGE, linewidth=2, markersize=5, label='Delta=50')
ax.axhline(y=1, color='#888780', linewidth=0.8, linestyle=':')
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xlabel('Thread Count'); ax.set_ylabel('Speedup vs Serial Dijkstra')
ax.set_title('SSSP Speedup vs Thread Count\nsoc-LiveJournal1, numactl --interleave=all')
ax.legend(frameon=False, fontsize=10); style(ax); plt.tight_layout()
plt.savefig('chart1_speedup.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart1 done")

# chart 2 — sssp time
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.axhline(y=serial_time, color='#993C1D', linewidth=1.5, linestyle='-', label=f'Serial Dijkstra ({serial_time:.2f}s)')
ax.plot(threads_25, d25['sssp_time_s'], 'o-', color=BLUE,   linewidth=2, markersize=5, label='Delta=25')
ax.plot(threads_50, d50['sssp_time_s'], 's-', color=ORANGE, linewidth=2, markersize=5, label='Delta=50')
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xlabel('Thread Count'); ax.set_ylabel('SSSP Time (seconds)')
ax.set_title('SSSP Elapsed Time vs Thread Count\nsoc-LiveJournal1, numactl --interleave=all')
ax.legend(frameon=False, fontsize=10); style(ax); plt.tight_layout()
plt.savefig('chart2_sssp_time.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart2 done")

# chart 3 — timing breakdown stacked bar (delta=25)
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
x = np.arange(len(threads_25)); w = 0.55
cols = {
    'parallel_relaxation_s': ('#378ADD', 'Parallel relaxation'),
    'parallel_merge_s':      ('#5BA85A', 'Parallel merge'),
    'snapshot_s':            ('#D85A30', 'Snapshot'),
    'flatten_processed_s':   ('#9B6BB5', 'Flatten'),
    'outgoing_clear_s':      ('#E8A020', 'Outgoing clear'),
    'find_min_bucket_s':     ('#888780', 'Find min bucket'),
}
bottoms = np.zeros(len(threads_25))
for col, (color, label) in cols.items():
    vals = d25[col].values
    ax.bar(x, vals, w, bottom=bottoms, color=color, label=label, zorder=3)
    bottoms += vals
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in threads_25], fontsize=10)
ax.set_xlabel('Thread Count'); ax.set_ylabel('Time (seconds)')
ax.set_title('Timing Breakdown by Section — Delta=25\nsoc-LiveJournal1, numactl --interleave=all')
ax.legend(frameon=False, fontsize=9, loc='upper right'); style(ax); plt.tight_layout()
plt.savefig('chart3_timing_breakdown.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart3 done")

# chart 4 — preprocessing vs sssp
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
x = np.arange(len(threads_25)); w = 0.35
bars1 = ax.bar(x - w/2, d25['sssp_time_s'],     w, color=BLUE,       label='SSSP time', zorder=3)
bars2 = ax.bar(x + w/2, d25['preprocessing_s'], w, color=LIGHT_BLUE,
               edgecolor=BLUE, linewidth=1, label='Preprocessing (CSR build)', zorder=3)
for bar, val in zip(bars1, d25['sssp_time_s'].values):
    if val > 0.3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=7, color='#444441')
for bar, val in zip(bars2, d25['preprocessing_s'].values):
    if val > 0.3:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=7, color='#185FA5')
ax.set_xticks(x); ax.set_xticklabels([str(t) for t in threads_25], fontsize=10)
ax.set_xlabel('Thread Count'); ax.set_ylabel('Time (seconds)')
ax.set_title('SSSP vs Preprocessing Time — Delta=25\nPreprocessing = one-time local CSR build (amortized)')
ax.legend(frameon=False, fontsize=10); style(ax); plt.tight_layout()
plt.savefig('chart4_preprocess_vs_sssp.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart4 done")

# chart 5 — bandwidth
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.axhline(y=400, color='#993C1D', linewidth=1.5, linestyle='--', label='Peak node bandwidth (400 GB/s)')
ax.plot(threads_25, d25['effective_bandwidth_gbs'], 'o-', color=BLUE,   linewidth=2, markersize=5, label='Delta=25')
ax.plot(threads_50, d50['effective_bandwidth_gbs'], 's-', color=ORANGE, linewidth=2, markersize=5, label='Delta=50')
ax.set_xscale('log', base=2)
ax.set_xlabel('Thread Count'); ax.set_ylabel('Effective Memory Bandwidth (GB/s)')
ax.set_title('Effective Memory Bandwidth vs Thread Count\n~4-6% of peak — memory-bound irregular access pattern')
ax.legend(frameon=False, fontsize=10); style(ax); plt.tight_layout()
plt.savefig('chart5_bandwidth.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart5 done")

# chart 6 — delta comparison total vs sssp
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')
ax.axhline(y=serial_time, color='#993C1D', linewidth=1.5, linestyle='-', label=f'Serial Dijkstra ({serial_time:.2f}s)')
ax.plot(threads_25, d25['total_time_s'],  'o-',  color=BLUE,   linewidth=2,   markersize=5, label='Delta=25 (total)')
ax.plot(threads_50, d50['total_time_s'],  's-',  color=ORANGE, linewidth=2,   markersize=5, label='Delta=50 (total)')
ax.plot(threads_25, d25['sssp_time_s'],   'o--', color=BLUE,   linewidth=1.2, markersize=4, alpha=0.6, label='Delta=25 (SSSP only)')
ax.plot(threads_50, d50['sssp_time_s'],   's--', color=ORANGE, linewidth=1.2, markersize=4, alpha=0.6, label='Delta=50 (SSSP only)')
ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
ax.set_xlabel('Thread Count'); ax.set_ylabel('Time (seconds)')
ax.set_title('Total vs SSSP-only Time — Delta Comparison\nSolid = total (incl. preprocessing), Dashed = SSSP only')
ax.legend(frameon=False, fontsize=9); style(ax); plt.tight_layout()
plt.savefig('chart6_delta_comparison.png', dpi=180, bbox_inches='tight'); plt.close()
print("chart6 done")

print("All charts saved.")