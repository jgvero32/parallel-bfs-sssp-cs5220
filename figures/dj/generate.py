import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

CYAN    = "#28DAF1"
MAGENTA = "#df55dd"
GREY    = "#3F2D21"
RED     = "#144E51"

def style(ax, xticks=None, xticklabels=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D3D1C7')
    ax.spines['bottom'].set_color('#D3D1C7')
    ax.yaxis.grid(True, color='#E8E6DF', zorder=0, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors='#444441', labelsize=9)
    ax.yaxis.label.set_color('#444441')
    ax.xaxis.label.set_color('#444441')
    ax.title.set_color('#2C2C2A')
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticklabels], fontsize=9)

def label_points(ax, xs, ys, color, fmt='{:.1f}x', offset=(4, 4), fontsize=8, skip_threshold=None):
    prev_y = None
    for x, y in zip(xs, ys):
        if skip_threshold and y < skip_threshold:
            continue
        label = fmt.format(y)
        # alternate above/below if points are close
        dy = offset[1]
        if prev_y is not None and abs(y - prev_y) < 0.15 * y:
            dy = -offset[1] - 8
        ax.annotate(label, xy=(x, y), xytext=(offset[0], dy),
                    textcoords='offset points', fontsize=fontsize,
                    color=color, fontweight='500')
        prev_y = y


# ── Chart 1: SSSP Speedup vs Threads ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

all_threads = sorted(set([1] + list(threads_25) + list(threads_50)))
ideal_x = np.array([1] + list(threads_25))
ideal_y = ideal_x.astype(float)

ax.plot(ideal_x, ideal_y, '--', color=GREY, linewidth=1.2, label='Ideal linear', zorder=1)

# serial point at (1, 1)
ax.scatter([1], [1.0], color='black', zorder=5, s=50)
ax.annotate('serial\n1.0x', xy=(1, 1.0), xytext=(6, -14),
            textcoords='offset points', fontsize=7.5, color='#444441')

ax.plot(threads_25, d25['sssp_speedup'], 'o-', color=CYAN,    linewidth=2, markersize=5, label='Delta=25', zorder=3)
ax.plot(threads_50, d50['sssp_speedup'], 's-', color=MAGENTA, linewidth=2, markersize=5, label='Delta=50', zorder=3)

label_points(ax, threads_25, d25['sssp_speedup'].values, CYAN,    skip_threshold=1.5)
label_points(ax, threads_50, d50['sssp_speedup'].values, MAGENTA, skip_threshold=1.5)

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
style(ax, xticks=all_threads, xticklabels=all_threads)
ax.set_xlabel('Thread count')
ax.set_ylabel('Speedup vs serial Dijkstra')
ax.set_title('SSSP speedup vs thread count — soc-LiveJournal1')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('chart1_speedup.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart1 done")


# ── Chart 2: SSSP Time vs Threads ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

ax.axhline(y=serial_time, color=RED, linewidth=1.5, linestyle='-',
           label=f'Serial Dijkstra ({serial_time:.2f}s)', zorder=2)
# serial point
ax.scatter([1], [serial_time], color=RED, zorder=5, s=50)
ax.annotate(f'{serial_time:.2f}s', xy=(1, serial_time), xytext=(6, 4),
            textcoords='offset points', fontsize=7.5, color=RED)

ax.plot(threads_25, d25['sssp_time_s'], 'o-', color=CYAN,    linewidth=2, markersize=5, label='Delta=25', zorder=3)
ax.plot(threads_50, d50['sssp_time_s'], 's-', color=MAGENTA, linewidth=2, markersize=5, label='Delta=50', zorder=3)

label_points(ax, threads_25, d25['sssp_time_s'].values, CYAN,    fmt='{:.2f}s', skip_threshold=0.5)
label_points(ax, threads_50, d50['sssp_time_s'].values, MAGENTA, fmt='{:.2f}s', skip_threshold=0.5)

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
style(ax, xticks=all_threads, xticklabels=all_threads)
ax.set_xlabel('Thread count')
ax.set_ylabel('SSSP time (seconds)')
ax.set_title('SSSP elapsed time vs thread count — soc-LiveJournal1')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('chart2_sssp_time.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart2 done")


# ── Chart 3: Timing Breakdown Stacked Bar (delta=25) ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

x = np.arange(len(threads_25)); w = 0.55
cols = {
    'parallel_relaxation_s': ('#00BCD4', 'Parallel relaxation'),
    'parallel_merge_s':      ("#AFDE8F", 'Parallel merge'),
    'snapshot_s':            ('#E91E8C', 'Snapshot'),
    'flatten_processed_s':   ("#71219C", 'Flatten'),
    'outgoing_clear_s':      ("#FFA200", 'Outgoing clear'),
    'find_min_bucket_s':     ('#888780', 'Find min bucket'),
}
bottoms = np.zeros(len(threads_25))
for col, (color, label) in cols.items():
    vals = d25[col].values
    ax.bar(x, vals, w, bottom=bottoms, color=color, label=label, zorder=3)
    bottoms += vals

style(ax, xticks=x, xticklabels=threads_25)
ax.set_xlabel('Thread count')
ax.set_ylabel('Time (seconds)')
ax.set_title('Timing breakdown by section — delta=25, soc-LiveJournal1')
ax.legend(frameon=False, fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig('chart3_timing_breakdown.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart3 done")


# ── Chart 4: Preprocessing vs SSSP Time ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

x = np.arange(len(threads_25)); w = 0.35
bars1 = ax.bar(x - w/2, d25['sssp_time_s'],     w, color=CYAN,
               label='SSSP time', zorder=3)
bars2 = ax.bar(x + w/2, d25['preprocessing_s'], w, color=MAGENTA,
               alpha=0.55, edgecolor=MAGENTA, linewidth=1,
               label='Preprocessing (CSR build)', zorder=3)

for bar, val in zip(bars1, d25['sssp_time_s'].values):
    if val > 0.25:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=7, color='#185FA5')
for bar, val in zip(bars2, d25['preprocessing_s'].values):
    if val > 0.25:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f'{val:.2f}s', ha='center', va='bottom', fontsize=7, color='#880050')

style(ax, xticks=x, xticklabels=threads_25)
ax.set_xlabel('Thread count')
ax.set_ylabel('Time (seconds)')
ax.set_title('SSSP vs preprocessing time — delta=25\npreprocessing = one-time local CSR build (amortized)')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('chart4_preprocess_vs_sssp.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart4 done")


# ── Chart 5: Effective Bandwidth vs Threads ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

ax.axhline(y=400, color=RED, linewidth=1.5, linestyle='--',
           label='Peak node bandwidth (400 GB/s)', zorder=2)
ax.annotate('400 GB/s peak', xy=(256, 400), xytext=(-60, 6),
            textcoords='offset points', fontsize=8, color=RED)

ax.plot(threads_25, d25['effective_bandwidth_gbs'], 'o-', color=CYAN,    linewidth=2, markersize=5, label='Delta=25', zorder=3)
ax.plot(threads_50, d50['effective_bandwidth_gbs'], 's-', color=MAGENTA, linewidth=2, markersize=5, label='Delta=50', zorder=3)

label_points(ax, threads_25, d25['effective_bandwidth_gbs'].values, CYAN,    fmt='{:.1f}', skip_threshold=5)
label_points(ax, threads_50, d50['effective_bandwidth_gbs'].values, MAGENTA, fmt='{:.1f}', skip_threshold=5)

ax.set_xscale('log', base=2)
style(ax, xticks=all_threads, xticklabels=all_threads)
ax.set_xlabel('Thread count')
ax.set_ylabel('Effective memory bandwidth (GB/s)')
ax.set_title('Effective memory bandwidth vs thread count\n~4–6% of peak — irregular random access pattern')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('chart5_bandwidth.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart5 done")


# ── Chart 6: Delta Comparison Total vs SSSP ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white'); ax.set_facecolor('white')

ax.axhline(y=serial_time, color=RED, linewidth=1.5, linestyle='-',
           label=f'Serial Dijkstra ({serial_time:.2f}s)', zorder=2)
ax.scatter([1], [serial_time], color=RED, zorder=5, s=50)

ax.plot(threads_25, d25['total_time_s'],  'o-',  color=CYAN,    linewidth=2,   markersize=5, label='Delta=25 total')
ax.plot(threads_50, d50['total_time_s'],  's-',  color=MAGENTA, linewidth=2,   markersize=5, label='Delta=50 total')
ax.plot(threads_25, d25['sssp_time_s'],   'o--', color=CYAN,    linewidth=1.2, markersize=4, alpha=0.6, label='Delta=25 SSSP only')
ax.plot(threads_50, d50['sssp_time_s'],   's--', color=MAGENTA, linewidth=1.2, markersize=4, alpha=0.6, label='Delta=50 SSSP only')

ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
style(ax, xticks=all_threads, xticklabels=all_threads)
ax.set_xlabel('Thread count')
ax.set_ylabel('Time (seconds)')
ax.set_title('Total vs SSSP-only time — delta comparison\nsolid = total (incl. preprocessing), dashed = SSSP only')
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig('chart6_delta_comparison.png', dpi=180, bbox_inches='tight')
plt.close()
print("chart6 done")

print("All charts saved.")