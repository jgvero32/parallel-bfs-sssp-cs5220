import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# FILL IN YOUR DATA HERE
# ============================================================

threads = [1, 4, 8, 16, 32, 64]

# actual times from your runs
parallel_actual = [9.69, 3.64, 2.89, 1.99, 2.22, 1.44]
merge_actual    = [7.30, 7.37, 7.46, 7.61, 10.11, 9.23]

# ============================================================
# computed — no need to change below this line
# ============================================================

parallel_1 = parallel_actual[0]
merge_1    = merge_actual[0]
expected_parallel = [parallel_1 / t for t in threads]
expected_merge    = [merge_1] * len(threads)

# ============================================================

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bar_width  = 0.4
inner_gap  = 0.08
group_gap  = 0.5

tick_positions = []
tick_labels    = []
x = 0

for i, t in enumerate(threads):
    if t == 1:
        ax.bar(x, parallel_actual[i], bar_width, color='#378ADD', zorder=3)
        ax.bar(x, merge_actual[i], bar_width, bottom=parallel_actual[i],
               color='#FAECE7', edgecolor='#D85A30', linewidth=1.2, linestyle='--', zorder=3)
        ax.text(x + bar_width/2, parallel_actual[i]/2, f'{parallel_actual[i]:.1f}s',
                ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
        ax.text(x + bar_width/2, parallel_actual[i] + merge_actual[i]/2, f'{merge_actual[i]:.1f}s',
                ha='center', va='center', fontsize=8.5, color='#993C1D', fontweight='bold')
        total = parallel_actual[i] + merge_actual[i]
        ax.text(x + bar_width/2, total + 0.3, f'{total:.1f}s',
                ha='center', va='bottom', fontsize=8.5, color='#444441')
        tick_positions.append(x + bar_width/2)
        tick_labels.append('1 thread')
        x += bar_width + group_gap
    else:
        x_act = x
        x_exp = x + bar_width + inner_gap

        # actual bar
        ax.bar(x_act, parallel_actual[i], bar_width, color='#378ADD', zorder=3)
        ax.bar(x_act, merge_actual[i], bar_width, bottom=parallel_actual[i],
               color='#FAECE7', edgecolor='#D85A30', linewidth=1.2, linestyle='--', zorder=3)
        if parallel_actual[i] > 0.8:
            ax.text(x_act + bar_width/2, parallel_actual[i]/2, f'{parallel_actual[i]:.1f}s',
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.text(x_act + bar_width/2, parallel_actual[i] + merge_actual[i]/2, f'{merge_actual[i]:.1f}s',
                ha='center', va='center', fontsize=8, color='#993C1D', fontweight='bold')
        total_act = parallel_actual[i] + merge_actual[i]
        ax.text(x_act + bar_width/2, total_act + 0.3, f'{total_act:.1f}s',
                ha='center', va='bottom', fontsize=8, color='#444441')

        # expected bar
        ep = expected_parallel[i]
        em = expected_merge[i]
        ax.bar(x_exp, ep, bar_width, color='#B5D4F4', zorder=3)
        ax.bar(x_exp, em, bar_width, bottom=ep,
               color='#FAECE7', edgecolor='#D85A30', linewidth=1.2, linestyle='--', zorder=3)
        if ep > 0.6:
            ax.text(x_exp + bar_width/2, ep/2, f'{ep:.1f}s',
                    ha='center', va='center', fontsize=8, color='#0C447C', fontweight='bold')
        ax.text(x_exp + bar_width/2, ep + em/2, f'{em:.1f}s',
                ha='center', va='center', fontsize=8, color='#993C1D', fontweight='bold')
        total_exp = ep + em
        ax.text(x_exp + bar_width/2, total_exp + 0.3, f'{total_exp:.1f}s',
                ha='center', va='bottom', fontsize=8, color='#888780')

        ax.text(x_act + bar_width/2, -1.2, 'actual',   ha='center', va='top', fontsize=7.5, color='#888780')
        ax.text(x_exp + bar_width/2, -1.2, 'expected', ha='center', va='top', fontsize=7.5, color='#888780')

        tick_positions.append((x_act + x_exp + bar_width) / 2)
        tick_labels.append(f'{t} threads')
        x += 2*bar_width + inner_gap + group_gap

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=10.5)
ax.set_ylabel('Time (seconds)', fontsize=11, color='#444441')
ax.set_ylim(-2.2, max(p+m for p,m in zip(parallel_actual, merge_actual)) * 1.15)
ax.set_title('Delta-stepping SSSP: actual vs expected time breakdown\n'
             'LiveJournal, delta=50 — expected assumes perfect parallel scaling, merge held constant',
             fontsize=11, color='#2C2C2A', pad=14)
ax.yaxis.set_tick_params(labelcolor='#888780')
ax.xaxis.set_tick_params(labelcolor='#444441')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#D3D1C7')
ax.spines['bottom'].set_visible(False)
ax.yaxis.grid(True, color='#E8E6DF', zorder=0)
ax.set_axisbelow(True)
ax.axhline(0, color='#D3D1C7', linewidth=0.8)

p1 = mpatches.Patch(color='#378ADD', label='Parallel section (actual)')
p2 = mpatches.Patch(color='#B5D4F4', label='Parallel section (expected = parallel_1thread / n)')
p3 = mpatches.Patch(facecolor='#FAECE7', edgecolor='#D85A30', linewidth=1.2, linestyle='--', label='Serial merge')
ax.legend(handles=[p1, p2, p3], fontsize=9.5, frameon=False, loc='upper right')

plt.tight_layout()
plt.savefig('sssp_chart.png', dpi=180, bbox_inches='tight')
print("saved to sssp_chart.png")