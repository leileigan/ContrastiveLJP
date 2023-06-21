import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman Cyr')

x1 = [0.1, 0.5, 1, 10, 20]
#x1 = [1, 2, 3, 4, 5]
accu_y1 = [81.99, 81.81, 82.30, 83.41, 83.35]
law_y1 = [76.61, 76.19, 77.39, 78.08, 77.22]
term_y1 = [34.92, 34.46, 34.88, 32.93, 33.52]

fig = plt.figure(figsize=(15, 7))

ax1 = fig.add_subplot()
ax1.spines['bottom'].set_linewidth(4)
ax1.spines['left'].set_linewidth(4)
ax1.spines['top'].set_linewidth(4)
ax1.spines['right'].set_linewidth(4)
ax1.tick_params(labelsize=25)

marker_size=15
ax1.plot(x1, accu_y1, color='b', marker='o', linestyle='-', linewidth='5', markersize=marker_size, label='Charge')
ax1.plot(x1, law_y1,  color='y', marker='x', linestyle='-.', linewidth='5', markersize=marker_size, label='Law')
ax1.plot(x1, term_y1, color='b', marker='x', linestyle='-.', linewidth='5', markersize=marker_size, label='Penalty Term')

ax1.set_xticks([0.1, 0.5, 1, 10, 20])
ax1.set_yticks([30, 50, 70, 90])
ax1.set_xlabel("Alpha Values", fontsize=35)
ax1.set_ylabel('F1 Value', fontsize=35)
#ax2.set_xlim([0, 550])

lines, labels = [], []
ax1_lines, ax1_labels = fig.axes[0].get_legend_handles_labels()

plt.legend(ax1_lines, ax1_labels, loc=(0.75, 0.03), fontsize=20)
plt.show()
plt.savefig('dev_sst.pdf', bbox_inches='tight', format='pdf')