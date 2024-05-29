import matplotlib.pyplot as plt
import numpy as np
import json

with open('FovVideoVDP_flikcer_JOD_results.json', 'r') as fp:
    all_quality_data = json.load(fp)

quality_array = np.array(all_quality_data['quality'])
radius_list = all_quality_data['radius_list']
frr_list = all_quality_data['frr_list']

plt.figure(figsize=(8,9))
for radius_index in range(len(radius_list)):
    radius_value = radius_list[radius_index]
    plt.subplot(len(radius_list), 1, radius_index + 1)
    quality_array_size = 10-quality_array[radius_index]
    plt.plot(frr_list, quality_array_size, '-o', linewidth=2)
    plt.xlim([0.4, 18])
    plt.ylim([0, 0.1])
    # plt.xscale('log')
    plt.yscale('log')
    plt.xticks(frr_list, [str(i) for i in frr_list])
    plt.yticks([0.0001, 0.001, 0.01, 0.1, 1])
    plt.xlim([0.25, 16])
    plt.ylim([0.0001, 1])
    plt.ylabel('10 - JOD Quality')
    plt.title(f'Radius {radius_value}')
    plt.grid(True)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=5)
plt.xlabel('VRR Frequency')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4)
# plt.tight_layout()
plt.show()