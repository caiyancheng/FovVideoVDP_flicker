import matplotlib.pyplot as plt
import numpy as np
import json
import math

with open('E:\Py_codes\FovVideoVDP_flicker\VRR_Flicker/FovVideoVDP_flikcer_JOD_results_L_peak_10_foveated_min_freq_01.json', 'r') as fp:
    all_quality_data = json.load(fp)

quality_array = np.array(all_quality_data['quality'])
size_list = all_quality_data['size_list']
area_list = []
for size_value in size_list:
    if size_value == 'full':
        area_list.append(62.666 * 37.808)
    else:
        area_list.append(math.pi * (size_value/2)**2)

vrr_f_list = all_quality_data['vrr_f_list']

plt.figure(figsize=(8,9))
for vrr_f_index in range(len(vrr_f_list)):
    vrr_f_value = vrr_f_list[vrr_f_index]
    plt.subplot(int(len(vrr_f_list)/2), 2, vrr_f_index + 1)
    quality_array_vrr_f = 10-quality_array[:,vrr_f_index]
    plt.plot(area_list, quality_array_vrr_f, '-o', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(area_list, ["{:.2f}".format(area) for area in area_list])
    plt.yticks([0.0001, 0.001, 0.01, 0.1])
    plt.xlim([min(area_list)*0.9, max(area_list)*1.1])
    plt.ylim([0.0001, 0.1])
    plt.ylabel('10 - JOD Quality')
    plt.title(f'Frequency of RR Switch {vrr_f_value}')
    plt.grid(True)
    if vrr_f_index > len(vrr_f_list) - 3:
        plt.xlabel('Area')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4)
# plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=5)


# plt.tight_layout()
plt.show()