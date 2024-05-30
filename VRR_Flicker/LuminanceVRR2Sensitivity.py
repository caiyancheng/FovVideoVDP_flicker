import json
import numpy as np
import os

class LuminanceVRR2Sensitivity:
    def __init__(self):
        json_file_path = 'Luminance_FRR_to_Sensitivity.json'
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
            coefficients = json_data['coefficients_1d']
        self.coefficients = coefficients['Coefficients']
        self.model_terms = coefficients['ModelTerms']

    def polyvaln(self, x1, x2):
        result = 0
        for model_term_index in range(len(self.model_terms)):
            model_term_value = self.model_terms[model_term_index]
            result += self.coefficients[model_term_index] * (x1 ** model_term_value[0]) * (x2 ** model_term_value[1])
        return result

    def LT2S(self, luminance, t_frequency):
        log10_luminance = np.log10(luminance)
        sensitivity = 10 ** self.polyvaln(log10_luminance, t_frequency)
        return sensitivity

    def LT2S_log(self, log10_luminance, t_frequency):
        log10_sensitivity = self.polyvaln(log10_luminance, t_frequency)
        return log10_sensitivity


if __name__ == '__main__':
    luminance_vrr_2_sensitivity = LuminanceVRR2Sensitivity()
    luminance = 0.5  # example value
    t_frequency = 5.0  # example value
    sensitivity = luminance_vrr_2_sensitivity.LT2S(luminance, t_frequency)
    contrast = 1 / sensitivity
    print(contrast)