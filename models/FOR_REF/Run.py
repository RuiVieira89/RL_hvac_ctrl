
import numpy as np

from thermal_model import Thermal_model

test = Thermal_model()

t = -1
for _ in test.solar:
    t = t + 1
    dew_point_confort_ = test.dew_point_confort()
    thermal_balance_plot = test.thermal_balance_update(t,0.01)
    print(f'===== time={t} ======\n')
    print(f'energy balance: {thermal_balance_plot}')
    print(f'confort level: {dew_point_confort_}\n')
    print(f'solar intensity = {test.solar[t]}')
    print(f'Dew Point = {test.dew_point}')
    print(f'inside temperature = {test.temperature_inside}')
    print('\n')

