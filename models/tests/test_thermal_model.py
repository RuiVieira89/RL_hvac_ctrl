
import sys
sys.path.append('/home/rui/dev/RL_hvac_ctrl/models/')
sys.path.append('/home/rui/dev/RL_hvac_ctrl/models/Thermal_model.py')
from models.thermal_model import Thermal_model

import matplotlib.pyplot as plt

# import itertools
# import warnings

import numpy as np


def main():
    # Analysis

    test = Thermal_model()
    t = - 1

    dew_plot = []
    inside_Temp_plot = []
    E_balance_plot = []

    mass_updt = 0.01

    for _ in test.solar:
        t = t + 1
        dew_point_confort_ = test.dew_point_confort()
        thermal_balance_plot = test.thermal_balance_update(t, mass_updt, 1000)
        print(f'===== time={t} ======\n')
        print(f'energy balance: {thermal_balance_plot}')
        print(f'confort level: {dew_point_confort_}\n')
        print(f'solar intensity = {test.solar[t]}')
        print(f'Dew Point = {test.dew_point}')
        print(f'inside temperature = {test.temperature_inside}')
        print('\n')

        # info to print
        dew_plot = np.append(dew_plot, test.dew_point)
        inside_Temp_plot = np.append(inside_Temp_plot, test.temperature_inside)
        E_balance_plot = np.append(E_balance_plot, thermal_balance_plot)

        if inside_Temp_plot[t] > test.temperature_outside[t]:
            mass_updt = mass_updt + 0.01
        elif mass_updt >= 0.01:
            mass_updt = mass_updt - 0.01

        if test.temperature_inside > 32:
            break
        elif test.temperature_inside < 0:
            break

    fig, ax = plt.subplots()

    ax.plot(dew_plot)
    ax.plot(inside_Temp_plot)
    ax.plot(test.temperature_outside)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('Temperature [ºC]')
    ax.set_title('Temperatures')
    ax.legend(['TDew', 'Tcabin', 'Toutside'], loc='best')

    ax2 = ax.twinx()
    ax2.plot(test.humidity_inside, 'r-')
    ax2.set_ylabel('Humidity (in red)')

    fig.tight_layout()

    fig1, ax1 = plt.subplots()
    ax1.plot(E_balance_plot)
    ax1.plot(np.zeros(E_balance_plot.shape))
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('E balance [W]')
    ax1.set_title('Energy bal.')
    ax1.legend(['E bal.'], loc='best')

    fig1.tight_layout()

    plt.show()


'''
    This is for testing only
'''
if __name__ == "__main__":

    main()
