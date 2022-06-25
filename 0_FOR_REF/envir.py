import numpy as np
from tensorforce.environments import Environment
from thermal_model import Thermal_model


class HvacEnv(Environment):
    """
    HVAC RL Environment.
    """

    def __init__(self):
        super().__init__()
        self.Thermal_model = Thermal_model()
        self.NUM_ACTIONS = 2
        self.NUM_NEW_MASS = 1
        self.NUM_AC_BALANCE = 1
        self.max_step_per_episode = 1000
        self.finished = False
        self.episode_end = False
        self.STATES_SIZE = 1

    def states(self):
        """
        Defines the states of the environment.

        Returns:
            dictionary with
        """

        return dict(type="float", shape=(self.STATES_SIZE,))

    def actions(self):
        """
        Defines the actions allowed to do in the environment.
        """

        return {
            "new_mass": dict(type="float", shape=(self.NUM_NEW_MASS,),
                             min_value=18.0/(60*60), max_value=200.0/(60*60)),
            "ac_balance": dict(type="float", shape=(self.NUM_AC_BALANCE,),
                               min_value=-10.0, max_value=10.0),
        }

    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        return self.max_step_per_episode

    def reset(self):
        """
        Resets the environment to its initial state.
        """

        state = np.zeros(shape=(self.STATES_SIZE,))
        self.Thermal_model = Thermal_model()
        return state

    def execute(self, actions):
        """
        """

        reward = 0
        nb_timesteps = 1

        for i in range(1, nb_timesteps + 1):
            next_state = [self.Thermal_model.thermal_balance_update(
                nb_timesteps, actions['new_mass'][0],
                actions['ac_balance'][0])]
            reward += self.reward()

            if self.terminal():
                reward = reward / i
                break

        if i == nb_timesteps:
            reward = reward / nb_timesteps

        return next_state, self.terminal(), reward

    def terminal(self):
        """
        Defines in which conditions an episode will be finished.
        """

        self.finished = self.Thermal_model.dew_point == 17
        # NOT SURE IF IT'S NECESSARY!!! => DEFINE WHEN COMES TO AN END
        self.episode_end = (
            self.Thermal_model.Time_Step > self.max_step_per_episode
            ) or (self.Thermal_model.temperature_inside > 30)

        return self.finished or self.episode_end

    def reward(self):
        """
        Defines which rewards it's possible to receive in the environment.

        1 if confort == 2
        -1 if confort == 1
        -100 otherwise
        """

        if self.finished:
            reward = 100000
        else:
            confort = self.Thermal_model.dew_point_confort()
            reward = 1 if (confort == 2) else -1 if (confort == 1) else -100

        return reward
