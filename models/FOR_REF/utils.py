
import os
import time
import itertools
import shutil
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt

from tensorforce import Agent

from thermal_model import Thermal_model
from envir import HvacEnv

def update_dict(rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep, name, results_dict):
    """
    Updates a dictionary with the avergaes values of each episode.
    """

    results_dict[name]['rewards_avg_ep'] =np.append(results_dict[name]['rewards_avg_ep'], rewards_avg_ep)
    results_dict[name]['newMass_avg_ep'] =np.append(results_dict[name]['newMass_avg_ep'], newMass_avg_ep)
    results_dict[name]['acBalance_avg_ep'] =np.append(results_dict[name]['acBalance_avg_ep'], acBalance_avg_ep)
    results_dict[name]['thermalBalance_avg_ep'] =np.append(results_dict[name]['thermalBalance_avg_ep'], thermalBalance_avg_ep)

    return results_dict


def run(environment, agent, n_episodes, max_step_per_episode, combination, test=False):
    """
    Train agent for n_episodes

    There are two modes, training mode (where exploration is allowed), used for training 
    the agent over a large number of episodes; 
    and test mode (where exploration is not allowed) used to evaluate the agent’s performance.
    """
    
    rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep = (np.array([]) for i in range(4))

    #environment.FlightModel.max_step_per_episode = max_step_per_episode
    # Loop over episodes
    for i in range(n_episodes):
        # Initialize episode
        episode_length, rewards_ep, new_mass, ac_balance, thermalBalance = (0 for i in range(5))
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False

        while not terminal:  # While an episode has not yet terminated
            if test:  # Test mode (deterministic, no exploration)
                episode_length += 1
                actions, internals = agent.act(
                    states=states, internals=internals, evaluation=True
                )
                states, terminal, reward = environment.execute(actions=actions)
                if episode_length > max_step_per_episode:
                    terminal = True
                
                # increment values
                rewards_ep += reward
                new_mass += actions['new_mass'][0]
                ac_balance += actions['ac_balance'][0]
                thermalBalance += states[0]

            else:  # Train mode (exploration and randomness)
                episode_length += 1
                actions = agent.act(states=states)
                states, terminal, reward = environment.execute(actions=actions)
                agent.observe(terminal=terminal, reward=reward)
                
                rewards_ep += reward

                if episode_length > max_step_per_episode:
                    terminal = True

                # increment values
                rewards_ep += reward
                new_mass += actions['new_mass'][0]
                ac_balance += actions['ac_balance'][0]
                thermalBalance += states[0]
        
        # increment arrays
        rewards_avg_ep = np.append(rewards_avg_ep, rewards_ep / episode_length)
        newMass_avg_ep = np.append(newMass_avg_ep, new_mass / episode_length)
        acBalance_avg_ep = np.append(acBalance_avg_ep, ac_balance / episode_length)
        thermalBalance_avg_ep = np.append(thermalBalance_avg_ep, thermalBalance / episode_length)
        
    return rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep


def runner(
    environment,
    agent,
    max_step_per_episode,
    n_episodes,
    n_episodes_test=1,
    combination=1,
):
    # Train agent
    results = {
        'train': {
            'rewards_avg_ep': np.array([]),
            'newMass_avg_ep': np.array([]),
            'acBalance_avg_ep': np.array([]),
            'thermalBalance_avg_ep': np.array([]) 
        },
        'test': {
            'rewards_avg_ep': np.array([]),
            'newMass_avg_ep': np.array([]),
            'acBalance_avg_ep': np.array([]),
            'thermalBalance_avg_ep': np.array([]) 
        }       
    }

    for i in range(round(n_episodes / 100)): #Divide the number of episodes into batches of 100 episodes
        print("=====================================================")
        print(f" Batch #: {i} / {round(n_episodes / 100)}")
        print("=====================================================")
        # Train Agent for 100 episode
        rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep = run(environment, agent, 100, max_step_per_episode, combination=combination)
        results = update_dict(rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep,'train', results) 

        # Test Agent for this batch
        rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep = run(
            environment,
            agent,
            n_episodes_test,
            max_step_per_episode,
            combination=combination,
            test=True
        )
        results = update_dict(rewards_avg_ep, newMass_avg_ep, acBalance_avg_ep, thermalBalance_avg_ep,'test', results)

    # Plot the evolution of the agent over the batches
    sets_ = ['train', 'test']
    for set_ in sets_:
        for keys in results[set_]:
            name = keys.split('_')[0]
            fig, ax = plt.subplots()
            ax.plot(results[set_][keys], label=name, color='#FF0000')
            ax.set_xlabel('Episodes')
            ax.set_ylabel(f'{set_} - Avg {name} per episode')
            ax.set_title(f'{set_} - Avg {name} per episode')
            plt.savefig(f"results/Graphs/{set_}-Avg_{name}_per_episode")

    #Terminate the agent and the environment
    agent.close()
    environment.close()

# Instantiate a Tensorforce agent
def create_agent(environment):
    return Agent.create(
        agent='ppo', environment=environment,
        # Automatically configured network
        network='auto',
        # Optimization
        batch_size=10, update_frequency=2, learning_rate=1e-3, subsampling_fraction=0.2,
        optimization_steps=5,
        # Reward estimation
        likelihood_ratio_clipping=0.2, discount=0.99, estimate_terminal=False,
        # Critic
        critic_network='auto',
        critic_optimizer=dict(optimizer='adam', multi_step=10, learning_rate=1e-3),
        # Preprocessing
        preprocessing=None,
        # Exploration
        exploration=0.0, variable_noise=0.0,
        # Regularization
        l2_regularization=0.0, entropy_regularization=0.0,
        # TensorFlow etc
        name='agent', device=None, parallel_interactions=1, seed=None, execution=None, saver=None,
        summarizer=None, recorder=None
)


# Instantiate our Thermal Model
# ThermalModel = Thermal_model()
# # Call runner
# runner(
#     environment,
#     agent,
#     max_step_per_episode=1000,
#     n_episodes=200)