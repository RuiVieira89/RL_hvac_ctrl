from thermal_model import Thermal_model
from envir import HvacEnv
import utils as utls


def main():
    # Instantiane our environment and agent
    environment = HvacEnv()
    agent = utls.create_agent(environment)

    utls.runner(
        environment,
        agent,
        max_step_per_episode=1000,
        n_episodes=1000)


if __name__ == "__main__":
    main()
