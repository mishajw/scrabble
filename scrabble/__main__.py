from scrabble.scrabble_environment import ScrabbleEnvironment
import tensorforce.agents
import tensorforce.execution


def main():
    environment = ScrabbleEnvironment(5)

    agent = tensorforce.agents.PPOAgent(
        states_spec=environment.states,
        actions_spec=environment.actions,
        network_spec=[
            dict(type='dense', size=64),
            dict(type='dense', size=64)],
        batch_size=1000,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4))

    runner = tensorforce.execution.Runner(agent=agent, environment=environment)
    runner.run()


if __name__ == "__main__":
    main()
