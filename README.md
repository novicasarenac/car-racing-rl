# car-racing-rl

Implementation of Reinforcement Learning algorithms in [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment. Implemented algorithms:
* Deep Q-Network (DQN)
* Advantage Actor Critic (A2C)
* Asynchronous Advantage Actor Critic (A3C)

## Setup and running ##

Requirements: `python 3.6`

To install all required dependencies run:
```bash
pip install -r requirements.txt
```

Start training / inference / evaluation with:
```bash
python -m run --<action> -m=<model>
```
Possible values for parameter `action` are: `train`, `inference` and `evaluate`.

Possible values for parameter `model` are: `dqn`, `a2c` and `a3c`.

Hyperparameters can be changed in .json files in `/params` directory.