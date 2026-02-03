# nanoRL 

> **Reinforcement Learning, stripped down.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Educational-orange)](https://github.com/ritwikraha/nanoRL)

**nanoRL** is a minimalist, hackable, and educational implementation of modern Reinforcement Learning (RL) algorithms. Inspired by the "nano" philosophy (e.g., `nanoGPT`), this repository aims to make complex RL concepts accessible by providing clean, single-file implementations of state-of-the-art algorithms.

It is designed for students, researchers, and engineers who want to understand the "nuts and bolts" of RL without the overhead of massive frameworks.

##  Features

* **Minimalist Codebase**: Algorithms are implemented in as few lines as possible without sacrificing readability.
* **Educational**: Heavy commenting and clear variable naming to aid understanding.
* **Hackable**: No complex abstractions or config hell. Just pure code you can modify and extend.
* **Core Algorithms**: (Planned/Included)
    * Proximal Policy Optimization (PPO)
    * Deep Q-Networks (DQN)
    * REINFORCE / Vanilla Policy Gradient
* **Framework Agnostic Principles**: Focused on the logic of RL (Implementations likely in JAX/Keras or PyTorch).

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ritwikraha/nanoRL.git](https://github.com/ritwikraha/nanoRL.git)
    cd nanoRL
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

##  Usage

To train an agent on a standard environment (e.g., CartPole from Gymnasium):

```bash
python train.py --algo ppo --env CartPole-v1
```

##  Project Structure

```
nanoRL/
├── algorithms/       # Core algorithm implementations
│   ├── ppo.py        # Proximal Policy Optimization
│   ├── dqn.py        # Deep Q-Network
│   └── reinforce.py  # REINFORCE
├── envs/             # Environment wrappers (if any)
├── train.py          # Main training loop
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
## Philosophy
Code is Documentation: The best way to learn an algorithm is to implement it.

Simplicity > Completeness: We prioritize a clean implementation of the core idea over supporting every possible feature or edge case.

"Learning is Probabilistic": Experimentation is key.

## Contributing
Contributions are welcome! If you find a bug or want to add a "nano" implementation of a new algorithm:

- Fork the repo.
- Create a new branch.
- Submit a Pull Request.

Please ensure your code follows the philosophy of simplicity and readability.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
```
@misc{nanoRL2026,
  author = {Raha, Ritwik and Aritra Roy Gosthipaty},
  title = {nanoRL: A minimalist reinforcement learning library},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ritwikraha/nanoRL}},
  email = {ritwikraha.nsec@gmail.com, aritra.born2fly@gmail.com}
}
```
