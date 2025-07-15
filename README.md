# 🎯 SmartDartCorrector

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/joaoMartinSaquet)

A sophisticated reinforcement learning and genetic programming system for correcting user inputs in a smart dart game environment. This project implements multiple approaches to improve user performance through intelligent correction mechanisms.

## 🌟 Features

- **🤖 Multiple AI Approaches**: Reinforcement Learning (REINFORCE, PPO, SAC) and Genetic Programming (CGP)
- **🎮 Godot Game Integration**: Custom Godot environment with multiple game variants
- **🔧 Smart Correction System**: Intelligent user input correction with configurable perturbation
- **📊 Performance Analysis**: Comprehensive visualization and analysis tools
- **🎯 User Simulation**: VITE-based user simulator for realistic behavior modeling
- **⚙️ Hyperparameter Optimization**: Optuna integration for automated tuning

## 🏗️ Architecture

### 📂 Project Structure

```
SmartDartCorrector/
├── 🧠 classic_rl/                    # Classic Reinforcement Learning
│   ├── rl_corrector.py               # REINFORCE & PPO implementations
│   ├── policy.py                     # Neural network policies
│   ├── buffer.py                     # Experience replay buffers
│   └── PPO.py                        # PPO algorithm implementation
├── 🧬 GA/                           # Genetic Algorithm approaches
│   ├── cgp_corrector.py              # Cartesian Genetic Programming
│   └── pyCGP/                        # CGP framework
├── 🎮 games/                        # Godot game environments
│   ├── SmartDartEnvNormalized/       # Normalized environment
│   ├── SmartDartMultiEnv/            # Multi-environment setup
│   ├── SmartDartPlusDist/            # Enhanced with distance metrics
│   └── SmartDartSingleEnv/           # Single environment
├── 🔧 common/                       # Shared utilities
│   ├── corrector.py                  # Base correction classes
│   ├── user_simulator.py             # VITE user simulation
│   ├── perturbation.py               # Input perturbation systems
│   └── rolloutenv.py                 # Environment rollout utilities
├── 🏃 training_scripts/             # Training pipelines
│   ├── train_corrector.py            # Main training script
│   ├── train_ppo_sb3.py              # PPO with Stable Baselines3
│   ├── train_sac_sb3.py              # SAC with Stable Baselines3
│   └── train_cgp.py                  # CGP training
├── 📊 notebooks/                    # Analysis and visualization
│   └── viz.ipynb                     # Performance visualization
└── 🔨 usefull_scripts/              # Utility scripts
    └── run_u_sim.py                  # User simulation runner
```

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Godot Engine (for game environment)

### 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/joaoMartinSaquet/SmartDartCorrector.git
cd SmartDartCorrector
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install additional dependencies:**
```bash
pip install stable-baselines3[extra]
pip install optuna
pip install wandb
```

### 🎮 Game Environment Setup

The project includes pre-built Godot environments in the `games/` directory. Each environment variant offers different features:

- **SmartDartEnvNormalized**: Standard normalized environment
- **SmartDartMultiEnv**: Multiple parallel environments
- **SmartDartPlusDist**: Enhanced with distance calculations
- **SmartDartSingleEnv**: Single environment for debugging

## 🎯 Usage

### 🏃 Training a Corrector

**Basic Training:**
```bash
python training_scripts/train_corrector.py --method rl --perturbation_std 20.0
```

**With Perturbation:**
```bash
python training_scripts/train_corrector.py --method rl --perturbation_std 15.0
```

**Without Perturbation:**
```bash
python training_scripts/train_corrector.py --method rl --no_perturbation
```

**CGP Training:**
```bash
python training_scripts/train_corrector.py --method cgp --perturbation_std 10.0
```

### 🧠 Advanced Training with Stable Baselines3

**PPO Training:**
```bash
python training_scripts/train_ppo_sb3.py --timesteps 1000000 --n-envs 4
```

**SAC Training:**
```bash
python training_scripts/train_sac_sb3.py --timesteps 1000000 --n-envs 2
```

**With Hyperparameter Optimization:**
```bash
python training_scripts/train_ppo_sb3.py --timesteps 50000 --optuna-trials 50 --eval-episodes 10
```

### 🎮 Running User Simulation

```bash
python usefull_scripts/run_u_sim.py --perturbator Noise --N 100 --perturbation_std 20.0
```

### 📊 Training with Weights & Biases

```bash
python training_scripts/train_corrector.py --method rl --wandb
```

## 🔬 Methods & Algorithms

### 🧠 Reinforcement Learning

1. **REINFORCE**: Policy gradient method with baseline
   - Stochastic policy optimization
   - Configurable neural network architectures (MLP, LSTM, StackedMLP)
   - Gaussian action distribution

2. **PPO (Proximal Policy Optimization)**: 
   - Clipped surrogate objective
   - Adaptive KL penalty
   - Stable policy updates

3. **SAC (Soft Actor-Critic)**:
   - Off-policy learning
   - Entropy regularization
   - Continuous action spaces

### 🧬 Genetic Programming

**Cartesian Genetic Programming (CGP)**:
- Evolutionary correction function discovery
- Mathematical expression evolution
- Parallel population evaluation
- Configurable function libraries

### 🎯 User Simulation

**VITE Controller**:
- Biologically-inspired movement model
- Configurable dynamics parameters
- Realistic human-like behavior
- Adaptive target acquisition

### 🔧 Perturbation Systems

- **Normal Jittering**: Gaussian noise injection
- **Bias Perturbation**: Systematic offset introduction
- **Configurable Parameters**: Adjustable noise levels

## 📊 Performance Metrics

The system tracks multiple performance indicators:

- **Reward Accumulation**: Episode reward progression
- **Correction Accuracy**: Input correction effectiveness
- **Training Convergence**: Learning curve analysis
- **Robustness**: Performance under perturbations

## 🎛️ Configuration

### 🎯 Key Parameters

| Parameter | Description | Default |
|-----------|-------------|--------|
| `perturbation_std` | Standard deviation for noise injection | 20.0 |
| `hidden_size` | Neural network hidden layer size | 256 |
| `learning_rate` | Learning rate for optimization | 1e-4 |
| `n_episodes` | Number of training episodes | 50 |
| `batch_size` | Batch size for training | 64 |

### 🎮 Environment Variants

```python
# Environment selection
env_path = "games/SmartDartEnvNormalized/smartDartEnv.x86_64"
env = StableBaselinesGodotEnv(env_path=env_path, show_window=False, n_parallel=1)
```

## 📈 Results & Analysis

### 📊 Visualization

Use the provided Jupyter notebook for performance analysis:

```bash
jupyter notebook notebooks/viz.ipynb
```

### 📋 Logging

- **Weights & Biases**: Comprehensive experiment tracking
- **CSV Logs**: Detailed performance metrics
- **Model Checkpoints**: Save and load trained models

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PPO Implementation**: Based on [nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- **SAC Implementation**: Using Stable Baselines3
- **Godot RL**: Integration with Godot game engine
- **Research Community**: Various papers and implementations in RL and GP

## 📞 Contact

**João Martin Saquet** - [@joaoMartinSaquet](https://github.com/joaoMartinSaquet)

Project Link: [https://github.com/joaoMartinSaquet/SmartDartCorrector](https://github.com/joaoMartinSaquet/SmartDartCorrector)

---

<div align="center">
Made with ❤️ for advancing human-computer interaction through intelligent correction systems
</div>
