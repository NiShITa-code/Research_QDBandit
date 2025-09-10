# Unified Multi-Armed Bandit Framework for Adversarial Prompt Generation

A comprehensive, modular framework implementing multiple bandit algorithms for automated red teaming of Large Language Models through quality-diversity optimization.

## Overview

This unified framework addresses the critical need for systematic comparison of multi-armed bandit algorithms in adversarial prompt generation. By providing a single, modular codebase that preserves the exact mathematical implementations of each algorithm while ensuring perfect reproducibility, researchers can conduct fair algorithmic comparisons without implementation bias.

## Key Features

### 🏗️ **Unified Architecture**
- **Single Interface**: All algorithms implement the same `BanditAlgorithm` abstract base class
- **Preserved Logic**: Every formula, update rule, and implementation detail remains mathematically identical
- **Fair Comparison**: Identical experiment structure, logging, and analysis across all algorithms
- **Easy Switching**: Change algorithm with just `--algorithm thompson_sampling` vs `--algorithm ucb`

### 🎯 **Algorithm Coverage**
- **Thompson Sampling**: Bayesian approach with Normal-Normal conjugacy
- **MOSS**: Minimax optimal strategy with logarithmic confidence bounds  
- **UCB**: Upper confidence bound with exploration bonuses
- **Epsilon-Greedy**: Probabilistic exploration with optional decay
- **Random**: Uniform random baseline for comparison

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd unified-bandit-framework

# Install dependencies
pip install -r requirements.txt

# Ensure you have the required modules
pip install numpy scipy matplotlib seaborn
```

## Quick Start

### Single Algorithm Experiment

```bash
# Thompson Sampling with Bayesian updates
python unified_bandit.py --algorithm thompson_sampling --seed 42 --max_iters 500

# MOSS with minimax optimal bounds
python unified_bandit.py --algorithm moss --seed 42 --max_iters 500

# UCB with custom exploration parameter
python unified_bandit.py --algorithm ucb --ucb_c 1.0 --seed 42 --max_iters 500

# Epsilon-Greedy with decay
python unified_bandit.py --algorithm epsilon_greedy --epsilon 0.2 --epsilon_decay 0.001 --seed 42

# Random baseline
python unified_bandit.py --algorithm random --seed 42 --max_iters 500
```

### Comparative Study

```bash
# Run all algorithms with identical conditions
for alg in thompson_sampling moss ucb epsilon_greedy random; do
  python unified_bandit.py --algorithm $alg --seed 42 --max_iters 500 --fitness_threshold 0.6
done

# Or use the built-in comparative study
python unified_bandit.py --comparative-study
```

## Configuration

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--algorithm` | Bandit algorithm to use | `thompson_sampling` |
| `--seed` | Random seed for reproducibility | `42` |
| `--max_iters` | Maximum number of iterations | `1000` |
| `--num_mutations` | Prompts generated per iteration | `10` |
| `--fitness_threshold` | Archive inclusion threshold | `0.6` |

### Algorithm-Specific Parameters

#### Thompson Sampling
- `--ts_prior_mean`: Prior mean (default: 0.5)
- `--ts_prior_precision`: Prior precision (default: 1.0)  
- `--ts_noise_precision`: Noise precision (default: 4.0)

#### UCB
- `--ucb_c`: Exploration constant (default: 0.7)

#### Epsilon-Greedy
- `--epsilon`: Exploration probability (default: 0.1)
- `--epsilon_decay`: Decay rate per iteration (default: 0.0)
- `--min_epsilon`: Minimum epsilon value (default: 0.01)

## Architecture

### Class Hierarchy

```
BanditAlgorithm (Abstract Base Class)
├── ThompsonSamplingAlgorithm
├── MOSSAlgorithm  
├── UCBAlgorithm
├── EpsilonGreedyAlgorithm
└── RandomAlgorithm

BanditArm (Base Class)
├── ThompsonSamplingArm
├── MOSSArm
├── UCBArm
├── EpsilonGreedyArm
└── RandomArm
```

### Key Components

1. **Unified Interface**: All algorithms implement the same `select_arm()` method
2. **Individual Updates**: Each prompt-level fitness score updates arm statistics immediately
3. **Entropy Tracking**: Consistent selection entropy calculation across all algorithms
4. **Checkpoint Management**: Complete experiment state preservation and recovery

## Mathematical Formulations

### Thompson Sampling
Bayesian updates using Normal-Normal conjugacy:
```
τ₁ = τ₀ + τ (precision update)
μ₁ = (τ₀μ₀ + τx) / τ₁ (mean update)
```

### UCB (Upper Confidence Bound)
```
UCB_t(i) = μ̂ᵢ,t + C√(ln(t)/nᵢ,t)
```

### MOSS (Minimax Optimal Strategy)
```
MOSS_t(i) = μ̂ᵢ,t + √(max(0, ln(T/(K·nᵢ,t)))/nᵢ,t)
```

### Epsilon-Greedy
```
Action = {
  argmax μ̂ᵢ,t  with probability 1-ε
  random       with probability ε
}
```

### Selection Entropy
```
H = -1/log(Nc) * Σ[nc(i)/Ns * log(nc(i)/Ns)]
```

## Output Structure

```
logs/unified_bandit/
├── thompson_sampling/
│   └── model_name/dataset/seed_42/
│       ├── plots/
│       ├── enhanced_analysis_thompson_sampling_seed_42.json
│       └── experiment_seed_42.txt
├── moss/
├── ucb/
├── epsilon_greedy/
└── random/
```

To reproduce results:
```bash
python unified_bandit.py --algorithm thompson_sampling --seed 42
```

### Checkpointing and Recovery

```bash
# Auto-resume from latest checkpoint
python unified_bandit.py --algorithm moss --auto_resume

# Resume from specific checkpoint
python unified_bandit.py --algorithm ucb \
  --resume_from_checkpoint ./checkpoints/ucb_experiment_iter_000500.pkl
```

## Performance Analysis

The framework provides comprehensive analysis including:

- **Selection Entropy**: Measure of exploration vs exploitation balance
- **Coverage Metrics**: Fraction of descriptor combinations explored
- **Algorithm-Specific Metrics**: Posterior statistics, confidence bounds, etc.
- **Quality-Diversity Scores**: Archive performance and diversity measures

## Contributing

When adding new algorithms:

1. Inherit from `BanditAlgorithm` base class
2. Implement required methods: `get_name()`, `get_description()`, `create_arms()`, `select_arm()`
3. Create corresponding arm class inheriting from `BanditArm`
4. Add algorithm type to `AlgorithmType` enum
5. Update `create_algorithm()` factory function


## License

This project is licensed under the MIT License - see the LICENSE file for details.


---

*This framework enables rigorous scientific comparison of multi-armed bandit algorithms for adversarial prompt generation while maintaining the highest standards of reproducibility and mathematical precision.*
