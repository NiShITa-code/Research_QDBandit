
# QD-Bandit: Quality-Diversity Selection as Multi-Armed Bandit Problem for Adversarial Prompt Generation

QD-Bandit addresses a fundamental inefficiency in current quality-diversity approaches for automated adversarial prompt generation: the reliance on heuristic descriptor selection strategies that fail to learn from evaluation history. By formulating descriptor combination selection as a multi-armed bandit problem, QD-Bandit replaces uniform random sampling with principled exploration-exploitation strategies that adaptively focus computational resources on promising attack vectors while maintaining theoretical guarantees for comprehensive vulnerability discovery.

## Research Contributions

### 1. Novel Problem Formulation
- **First** to model descriptor selection in quality-diversity red teaming as a multi-armed bandit problem
- Theoretical framework bridging adversarial ML security and online learning theory
- Stochastic environment assumptions enabling bandit algorithm guarantees

### 2. Individual Update Mechanism
- **Key Innovation**: Prompt-level bandit updates vs traditional batch aggregation
- 10x increase in learning signals per iteration compared to existing methods
- Enhanced sample efficiency through immediate feedback incorporation
- Granular performance assessment within descriptor combinations

### 3. Comprehensive Algorithmic Analysis
- Systematic comparison of Thompson Sampling, MOSS, UCB, and ε-greedy strategies
- Preservation of exact mathematical implementations for fair evaluation
- Selection entropy analysis for exploration-exploitation characterization
- Regret bound validation in adversarial prompt generation context

### 4. Empirical Validation Framework
- Perfect reproducibility through comprehensive seed management
- Standardized experimental protocols enabling cross-algorithm comparison
- Quality-diversity metrics adapted for adversarial prompt evaluation
- Statistical significance testing across multiple experimental conditions



## QD-Bandit integrates four representative bandit algorithms spanning different theoretical paradigms:

#### Thompson Sampling (Bayesian)
Maintains posterior distributions over arm rewards using Normal-Normal conjugacy:

```
Posterior Update: τ₁ = τ₀ + τ, μ₁ = (τ₀μ₀ + τx)/τ₁
Selection: θᵢ ~ N(μᵢ, 1/τᵢ), a* = argmax θᵢ
```

#### MOSS (Minimax Optimal)
Horizon-aware confidence bounds achieving minimax optimal regret:

```
MOSS_t(i) = μ̂ᵢ,t + √(max(0, ln(T/(K·nᵢ,t)))/nᵢ,t)
```

#### UCB (Upper Confidence Bound)
Optimism under uncertainty with tunable exploration:

```
UCB_t(i) = μ̂ᵢ,t + C√(ln(t)/nᵢ,t)
```

#### ε-Greedy (Probabilistic)
Simple exploration with optional decay schedule:

```
a_t = {argmax μ̂ᵢ,t with prob 1-ε, random with prob ε}
```

### Evaluation Metrics
- **Attack Success Rate**:
- **Selection Entropy**: H = -1/log(K) Σᵢ pᵢlog(pᵢ) quantifying exploration
- **Archive Coverage**: |{a ∈ A : selected(a)}|/|A| measuring diversity
- **Quality-Diversity Score**: Σ fitness(prompts) measuring aggregate performance


### Experimental Conditions

#### Datasets
- **HarmBench**: 400 prompts across 18 harm categories
- **AdversarialQA**: 500 adversarially crafted instructions

#### Target Models
- **Instruction-tuned LLMs**: Qwen2.5-7B, Llama-2-7B, Ministral-8B


## Usage

### Installation
```bash
git clone https://github.com/NiShITa-code/Research_QDBandit
cd qd-bandit
pip install -e .
```

### Basic Experiment
```bash
# Run QD-Bandit with Thompson Sampling
python -m QDBandit.run_ts \
    --max_iters 1000 \
    --dataset data/harmbench.json \
    --target_llm "Qwen/Qwen2.5-7B-Instruct"

## Related Work

### Quality-Diversity Optimization
- **MAP-Elites** [Mouret & Clune, 2015]: Foundational QD framework
- **CVT-MAP-Elites** [Vassiliades et al., 2018]: Centroidal Voronoi tessellations
- **QD-PG** [Pierrot et al., 2022]: Policy gradient methods for QD

### Adversarial Prompt Generation  
- **Rainbow Teaming** [Davis et al., 2023]: QD for LLM red teaming
- **RainbowPlus** [Dang et al., 2024]: Enhanced descriptor taxonomies
- **GPTFuzz** [Yu et al., 2023]: Fuzzing-based prompt generation

### Multi-Armed Bandits
- **UCB1** [Auer et al., 2002]: Upper confidence bound algorithm
- **Thompson Sampling** [Thompson, 1933; Agrawal & Goyal, 2012]: Bayesian approach
- **MOSS** [Audibert & Bubeck, 2009]: Minimax optimal strategy

## Limitations and Future Work

### Current Limitations
- **Static Descriptor Taxonomies**: Fixed descriptor sets throughout experiments
- **Stationarity Assumption**: May not hold for adaptive target models
- **Limited Contextual Information**: No incorporation of prompt features
- **Evaluation Bottleneck**: Safety classifier latency limits throughput

### Future Directions
- **Dynamic Descriptor Discovery**: Online learning of new descriptor categories
- **Non-stationary Bandits**: Adaptation to evolving model defenses
- **Contextual QD-Bandit**: Feature-based arm selection strategies
- **Multi-objective Optimization**: Simultaneous quality and diversity optimization
- **Adversarial Robustness**: Defense against adaptive evaluation targets
