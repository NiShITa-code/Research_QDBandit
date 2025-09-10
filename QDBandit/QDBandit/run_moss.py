import sys
import argparse
import random
import json
import time
import logging
import math
import pickle
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
import traceback
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

from QDBandit.scores import BleuScoreNLTK, LlamaGuard
from QDBandit.utils import (
    load_txt,
    load_json,
    initialize_language_models,
    save_iteration_log,
)
from QDBandit.archive import Archive
from QDBandit.configs import ConfigurationLoader
from QDBandit.prompts import MUTATOR_PROMPT, TARGET_PROMPT

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=None):
    """Set random seeds for reproducibility across all libraries."""
    if seed is None:
        seed = 42  # Default seed
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # For matplotlib reproducibility (if using random colors/styles)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10(np.linspace(0, 1, 10)))
    
    logger.info(f"Random seeds set to {seed} for reproducibility")
    logger.info("   - Python random module: ✓")
    logger.info("   - NumPy random: ✓")
    logger.info("   - Matplotlib: ✓")
    
    return seed

class SelectionEntropyTracker:
    """
    Track selection entropy according to formula.
    
    Selection Entropy = -1/log(Nc) * Σ(i=1 to Nc) [nc(i)/Ns * log(nc(i)/Ns)]
    
    Where:
    - Nc = total number of cells/actions/arms
    - nc(i) = number of times cell/action i was selected  
    - Ns = total number of selections
    """
    
    def __init__(self, num_arms, seed=None):
        self.num_arms = num_arms
        self.selection_counts = defaultdict(int)
        self.total_selections = 0
        self.entropy_history = []
        self.selection_history = []
        self.normalization_factor = 1.0 / np.log(self.num_arms) if self.num_arms > 1 else 1.0
        self.seed = seed
        
        logger.info(f"Initialized SelectionEntropyTracker for {num_arms} arms")
        logger.info(f"Normalization factor: 1/log({num_arms}) = {self.normalization_factor:.4f}")
        if seed is not None:
            logger.info(f"Seed: {seed}")
    
    def update_selection(self, arm_id):
        self.selection_counts[arm_id] += 1
        self.total_selections += 1
        self.selection_history.append(arm_id)
        current_entropy = self.calculate_selection_entropy()
        self.entropy_history.append(current_entropy)
    
    def calculate_selection_entropy(self):
        if self.total_selections == 0:
            return 0.0
        shannon_entropy = 0.0
        for arm_id in range(self.num_arms):
            nc_i = self.selection_counts[arm_id]
            if nc_i > 0:
                p_i = nc_i / self.total_selections
                shannon_entropy -= p_i * np.log(p_i)
        normalized_entropy = self.normalization_factor * shannon_entropy
        return normalized_entropy
    
    def get_max_entropy(self):
        return 1.0
    
    def get_selection_distribution(self):
        if self.total_selections == 0:
            return np.ones(self.num_arms) / self.num_arms
        distribution = np.zeros(self.num_arms)
        for arm_id in range(self.num_arms):
            distribution[arm_id] = self.selection_counts[arm_id] / self.total_selections
        return distribution
    
    def get_diversity_metrics(self):
        current_entropy = self.calculate_selection_entropy()
        num_selected_arms = sum(1 for count in self.selection_counts.values() if count > 0)
        selection_counts_list = [self.selection_counts[i] for i in range(self.num_arms)]
        return {
            'selection_entropy': current_entropy,
            'max_entropy': 1.0,
            'total_selections': self.total_selections,
            'num_selected_arms': num_selected_arms,
            'coverage_rate': num_selected_arms / self.num_arms,
            'most_selected_count': max(selection_counts_list) if selection_counts_list else 0,
            'selection_counts': selection_counts_list,
            'selection_std': np.std(selection_counts_list),
            'uniformity_score': 1.0 - current_entropy,
            'gini_coefficient': self.calculate_gini_coefficient()
        }
    
    def calculate_gini_coefficient(self):
        if self.total_selections == 0:
            return 0.0
        counts = [self.selection_counts[i] for i in range(self.num_arms)]
        counts = sorted(counts)
        n = len(counts)
        cumulative_sum = sum((i + 1) * count for i, count in enumerate(counts))
        total_sum = sum(counts)
        if total_sum == 0:
            return 0.0
        gini = (2 * cumulative_sum) / (n * total_sum) - (n + 1) / n
        return gini
    
    def get_recent_entropy(self, window_size=100):
        if len(self.selection_history) < window_size:
            return self.calculate_selection_entropy()
        recent_selections = self.selection_history[-window_size:]
        recent_counts = defaultdict(int)
        for arm_id in recent_selections:
            recent_counts[arm_id] += 1
        entropy = 0.0
        for arm_id in range(self.num_arms):
            count = recent_counts[arm_id]
            if count > 0:
                probability = count / window_size
                entropy -= probability * np.log(probability)
        return self.normalization_factor * entropy
    
    def analyze_exploration_exploitation(self):
        if len(self.entropy_history) < 10:
            return {'status': 'insufficient_data'}
        total_length = len(self.entropy_history)
        early_phase = self.entropy_history[:total_length//3]
        middle_phase = self.entropy_history[total_length//3:2*total_length//3]
        late_phase = self.entropy_history[2*total_length//3:]
        analysis = {
            'early_entropy_avg': np.mean(early_phase),
            'middle_entropy_avg': np.mean(middle_phase),
            'late_entropy_avg': np.mean(late_phase),
            'entropy_trend': 'decreasing' if np.mean(late_phase) < np.mean(early_phase) else 'increasing',
            'total_entropy_change': self.entropy_history[-1] - self.entropy_history[0],
            'convergence_rate': (np.mean(early_phase) - np.mean(late_phase)) / len(self.entropy_history)
        }
        return analysis
    
    def get_state_dict(self):
        return {
            'num_arms': self.num_arms,
            'selection_counts': dict(self.selection_counts),
            'total_selections': self.total_selections,
            'entropy_history': self.entropy_history,
            'selection_history': self.selection_history,
            'normalization_factor': self.normalization_factor,
            'seed': self.seed
        }
    
    def load_state_dict(self, state_dict):
        self.num_arms = state_dict['num_arms']
        self.selection_counts = defaultdict(int, state_dict['selection_counts'])
        self.total_selections = state_dict['total_selections']
        self.entropy_history = state_dict['entropy_history']
        self.selection_history = state_dict['selection_history']
        self.normalization_factor = state_dict['normalization_factor']
        self.seed = state_dict.get('seed', None)

def parse_arguments():
    parser = argparse.ArgumentParser(description="MOSS with Individual Updates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_samples", type=int, default=150, help="Number of initial seed prompts")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum number of iteration steps")
    parser.add_argument("--sim_threshold", type=float, default=0.6, help="Similarity threshold for prompt mutation")
    parser.add_argument("--num_mutations", type=int, default=10, help="Number of prompt mutations per iteration")
    parser.add_argument("--fitness_threshold", type=float, default=0.6, help="Minimum fitness score to add prompt to archive")
    parser.add_argument("--config_file", type=str, default="./configs/base-opensource.yml", help="Path to configuration file")
    parser.add_argument("--log_dir", type=str, default="./logs/final/moss_individual", help="Directory for storing logs")
    parser.add_argument("--log_interval", type=int, default=50, help="Number of iterations between log saves")
    parser.add_argument("--dataset", type=str, default="./data/harmbench.json", help="Dataset name")
    parser.add_argument("--target_llm", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Path to repository of target LLM")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle seed prompts")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/moss_individual", help="Directory for storing checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Number of iterations between checkpoint saves")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint if available")
    parser.add_argument("--keep_n_checkpoints", type=int, default=5, help="Number of recent checkpoints to keep")
    parser.add_argument("--negative_reward", type=float, default=-0.1, help="Negative reward for failed prompts")
    parser.add_argument("--eval_failure_reward", type=float, default=-0.05, help="Negative reward for evaluation failures")
    return parser.parse_args()

def load_descriptors(config):
    descriptors = {}
    for path, descriptor in zip(config.archive["path"], config.archive["descriptor"]):
        try:
            descriptors[descriptor] = load_txt(path)
            logger.info(f"Loaded {len(descriptors[descriptor])} {descriptor} descriptors from {path}")
        except Exception as e:
            logger.error(f"Failed to load descriptors from {path}: {e}")
            raise e
    logger.info(f"Available descriptor keys: {list(descriptors.keys())}")
    return descriptors

class CombinationMOSSArm:
    """
    MOSS (Minimax Optimal Strategy in the Stochastic case) arm for descriptor combinations.
    
    Based on Audibert & Bubeck (2009): "Minimax Policies for Adversarial and Stochastic Bandits"
    
    MOSS index: B_{i,s} = X̂_{i,s} + √(max(log(n/(Ks)), 0) / s)
    where:
    - X̂_{i,s} is empirical mean after s selections
    - n is horizon (max_iters)  
    - K is number of arms
    - s is number of selections of this arm
    """
    
    def __init__(self, category1_name, category2_name, descriptor1, descriptor2, combination_id, seed=None):
        self.category1_name = category1_name
        self.category2_name = category2_name
        self.descriptor1 = descriptor1
        self.descriptor2 = descriptor2
        self.combination_id = combination_id
        self.total_reward = 0.0
        self.num_selections = 0
        self.rewards_history = []  # For debugging and analysis
        self.seed = seed

    def moss_index(self, max_iters, num_arms):
        """
        Calculate MOSS index according to Audibert & Bubeck 2009.
        
        B_{i,s} = X̂_{i,s} + √(max(log(n/(Ks)), 0) / s)
        
        Returns infinity for unselected arms (ensures initial exploration).
        """
        if self.num_selections == 0:
            return float('inf')
        
        # Empirical mean (exploitation term)
        exploitation = self.total_reward / self.num_selections
        
        # MOSS confidence bound (exploration term)
        n = max_iters  # Horizon
        K = num_arms   # Number of arms
        s = self.num_selections
        
        # Key MOSS formula: max(log(n/(K*s)), 0)
        log_term = max(math.log(n / (K * s)), 0)
        exploration = math.sqrt(log_term / s)
        
        return exploitation + exploration

    def update(self, reward):
        """Update arm statistics with new reward - INDIVIDUAL UPDATE."""
        self.total_reward += reward
        self.num_selections += 1
        self.rewards_history.append(reward)
        
        logger.debug(f"MOSS Individual Update - Arm {self.combination_id}:")
        logger.debug(f"  Reward: {reward:.3f}")
        logger.debug(f"  New average: {self.get_average_reward():.3f}")
        logger.debug(f"  Total updates: {self.num_selections}")

    def get_average_reward(self):
        return self.total_reward / self.num_selections if self.num_selections > 0 else 0.0

    def get_descriptor_string(self):
        return f"- {self.category1_name}: {self.descriptor1}\n- {self.category2_name}: {self.descriptor2}"

    def get_statistics(self):
        if self.num_selections == 0:
            return {
                'avg_reward': 0.0,
                'selections': 0,
                'total_reward': 0.0,
                'reward_std': 0.0,
                'confidence_width': float('inf')
            }
        
        avg_reward = self.get_average_reward()
        reward_std = np.std(self.rewards_history) if len(self.rewards_history) > 1 else 0.0
        
        return {
            'avg_reward': avg_reward,
            'selections': self.num_selections,
            'total_reward': self.total_reward,
            'reward_std': reward_std,
        }
    
    def get_state_dict(self):
        return {
            'category1_name': self.category1_name,
            'category2_name': self.category2_name,
            'descriptor1': self.descriptor1,
            'descriptor2': self.descriptor2,
            'combination_id': self.combination_id,
            'total_reward': self.total_reward,
            'num_selections': self.num_selections,
            'rewards_history': self.rewards_history,
            'seed': self.seed
        }
    
    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        return (f"MOSSArm({self.combination_id}: {self.category1_name}+{self.category2_name}, "
                f"avg_reward={self.get_average_reward():.3f}, selections={self.num_selections})")

def initialize_combination_arms(descriptors_dict, seed=None):
    combination_arms = []
    combination_id = 0
    
    descriptor_keys = list(descriptors_dict.keys())
    logger.info(f"Available descriptor keys: {list(descriptors_dict.keys())}")
    
    if len(descriptor_keys) != 2:
        raise ValueError(f"Expected 2 descriptor categories, got {len(descriptor_keys)}: {descriptor_keys}")
    
    category1_key = descriptor_keys[0]
    category2_key = descriptor_keys[1]
    
    category1_descriptors = descriptors_dict[category1_key]
    category2_descriptors = descriptors_dict[category2_key]
    
    logger.info(f"Initializing MOSS with Individual Updates:")
    logger.info(f"- Category 1 ({category1_key}): {len(category1_descriptors)} items")
    logger.info(f"- Category 2 ({category2_key}): {len(category2_descriptors)} items")
    logger.info(f"- Total combinations: {len(category1_descriptors)} x {len(category2_descriptors)} = {len(category1_descriptors) * len(category2_descriptors)}")
    logger.info(f"- Seed: {seed}")
    
    for i, descriptor1 in enumerate(category1_descriptors):
        for j, descriptor2 in enumerate(category2_descriptors):
            arm = CombinationMOSSArm(
                category1_name=category1_key,
                category2_name=category2_key,
                descriptor1=descriptor1,
                descriptor2=descriptor2,
                combination_id=combination_id,
                seed=seed
            )
            combination_arms.append(arm)
            combination_id += 1
    
    return combination_arms

class CheckpointManager:
    def __init__(self, checkpoint_dir, experiment_name, keep_n_checkpoints=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_pattern = f"{experiment_name}_iter_*.pkl"
        
    def save_checkpoint(self, iteration, state_dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"{self.experiment_name}_iter_{iteration:06d}_{timestamp}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        state_dict['metadata'] = {
            'iteration': iteration,
            'timestamp': timestamp,
            'experiment_name': self.experiment_name,
            'save_time': datetime.now().isoformat()
        }
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            self._cleanup_old_checkpoints()
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        if checkpoint_path is None:
            logger.info("No checkpoint found to load")
            return None
        try:
            with open(checkpoint_path, 'rb') as f:
                state_dict = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            if 'metadata' in state_dict:
                metadata = state_dict['metadata']
                logger.info(f"  Iteration: {metadata.get('iteration', 'unknown')}")
                logger.info(f"  Saved at: {metadata.get('save_time', 'unknown')}")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def get_latest_checkpoint(self):
        checkpoint_files = list(self.checkpoint_dir.glob(self.checkpoint_pattern))
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        return checkpoint_files[0]
    
    def _cleanup_old_checkpoints(self):
        if self.keep_n_checkpoints <= 0:
            return
        checkpoint_files = list(self.checkpoint_dir.glob(self.checkpoint_pattern))
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        for old_checkpoint in checkpoint_files[self.keep_n_checkpoints:]:
            try:
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")

def create_experiment_state(combination_arms, entropy_tracker, adv_prompts, responses, 
                          scores, iters, arm_selection_counts, enhanced_analysis, args, current_iteration, seed):
    return {
        'current_iteration': current_iteration,
        'args': vars(args),
        'seed': seed,
        'combination_arms_state': [arm.get_state_dict() for arm in combination_arms],
        'entropy_tracker_state': entropy_tracker.get_state_dict(),
        'enhanced_analysis': enhanced_analysis,  
        'archives': {
            'adv_prompts': adv_prompts.__dict__ if hasattr(adv_prompts, '__dict__') else str(adv_prompts),
            'responses': responses.__dict__ if hasattr(responses, '__dict__') else str(responses),
            'scores': scores.__dict__ if hasattr(scores, '__dict__') else str(scores),
            'iters': iters.__dict__ if hasattr(iters, '__dict__') else str(iters),
        },
        'arm_selection_counts': arm_selection_counts,
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
    }

def restore_experiment_state(state_dict, descriptors_dict, seed):
    # Restore random states from checkpoint
    random.setstate(state_dict['random_state'])
    np.random.set_state(state_dict['numpy_random_state'])
    
    # Get seed from checkpoint or use provided seed
    checkpoint_seed = state_dict.get('seed', seed)
    
    combination_arms = initialize_combination_arms(descriptors_dict, seed=checkpoint_seed)
    arms_state = state_dict['combination_arms_state']
    for arm, arm_state in zip(combination_arms, arms_state):
        arm.load_state_dict(arm_state)
    
    entropy_tracker = SelectionEntropyTracker(len(combination_arms), seed=checkpoint_seed)
    entropy_tracker.load_state_dict(state_dict['entropy_tracker_state'])
    
    enhanced_analysis = state_dict.get('enhanced_analysis', [])
    adv_prompts = Archive("adv_prompts")
    responses = Archive("responses")
    scores = Archive("scores")
    iters = Archive("iterations")
    
    try:
        archives_data = state_dict['archives']
        if isinstance(archives_data['adv_prompts'], dict):
            adv_prompts.__dict__.update(archives_data['adv_prompts'])
            responses.__dict__.update(archives_data['responses'])
            scores.__dict__.update(archives_data['scores'])
            iters.__dict__.update(archives_data['iters'])
    except Exception as e:
        logger.warning(f"Could not fully restore archives: {e}")
    
    arm_selection_counts = state_dict['arm_selection_counts']
    current_iteration = state_dict['current_iteration']
    
    return combination_arms, entropy_tracker, adv_prompts, responses, scores, iters, arm_selection_counts, current_iteration, enhanced_analysis

def run_moss_with_individual_updates(
    args, config, seed_prompts=[], llms=None, fitness_fn=None, similarity_fn=None
):
    # Set random seed for reproducibility
    used_seed = set_random_seeds(args.seed)
    
    experiment_name = f"moss_individual_{Path(config.sample_prompts).stem}_{args.target_llm.split('/')[-1]}_seed{used_seed}"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=experiment_name,
        keep_n_checkpoints=args.keep_n_checkpoints
    )
    
    if not seed_prompts:
        seed_prompts = load_json(
            config.sample_prompts,
            field="question",
            num_samples=args.num_samples,
            shuffle=args.shuffle,
        )

    descriptors_dict = load_descriptors(config)
    
    start_iteration = 0
    checkpoint_loaded = False
    enhanced_analysis = []  # Initialize analysis tracking
    
    if args.resume_from_checkpoint:
        state_dict = checkpoint_manager.load_checkpoint(args.resume_from_checkpoint)
        if state_dict:
            logger.info(f"Resuming from specific checkpoint: {args.resume_from_checkpoint}")
            combination_arms, entropy_tracker, adv_prompts, responses, scores, iters, arm_selection_counts, start_iteration, enhanced_analysis = restore_experiment_state(
                state_dict, descriptors_dict, used_seed
            )
            start_iteration += 1
            checkpoint_loaded = True
    elif args.auto_resume:
        state_dict = checkpoint_manager.load_checkpoint()
        if state_dict:
            logger.info("Auto-resuming from latest checkpoint")
            combination_arms, entropy_tracker, adv_prompts, responses, scores, iters, arm_selection_counts, start_iteration, enhanced_analysis = restore_experiment_state(
                state_dict, descriptors_dict, used_seed
            )
            start_iteration += 1
            checkpoint_loaded = True
    
    if not checkpoint_loaded:
        logger.info("Starting fresh MOSS experiment with individual updates")
        combination_arms = initialize_combination_arms(descriptors_dict, seed=used_seed)
        entropy_tracker = SelectionEntropyTracker(len(combination_arms), seed=used_seed)
        adv_prompts = Archive("adv_prompts")
        responses = Archive("responses")
        scores = Archive("scores")
        iters = Archive("iterations")
        arm_selection_counts = {arm.combination_id: 0 for arm in combination_arms}
        enhanced_analysis = []  # Fresh analysis tracking
    
    logger.info(f"Initialized {len(combination_arms)} MOSS arms with INDIVIDUAL UPDATES")
    logger.info(f"Starting from iteration: {start_iteration}")
    logger.info(f"Random seed: {used_seed}")

    dataset_name = Path(config.sample_prompts).stem
    log_dir = Path(args.log_dir) / config.target_llm.model_kwargs["model"] / dataset_name / f"seed_{used_seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    for i in range(start_iteration, args.max_iters):
        iteration_start_time = time.time()
        logger.info(f"##### ITERATION: {i} (Seed: {used_seed}) #####")

        try:
            if i < len(seed_prompts):
                prompt = seed_prompts[i]
            else:
                if adv_prompts.flatten_values():
                    prompt = random.choice(adv_prompts.flatten_values())
                else:
                    prompt = seed_prompts[i % len(seed_prompts)]
            
            # MOSS: Select arm with highest MOSS index
            chosen_arm = max(combination_arms, key=lambda arm: arm.moss_index(args.max_iters, len(combination_arms)))
            
            entropy_tracker.update_selection(chosen_arm.combination_id)
            arm_selection_counts[chosen_arm.combination_id] += 1
            entropy_metrics = entropy_tracker.get_diversity_metrics()
            
            descriptor_str = chosen_arm.get_descriptor_string()
            moss_score = chosen_arm.moss_index(args.max_iters, len(combination_arms))
            
            logger.info(f"Selected combination {chosen_arm.combination_id}: {chosen_arm.category1_name} + {chosen_arm.category2_name}")
            logger.info(f"MOSS: avg_reward={chosen_arm.get_average_reward():.3f}, moss_index={moss_score:.3f}, selections={chosen_arm.num_selections}")
            logger.info(f"Entropy: {entropy_metrics['selection_entropy']:.3f} (0=uniform, 1=focused)")
            
            mutator_model = config.mutator_llm.model_kwargs["model"]
            prompt_ = MUTATOR_PROMPT.format(
                descriptor=descriptor_str.strip(), prompt=prompt
            )
            mutated_prompts = llms[mutator_model].batch_generate(
                [prompt_] * args.num_mutations, config.mutator_llm.sampling_params
            )
            
            mutated_prompts = [
                p for p in mutated_prompts if similarity_fn.score(p, prompt_) < args.sim_threshold
            ]
            logger.info(f"Generated {len(mutated_prompts)} mutated prompts after similarity filtering")

            if mutated_prompts:
                try:
                    target_prompts = [
                        TARGET_PROMPT.format(prompt=p.strip()) for p in mutated_prompts
                    ]
                    target_model = config.target_llm.model_kwargs["model"]
                    candidates = llms[target_model].batch_generate(
                        target_prompts, config.target_llm.sampling_params
                    )
                    
                    fitness_scores = fitness_fn.batch_score(
                        mutated_prompts, candidates, config.fitness_llm.sampling_params
                    )
                    
                    logger.info(f"Fitness scores: min={min(fitness_scores):.3f}, max={max(fitness_scores):.3f}, mean={np.mean(fitness_scores):.3f}")
                    
                    # INDIVIDUAL UPDATES: Update MOSS with each individual score
                    logger.info(f"Applying individual MOSS updates for {len(fitness_scores)} prompts")
                    for idx, score in enumerate(fitness_scores):
                        chosen_arm.update(score)  # Each score individually updates MOSS statistics
                        logger.debug(f"  Individual update {idx+1}/{len(fitness_scores)}: score={score:.3f}")
                    
                    logger.info(f"Updated MOSS arm {chosen_arm.combination_id} with {len(fitness_scores)} individual updates")
                    logger.info(f"New average reward: {chosen_arm.get_average_reward():.3f}, total updates: {chosen_arm.num_selections}")
                    
                    # Filter for archive (only high-quality prompts)
                    filtered_data = [
                        (p, c, s) for p, c, s in zip(mutated_prompts, candidates, fitness_scores)
                        if s > args.fitness_threshold
                    ]
                    
                    logger.info(f"Prompts passing fitness threshold: {len(filtered_data)}/{len(mutated_prompts)}")
                    
                    if filtered_data:
                        filtered_prompts, filtered_candidates, filtered_scores = zip(*filtered_data)
                        
                        # Update archives using tuple key as expected by Archive class
                        key = (chosen_arm.descriptor1, chosen_arm.descriptor2)
                        if not adv_prompts.exists(key):
                            adv_prompts.add(key, list(filtered_prompts))
                            responses.add(key, list(filtered_candidates))
                            scores.add(key, list(filtered_scores))
                            iters.add(key, [i] * len(filtered_prompts))
                        else:
                            adv_prompts.extend(key, list(filtered_prompts))
                            responses.extend(key, list(filtered_candidates))
                            scores.extend(key, list(filtered_scores))
                            iters.extend(key, [i] * len(filtered_prompts))
                            
                        logger.info(f"Archive updated with {len(filtered_prompts)} high-quality prompts")

                except Exception as eval_error:
                    logger.error(f"Evaluation failed: {eval_error}")
                    chosen_arm.update(args.eval_failure_reward)
                    logger.info(f"Updated MOSS arm {chosen_arm.combination_id} with evaluation failure reward {args.eval_failure_reward}")

            else:
                # No mutated prompts generated, apply negative reward
                chosen_arm.update(args.negative_reward)
                logger.info(f"No mutated prompts generated, updated MOSS arm {chosen_arm.combination_id} with negative reward {args.negative_reward}")

            # Enhanced analysis collection
            iteration_analysis = {
                'iteration': i,
                'seed': used_seed,
                'chosen_arm_id': chosen_arm.combination_id,
                'moss_index': moss_score,
                'avg_reward': chosen_arm.get_average_reward(),
                'num_selections': chosen_arm.num_selections,
                'entropy_metrics': entropy_metrics,
                'iteration_time': time.time() - iteration_start_time,
                'num_mutated_prompts': len(mutated_prompts) if mutated_prompts else 0,
                'prompts_above_threshold': len(filtered_data) if 'filtered_data' in locals() else 0
            }
            enhanced_analysis.append(iteration_analysis)

            # Detailed logging every 50 iterations
            if i > 0 and (i + 1) % 50 == 0:
                logger.info("=" * 80)
                logger.info(f"MOSS & ENTROPY STATISTICS (Iteration {i}, Seed: {used_seed})")
                logger.info("=" * 80)
                
                # MOSS arm statistics
                sorted_arms = sorted(combination_arms, key=lambda x: x.get_average_reward(), reverse=True)
                logger.info("Top 5 performing MOSS arms:")
                for arm in sorted_arms[:5]:
                    stats = arm.get_statistics()
                    logger.info(f"  {arm} (std: {stats['reward_std']:.3f})")
                
                # Selection entropy statistics
                entropy_stats = entropy_tracker.get_diversity_metrics()
                logger.info(f"Selection Entropy: {entropy_stats['selection_entropy']:.3f} / {entropy_stats['max_entropy']:.3f}")
                logger.info(f"Arms selected: {entropy_stats['num_selected_arms']}/{len(combination_arms)}")
                logger.info(f"Coverage rate: {entropy_stats['coverage_rate']:.1%}")
                logger.info(f"Gini coefficient: {entropy_stats['gini_coefficient']:.3f}")
                
                # Exploration-exploitation analysis
                exp_exp_analysis = entropy_tracker.analyze_exploration_exploitation()
                if exp_exp_analysis.get('status') != 'insufficient_data':
                    logger.info(f"Entropy trend: {exp_exp_analysis.get('entropy_trend', 'unknown')}")
                logger.info("=" * 80)

            # Save checkpoint at specified intervals
            if (i + 1) % args.checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at iteration {i} (seed: {used_seed})")
                state_dict = create_experiment_state(
                    combination_arms, entropy_tracker, adv_prompts, responses, scores, iters,
                    arm_selection_counts, enhanced_analysis, args, i, used_seed
                )
                checkpoint_path = checkpoint_manager.save_checkpoint(i, state_dict)
                if checkpoint_path:
                    logger.info(f"Checkpoint saved with entropy data: {checkpoint_path}")
                else:
                    logger.warning("Failed to save checkpoint")

            # Periodic logging
            if i > 0 and (i + 1) % args.log_interval == 0:
                timestamp = time.strftime(r"%Y%m%d-%H%M%S")
                save_iteration_log(
                    log_dir, adv_prompts, responses, scores, iters, f"moss_individual_iter_{i}_seed_{used_seed}", iteration=i
                )
                # Save enhanced analysis
                enhanced_analysis_file = log_dir / f"enhanced_analysis_moss_individual_seed_{used_seed}.json"
                with open(enhanced_analysis_file, 'w') as f:
                    json.dump(enhanced_analysis, f, indent=2)

        except Exception as e:
            logger.error(f"Error in iteration {i}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Apply negative reward for iteration failure
            if 'chosen_arm' in locals():
                chosen_arm.update(args.negative_reward)
                logger.info(f"Applied negative reward for iteration failure")
            
            # Save emergency checkpoint with full entropy data
            emergency_state_dict = create_experiment_state(
                combination_arms, entropy_tracker, adv_prompts, responses, scores, iters,
                arm_selection_counts, enhanced_analysis, args, i, used_seed
            )
            emergency_checkpoint_path = checkpoint_manager.checkpoint_dir / f"{experiment_name}_emergency_iter_{i}.pkl"
            try:
                with open(emergency_checkpoint_path, 'wb') as f:
                    pickle.dump(emergency_state_dict, f)
                logger.info(f"Emergency checkpoint saved with entropy data: {emergency_checkpoint_path}")
            except Exception as save_error:
                logger.error(f"Failed to save emergency checkpoint: {save_error}")
            
            # Continue to next iteration instead of crashing
            continue

    # Save final checkpoint with complete entropy history
    logger.info("Saving final checkpoint with complete entropy history")
    final_state_dict = create_experiment_state(
        combination_arms, entropy_tracker, adv_prompts, responses, scores, iters,
        arm_selection_counts, enhanced_analysis, args, args.max_iters - 1, used_seed
    )
    checkpoint_manager.save_checkpoint(args.max_iters - 1, final_state_dict)

    # Save final logs
    timestamp = time.strftime(r"%Y%m%d-%H%M%S")
    save_iteration_log(log_dir, adv_prompts, responses, scores, iters, f"moss_individual_final_seed_{used_seed}", iteration=args.max_iters - 1)
    enhanced_analysis_file = log_dir / f"enhanced_analysis_moss_individual_final_seed_{used_seed}.json"
    with open(enhanced_analysis_file, 'w') as f:
        json.dump(enhanced_analysis, f, indent=2)

    # Final comprehensive statistics
    logger.info("=== FINAL MOSS ANALYSIS WITH INDIVIDUAL UPDATES ===")
    logger.info(f"Random Seed Used: {used_seed}")
    sorted_arms = sorted(combination_arms, key=lambda x: x.get_average_reward(), reverse=True)
    logger.info("Top 10 performing combinations (by average reward):")
    for i, arm in enumerate(sorted_arms[:10]):
        logger.info(f"  {i+1}. {arm}")
    
    final_entropy_stats = entropy_tracker.get_diversity_metrics()
    final_exploration_analysis = entropy_tracker.analyze_exploration_exploitation()
    
    logger.info("=== FINAL ENTROPY ANALYSIS ===")
    logger.info(f"Final Selection Entropy: {final_entropy_stats['selection_entropy']:.3f}")
    logger.info(f"Final Coverage: {final_entropy_stats['num_selected_arms']}/{len(combination_arms)} arms ({final_entropy_stats['coverage_rate']:.1%})")
    logger.info(f"Selection Gini coefficient: {final_entropy_stats['gini_coefficient']:.3f}")
    
    logger.info("=== EXPLORATION VS EXPLOITATION ANALYSIS ===")
    if final_exploration_analysis.get('status') != 'insufficient_data':
        logger.info(f"Early entropy (avg): {final_exploration_analysis['early_entropy_avg']:.3f}")
        logger.info(f"Late entropy (avg): {final_exploration_analysis['late_entropy_avg']:.3f}")
        logger.info(f"Entropy trend: {final_exploration_analysis['entropy_trend']}")
        logger.info(f"Total entropy change: {final_exploration_analysis['total_entropy_change']:.3f}")
        logger.info(f"Convergence rate: {final_exploration_analysis['convergence_rate']:.4f}")
        if final_exploration_analysis['entropy_trend'] == 'decreasing':
            logger.info("Algorithm properly converged from exploration to exploitation")
        else:
            logger.info("Algorithm maintained high exploration throughout")
    
    # Generate comprehensive plots
    logger.info("Generating analysis plots...")
    generate_moss_plots(enhanced_analysis, entropy_tracker, combination_arms, log_dir, args, used_seed)
    
    # Create seed info file
    seed_info_file = log_dir / f"experiment_seed_{used_seed}.txt"
    with open(seed_info_file, 'w') as f:
        f.write(f"Random Seed Used: {used_seed}\n")
        f.write(f"Experiment: MOSS Individual Updates\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Iterations: {args.max_iters}\n")
        f.write(f"Total prompts in archive: {len(adv_prompts.flatten_values())}\n")
        f.write(f"Final entropy: {final_entropy_stats['selection_entropy']:.3f}\n")
        f.write(f"Arms explored: {final_entropy_stats['num_selected_arms']}/{len(combination_arms)}\n")
        f.write(f"Reproducibility: Use --seed {used_seed} to reproduce exact results\n")
    
    logger.info(f"Total prompts in archive: {len(adv_prompts.flatten_values())}")
    logger.info(f"REPRODUCIBILITY: Seed {used_seed} ensures identical results")

    return adv_prompts, responses, scores, entropy_tracker

def generate_moss_plots(enhanced_analysis, entropy_tracker, combination_arms, log_dir, args, seed):
    """Generate comprehensive analysis plots for MOSS experiment."""
    
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })
    
    logger.info(f"Saving plots to: {plots_dir}")
    
    if not enhanced_analysis:
        logger.warning("No analysis data available for plotting")
        return
    
    # Extract data for plotting
    iterations = [data['iteration'] for data in enhanced_analysis]
    entropy_values = [data['entropy_metrics']['selection_entropy'] for data in enhanced_analysis]
    coverage_values = [data['entropy_metrics']['coverage_rate'] for data in enhanced_analysis]
    moss_indices = [data['moss_index'] for data in enhanced_analysis if data['moss_index'] != float('inf')]
    avg_rewards = [data['avg_reward'] for data in enhanced_analysis]
    num_selections = [data['num_selections'] for data in enhanced_analysis]
    
    # 1. Main Dashboard Plot (2x3 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MOSS with Individual Updates - Complete Analysis (Seed: {seed})', fontsize=20, fontweight='bold')
    
    # Plot 1: Selection Entropy Over Time
    axes[0, 0].plot(iterations, entropy_values, 'b-', linewidth=2, alpha=0.8)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Balanced Exploration')
    axes[0, 0].set_title('Selection Entropy Evolution', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Selection Entropy [0=uniform, 1=focused]')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Coverage (Diversity) Over Time
    axes[0, 1].plot(iterations, [c * 100 for c in coverage_values], 'g-', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Archive Coverage (Diversity)', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Coverage (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MOSS Index Over Time (excluding infinity values)
    if moss_indices:
        moss_iterations = [iterations[i] for i, data in enumerate(enhanced_analysis) if data['moss_index'] != float('inf')]
        axes[0, 2].plot(moss_iterations, moss_indices, 'purple', linewidth=2, alpha=0.8)
        axes[0, 2].set_title('MOSS Index Evolution', fontweight='bold')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('MOSS Index')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'No finite MOSS indices\nto display', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].set_title('MOSS Index Evolution', fontweight='bold')
    
    # Plot 4: Average Reward Over Time
    axes[1, 0].plot(iterations, avg_rewards, 'orange', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Average Reward Evolution', fontweight='bold')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Average Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Number of Selections Over Time
    axes[1, 1].plot(iterations, num_selections, 'red', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('Arm Selection Count', fontweight='bold')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Number of Selections')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Arms Performance Distribution
    arms_with_data = [arm for arm in combination_arms if arm.num_selections > 0]
    if arms_with_data:
        arm_rewards = [arm.get_average_reward() for arm in arms_with_data]
        axes[1, 2].hist(arm_rewards, bins=20, alpha=0.7, color='brown', density=True)
        axes[1, 2].axvline(x=np.mean(arm_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(arm_rewards):.3f}')
        axes[1, 2].set_title('Arms Reward Distribution', fontweight='bold')
        axes[1, 2].set_xlabel('Average Reward')
        axes[1, 2].set_ylabel('Density')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No arm data\nto display', 
                       horizontalalignment='center', verticalalignment='center', 
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Arms Reward Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'moss_dashboard_seed_{seed}.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / f'moss_dashboard_seed_{seed}.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info("All plots generated successfully!")
    logger.info(f"Plots saved in: {plots_dir}")
    logger.info("Generated plots:")
    logger.info(f"   - moss_dashboard_seed_{seed}.png (Main overview)")

def main():
    """Main entry point for MOSS with individual updates."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set random seed early
        used_seed = set_random_seeds(args.seed)

        # Load configuration and seed prompts
        config = ConfigurationLoader.load(args.config_file)
        
        # Update configuration based on command-line arguments
        config.target_llm.model_kwargs["model"] = args.target_llm
        config.sample_prompts = args.dataset
        
        # Initialize language models and scoring functions
        logger.info("Initializing language models...")
        llms = initialize_language_models(config)
        fitness_fn = LlamaGuard(config.fitness_llm.model_kwargs)
        similarity_fn = BleuScoreNLTK()
        logger.info("Language models initialized successfully")
        
        # Show configuration
        print("=" * 80)
        print("MOSS INDIVIDUAL UPDATES + REPRODUCIBLE SEEDS")
        print("=" * 80)
        print(f"Random Seed: {used_seed}")
        print(f"Configuration: {config}")
        print(f"Max iterations: {args.max_iters}")
        print(f"Algorithm: MOSS with Individual Updates")
        print(f"Checkpoint interval: {args.checkpoint_interval}")
        print(f"Auto resume: {args.auto_resume}")
        print("\nREPRODUCIBILITY FEATURES:")
        print(f"Random seed set to: {used_seed}")
        print("All random number generators seeded for identical results")
        print("Experiment name includes seed for easy identification")
        print("Checkpoint files preserve random states")
        print("Log directories organized by seed")
        print("\nMOSS ALGORITHM:")
        print("MOSS index: B_{i,s} = X̂_{i,s} + √(max(log(n/(Ks)), 0) / s)")
        print("Individual Updates: Each prompt updates arm statistics separately")
        print("Minimax optimal regret bounds")
        print("=" * 80)
        
        # Run the adversarial prompt generation process with MOSS
        adv_prompts, responses, scores, entropy_tracker = run_moss_with_individual_updates(
            args,
            config,
            seed_prompts=[],
            llms=llms,
            fitness_fn=fitness_fn,
            similarity_fn=similarity_fn,
        )
        
        # Final entropy analysis
        final_entropy_stats = entropy_tracker.get_diversity_metrics()
        print("\n" + "=" * 80)
        print("MOSS INDIVIDUAL UPDATES EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Random Seed Used: {used_seed}")
        print(f"Total prompts generated: {len(adv_prompts.flatten_values())}")
        print(f"Final entropy: {final_entropy_stats['selection_entropy']:.3f} (0=uniform, 1=focused)")
        print(f"Arms coverage: {final_entropy_stats['num_selected_arms']}/{entropy_tracker.num_arms}")
        print("MOSS Algorithm: Minimax optimal strategy with individual updates")
        print(f"REPRODUCIBILITY: Seed {used_seed} ensures identical results")
        print(f"To reproduce this exact experiment, use: --seed {used_seed}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        print("\n" + "=" * 80)
        print("MOSS INDIVIDUAL UPDATES EXPERIMENT INTERRUPTED BY USER")
        print("=" * 80)
        print(f"Seed used: {used_seed}")
        print("The experiment can be resumed using the latest checkpoint with:")
        print(f"python {sys.argv[0]} --auto_resume --seed {used_seed} [other arguments]")
        
    except Exception as e:
        logger.error(f"Process failed with error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        print(f"\nERROR: MOSS experiment failed with: {e}")
        print(f"Seed used: {used_seed}")
        print("Check logs for more details. Emergency checkpoints with entropy data may have been saved.")
        raise

if __name__ == "__main__":
    main()