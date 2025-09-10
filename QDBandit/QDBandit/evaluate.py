import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from QDBandit.archive import Archive
from QDBandit.prompts import EVALUATE_PROMPT
from QDBandit.configs import EvalConfigLoader, EvaluationConfig
from QDBandit.llms.vllm import vLLM


class EvaluationProcessor:
    def __init__(
        self,
        config: EvaluationConfig,
        keyword: str
    ):
        """
        Initialize the evaluation processor.

        :param config: Configuration object
        :param keyword: Keyword of global config file name
        """
        self.config = config
        self.keyword = keyword

        # Initialize directories and parameters
        self.log_dir = Path(self.config.log_dir)
        self.method = self.config.method

        self.sampling_params = self.config.judge_llm.sampling_params
        self.llm = vLLM(self.config.judge_llm.model_kwargs)
        self.algo = self.config.algo

    def _save_eval_log(self, file_name: str, archive_instances: List[Archive]) -> None:
        """Save evaluation logs to a JSON file."""
        log_path = self.log_dir / f"eval_{self.method}_{file_name}.json"

        log_data = {
            archive.name: {str(key): value for key, value in archive._archive.items()}
            for archive in archive_instances
        }

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Log saved to {log_path}")

    def _process(
        self, adv_prompts: Archive, responses: Archive, scores: Archive, iters: Archive
    ) -> Dict[str, Archive]:
        """Process evaluation using median or max method."""
        eval_prompts = Archive("eval_prompts")
        eval_responses = Archive("eval_responses")
        eval_scores = Archive("eval_scores")
        eval_iters = Archive("eval_iters")
        judge_responses = Archive("judge_responses")

        # Process all prompts
        for key in adv_prompts._archive.keys():
            chosen_prompts = adv_prompts.get(key)
            chosen_responses = responses.get(key)
            chosen_scores = scores.get(key)
            chosen_iters = iters.get(key)

            queries = [
                EVALUATE_PROMPT.format(prompt=prompt, response=response)
                for prompt, response in zip(chosen_prompts, chosen_responses)
            ]
            outputs = self.llm.batch_generate(queries, self.sampling_params)

            eval_prompts.add(key, chosen_prompts)
            eval_responses.add(key, chosen_responses)
            eval_scores.add(key, chosen_scores)
            eval_iters.add(key, chosen_iters)
            judge_responses.add(key, outputs)

        return {
            "eval_prompts": eval_prompts,
            "eval_responses": eval_responses,
            "eval_scores": eval_scores,
            "eval_iters": eval_iters,
            "judge_responses": judge_responses,
        }

    def process_evaluations(self):
        """Process evaluations across log files."""
        # Get sorted log files starting with the specified method
        log_files = sorted(
            [f for f in os.listdir(self.log_dir) if f.startswith(self.algo) and self.keyword in f]
        )

        for idx, log_file in enumerate(log_files):
            # Load log file
            path = os.path.join(self.log_dir, log_file)
            with open(path, "r") as f:
                log_dict = json.load(f)

            # Create Archives from log data
            adv_prompts = Archive("adv_prompts")
            adv_prompts.load_from_dict(log_dict["adv_prompts"])
            responses = Archive("responses")
            responses.load_from_dict(log_dict["responses"])
            scores = Archive("scores")
            scores.load_from_dict(log_dict["scores"])
            iters = Archive("iters")
            iters.load_from_dict(log_dict["iters"])

            # Process based on method
            processed_archives = self._process(adv_prompts, responses, scores, iters)

            # Save evaluation log
            self._save_eval_log(log_files[idx], list(processed_archives.values()))


def main():
    parser = argparse.ArgumentParser(description="Evaluation Processing Script")
    parser.add_argument(
        "-c", "--config", default="configs/eval.yml", help="Path to configuration file"
    )
    parser.add_argument("-l", "--log_dir", help="Override log directory from config")
    parser.add_argument("-a", "--algo", default="rainbowplus", help="Override algorithm from config")
    parser.add_argument("-m", "--method", default="all", help="Override evaluation method from config")
    parser.add_argument("-k", "--keyword", default="global", help="Keyword of global config file name")

    args = parser.parse_args()

    # Update config with command-line arguments
    config = EvalConfigLoader.load(args.config)
    config.log_dir = args.log_dir or config.log_dir
    config.algo = args.algo or config.algo
    config.method = args.method or config.method

    processor = EvaluationProcessor(
        config=config,
        keyword=args.keyword
    )
    processor.process_evaluations()


if __name__ == "__main__":
    main()
