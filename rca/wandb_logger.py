"""Wandb integration for OpenRCA experiment tracking."""
import os
import json
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandbLogger:
    """Centralized wandb logging for OpenRCA experiments."""

    def __init__(self, project="OpenRCA", run_name=None, config=None, enabled=True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.run = None
        self.step_count = 0
        self.task_table = None
        self.trajectory_table = None

        if not self.enabled:
            if enabled and not WANDB_AVAILABLE:
                print("WARNING: wandb not installed. Install with: pip install wandb")
            return

        if run_name is None:
            run_name = f"openrca-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            reinit=True,
        )

        # Define tables for structured logging
        self.task_table = wandb.Table(columns=[
            "idx", "task_index", "catalog", "instruction",
            "prediction", "groundtruth",
            "passed_criteria", "failed_criteria",
            "score", "num_steps", "duration_s"
        ])
        self.trajectory_table = wandb.Table(columns=[
            "idx", "task_index", "step", "code", "result"
        ])

    def log_config(self, api_config, args):
        """Log experiment configuration."""
        if not self.enabled:
            return
        config_dict = {
            "model": api_config.get("MODEL", "unknown"),
            "source": api_config.get("SOURCE", "unknown"),
            "max_tokens": api_config.get("MAX_TOKENS", 8192),
        }
        # Log vLLM-specific config
        if api_config.get("SOURCE") == "vLLM":
            config_dict["vllm_endpoint"] = api_config.get("API_BASE", "unknown")
        wandb.config.update(config_dict)
        if hasattr(args, '__dict__'):
            wandb.config.update(vars(args))

    def log_step(self, idx, step, analysis=None, instruction=None, code=None, result=None):
        """Log a single agent step."""
        if not self.enabled:
            return
        self.step_count += 1
        log_dict = {
            "task_idx": idx,
            "agent_step": step,
            "global_step": self.step_count,
        }
        if analysis:
            log_dict["analysis_len"] = len(str(analysis))
        if code:
            log_dict["code_len"] = len(str(code))
        if result:
            log_dict["result_len"] = len(str(result))
        wandb.log(log_dict, step=self.step_count)

    def log_task_result(self, idx, task_index, catalog, instruction,
                        prediction, groundtruth,
                        passed_criteria, failed_criteria, score,
                        num_steps=0, duration_s=0.0,
                        trajectory=None, prompt=None):
        """Log a completed task evaluation result."""
        if not self.enabled:
            return

        # Log scalar metrics
        wandb.log({
            f"score/{catalog}": score,
            "score/current": score,
            "task_idx": idx,
            "num_steps": num_steps,
            "duration_s": duration_s,
        })

        # Add to task table
        if self.task_table is not None:
            self.task_table.add_data(
                idx, task_index, catalog,
                str(instruction)[:500],
                str(prediction)[:1000],
                str(groundtruth)[:500],
                str(passed_criteria),
                str(failed_criteria),
                score, num_steps, round(duration_s, 2)
            )

        # Log trajectory steps
        if trajectory and self.trajectory_table is not None:
            for step_i, step in enumerate(trajectory):
                self.trajectory_table.add_data(
                    idx, task_index, step_i + 1,
                    str(step.get('code', ''))[:2000],
                    str(step.get('result', ''))[:2000]
                )

        # Save prompt as artifact (every 10 tasks to avoid flooding)
        if prompt and idx % 10 == 0:
            try:
                artifact = wandb.Artifact(
                    f"prompt-task-{idx}", type="prompt",
                    description=f"Full prompt for task {idx} ({task_index})"
                )
                with artifact.new_file(f"prompt_{idx}.json", mode="w") as f:
                    json.dump({"messages": prompt} if isinstance(prompt, list) else prompt, f, ensure_ascii=False, indent=2)
                wandb.log_artifact(artifact)
            except Exception as e:
                print(f"Warning: Failed to log prompt artifact: {e}")

    def log_running_scores(self, scores, nums):
        """Log running aggregate scores."""
        if not self.enabled:
            return
        log_dict = {}
        for key in scores:
            if nums.get(key, 0) > 0:
                log_dict[f"accuracy/{key}"] = scores[key] / nums[key]
                log_dict[f"count/{key}"] = nums[key]
                log_dict[f"correct/{key}"] = scores[key]
        wandb.log(log_dict)

    def log_final_summary(self, scores, nums, dataset):
        """Log final summary metrics and tables."""
        if not self.enabled:
            return

        summary = {"dataset": dataset}
        for key in scores:
            if nums.get(key, 0) > 0:
                summary[f"final_accuracy/{key}"] = scores[key] / nums[key]
                summary[f"final_count/{key}"] = nums[key]

        wandb.log(summary)

        # Log tables
        if self.task_table is not None:
            wandb.log({"task_results": self.task_table})
        if self.trajectory_table is not None:
            wandb.log({"trajectories": self.trajectory_table})

    def finish(self):
        """Finish the wandb run."""
        if self.enabled and self.run:
            wandb.finish()
