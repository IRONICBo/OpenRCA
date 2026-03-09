import os
import sys
import json
import argparse
import threading
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from main.evaluate import evaluate
from rca.api_router import configs

from datetime import datetime
from time import time as timer
from loguru import logger
from nbformat import v4 as nbf
import pandas as pd

from rca.wandb_logger import WandbLogger


def run_with_timeout(func, args=(), kwargs=None, timeout=600):
    """Cross-platform timeout using threading (works on macOS/Linux/B200)."""
    kwargs = kwargs or {}
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution exceeded the time limit of {timeout}s")
    if exception[0]:
        raise exception[0]
    return result[0]


def main(args, uid, dataset, wb_logger):

    from rca.baseline.rca_agent.rca_agent import RCA_Agent
    import rca.baseline.rca_agent.prompt.agent_prompt as ap
    if dataset == "Telecom":
        import rca.baseline.rca_agent.prompt.basic_prompt_Telecom as bp
    elif dataset == "Bank":
        import rca.baseline.rca_agent.prompt.basic_prompt_Bank as bp
    elif dataset == "Market/cloudbed-1" or dataset == "Market/cloudbed-2":
        import rca.baseline.rca_agent.prompt.basic_prompt_Market as bp

    inst_file = f"dataset/{dataset}/query.csv"
    gt_file = f"dataset/{dataset}/record.csv"
    model_tag = configs['MODEL'].split('/')[-1]
    eval_file = f"test/result/{dataset}/agent-{args.tag}-{model_tag}.csv"
    obs_path = f"test/monitor/{dataset}/agent-{args.tag}-{model_tag}"
    unique_obs_path = f"{obs_path}/{uid}"

    if not os.path.exists(inst_file) or not os.path.exists(gt_file):
        raise FileNotFoundError(f"Dataset files not found. Please download the dataset first: {inst_file}, {gt_file}")

    instruct_data = pd.read_csv(inst_file)
    gt_data = pd.read_csv(gt_file)

    for subdir in ["history", "trajectory", "prompt"]:
        os.makedirs(f"{unique_obs_path}/{subdir}", exist_ok=True)

    if not os.path.exists(eval_file):
        os.makedirs(f"test/result/{dataset}", exist_ok=True)
        eval_df = pd.DataFrame(columns=["instruction", "prediction", "groundtruth", "passed", "failed", "score"])
    else:
        eval_df = pd.read_csv(eval_file)

    scores = {"total": 0, "easy": 0, "middle": 0, "hard": 0}
    nums = {"total": 0, "easy": 0, "middle": 0, "hard": 0}

    logger.info(f"Using dataset: {dataset}")
    logger.info(f"Using model: {model_tag}")
    logger.info(f"Using source: {configs['SOURCE']}")

    for idx, row in instruct_data.iterrows():

        if idx < args.start_idx:
            continue
        if idx > args.end_idx:
            break

        instruction = row["instruction"]
        task_index = row["task_index"]
        scoring_points = row["scoring_points"]
        task_id = int(task_index.split('_')[1])
        best_score = 0

        if task_id <= 3:
            catalog = "easy"
        elif task_id <= 6:
            catalog = "middle"
        else:
            catalog = "hard"

        for i in range(args.sample_num):
            uuid = uid + f"_#{idx}-{i}"
            nb = nbf.new_notebook()
            nbfile = f"{unique_obs_path}/trajectory/{uuid}.ipynb"
            promptfile = f"{unique_obs_path}/prompt/{uuid}.json"
            logfile = f"{unique_obs_path}/history/{uuid}.log"
            logger.remove()
            logger.add(sys.stdout, colorize=True, enqueue=True, level="INFO")
            logger.add(logfile, colorize=True, enqueue=True, level="INFO")
            logger.debug('\n' + "#"*80 + f"\n{uuid}: {task_index}\n" + "#"*80)

            task_start_time = timer()
            try:
                agent = RCA_Agent(ap, bp)

                def agent_run():
                    return agent.run(instruction, logger,
                                     max_step=args.controller_max_step,
                                     max_turn=args.controller_max_turn)

                prediction, trajectory, prompt = run_with_timeout(
                    agent_run, timeout=args.timeout
                )

                task_duration = timer() - task_start_time
                num_steps = len(trajectory) if trajectory else 0

                for step in trajectory:
                    code_cell = nbf.new_code_cell(step['code'])
                    result_cell = nbf.new_markdown_cell(f"```\n{step['result']}\n```")
                    nb.cells.append(code_cell)
                    nb.cells.append(result_cell)
                with open(nbfile, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, ensure_ascii=False, indent=4)
                logger.info(f"Trajectory has been saved to {nbfile}")

                with open(promptfile, 'w', encoding='utf-8') as f:
                    json.dump({"messages": prompt}, f, ensure_ascii=False, indent=4)
                logger.info(f"Prompt has been saved to {promptfile}")

                new_eval_df = pd.DataFrame([{"row_id": idx,
                                            "task_index": task_index,
                                            "instruction": instruction,
                                            "prediction": prediction,
                                            "groundtruth": '\n'.join([f'{col}: {gt_data.iloc[idx][col]}' for col in gt_data.columns if col != 'description']),
                                            "passed": "N/A",
                                            "failed": "N/A",
                                            "score": "N/A"}])
                eval_df = pd.concat([eval_df, new_eval_df], ignore_index=True)
                eval_df.to_csv(eval_file, index=False)

                passed_criteria, failed_criteria, score = evaluate(prediction, scoring_points)

                logger.info(f"Prediction: {prediction}")
                logger.info(f"Scoring Points: {scoring_points}")
                logger.info(f"Passed Criteria: {passed_criteria}")
                logger.info(f"Failed Criteria: {failed_criteria}")
                logger.info(f"Score: {score}")
                best_score = max(best_score, score)

                eval_df.loc[eval_df.index[-1], "passed"] = '\n'.join(passed_criteria)
                eval_df.loc[eval_df.index[-1], "failed"] = '\n'.join(failed_criteria)
                eval_df.loc[eval_df.index[-1], "score"] = score
                eval_df.to_csv(eval_file, index=False)

                # Log to wandb
                wb_logger.log_task_result(
                    idx=idx, task_index=task_index, catalog=catalog,
                    instruction=instruction, prediction=prediction,
                    groundtruth='\n'.join([f'{col}: {gt_data.iloc[idx][col]}' for col in gt_data.columns if col != 'description']),
                    passed_criteria=passed_criteria, failed_criteria=failed_criteria,
                    score=score, num_steps=num_steps, duration_s=task_duration,
                    trajectory=trajectory, prompt=prompt
                )

                temp_scores = scores.copy()
                temp_scores[catalog] += best_score
                temp_scores["total"] += best_score
                temp_nums = nums.copy()
                temp_nums[catalog] += 1
                temp_nums["total"] += 1

            except TimeoutError:
                task_duration = timer() - task_start_time
                logger.error(f"Loop {i} exceeded the time limit ({args.timeout}s) and was skipped")
                wb_logger.log_task_result(
                    idx=idx, task_index=task_index, catalog=catalog,
                    instruction=instruction, prediction="TIMEOUT",
                    groundtruth="", passed_criteria=[], failed_criteria=["TIMEOUT"],
                    score=0.0, num_steps=0, duration_s=task_duration
                )
                continue

            except Exception as e:
                task_duration = timer() - task_start_time
                logger.error(f"Task {idx} failed with error: {e}")
                wb_logger.log_task_result(
                    idx=idx, task_index=task_index, catalog=catalog,
                    instruction=instruction, prediction=f"ERROR: {str(e)}",
                    groundtruth="", passed_criteria=[], failed_criteria=[f"ERROR: {str(e)[:100]}"],
                    score=0.0, num_steps=0, duration_s=task_duration
                )
                continue

        scores = temp_scores
        nums = temp_nums

        # Log running scores
        wb_logger.log_running_scores(scores, nums)

    # Log final summary
    wb_logger.log_final_summary(scores, nums, dataset)

    # Print final results
    logger.info("=" * 60)
    logger.info(f"Final Results for {dataset}:")
    for key in ["easy", "middle", "hard", "total"]:
        if nums[key] > 0:
            logger.info(f"  {key}: {scores[key]}/{nums[key]} = {scores[key]/nums[key]:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":

    uid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    parser = argparse.ArgumentParser(description="OpenRCA Agent Standard Evaluation")
    parser.add_argument("--dataset", type=str, default="Market/cloudbed-1",
                        choices=["Market/cloudbed-1", "Market/cloudbed-2", "Bank", "Telecom"])
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=150)
    parser.add_argument("--controller_max_step", type=int, default=25)
    parser.add_argument("--controller_max_turn", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--tag", type=str, default='rca')
    parser.add_argument("--auto", action="store_true", default=False)
    # wandb arguments
    parser.add_argument("--wandb_project", type=str, default="OpenRCA")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=False,
                        help="Disable wandb logging")

    args = parser.parse_args()

    # Initialize wandb
    model_tag = configs['MODEL'].split('/')[-1]
    wandb_name = args.wandb_name or f"agent-{args.tag}-{model_tag}-{uid}"
    wb_logger = WandbLogger(
        project=args.wandb_project,
        run_name=wandb_name,
        config={
            "method": "rca_agent",
            "model": configs["MODEL"],
            "source": configs["SOURCE"],
            "tag": args.tag,
            "max_step": args.controller_max_step,
            "max_turn": args.controller_max_turn,
            "timeout": args.timeout,
            "sample_num": args.sample_num,
        },
        enabled=not args.no_wandb
    )
    wb_logger.log_config(configs, args)

    try:
        if args.auto:
            print(f"Auto mode is on. Model is fixed to {configs['MODEL']}")
            datasets = ["Market/cloudbed-1", "Market/cloudbed-2", "Bank", "Telecom"]
            for dataset in datasets:
                main(args, uid, dataset, wb_logger)
        else:
            dataset = args.dataset
            main(args, uid, dataset, wb_logger)
    finally:
        wb_logger.finish()
