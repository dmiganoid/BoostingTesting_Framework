import numpy as np
from models.trainer import BoostingBenchmarkTrainer
from sklearn.datasets import make_classification
import json
from os import mkdir
import os
from time import time
from models.trainer import load_algorithm
import pandas as pd
import argparse
import sys
import subprocess


def run_benchmark(cfg_file):
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", cfg_file
    )

    with open(config_path, "r") as file:
        configuration = json.load(file)

    print("=== Starting Boosting Benchmark ===")

    results_folder = f'{int(time())}-{cfg_file.split('.')[0]}'
    results_path = os.path.join('results', results_folder) 
    mkdir(results_path)

    algorithms_data = []
    for algorithm in configuration['algorithms']:
        algorithms_data.append(
            load_algorithm(
                algorithm=algorithm,
                algorithm_config=configuration['model'],
                base_estimator_cfg=configuration['estimator'],
                random_state=configuration['test']['random_state']
            )
        )
    trainer = BoostingBenchmarkTrainer(algorithms_data=algorithms_data)

    use_predefined = configuration['test'].get('use_predefined_datasets', False)

    if use_predefined:
        predefined_datasets = configuration['test'].get('predefined_datasets', [])

        if not predefined_datasets:
            for dataset in os.listdir('datasets'):
                predefined_datasets.append(dataset)

        for dataset_name in predefined_datasets:
            csv_full_path = os.path.join(os.path.dirname(__file__), "datasets", dataset_name)
            if not os.path.exists(csv_full_path):
                print(f"Not found: {csv_full_path}")
                continue
            data = np.genfromtxt(csv_full_path, delimiter=",")
            X, y = data[:, :-1], data[:, -1]

            test_name = f"{dataset_name}"

            trainer.fit_and_evaluate(
                X, y,
                test_size=configuration['test']['test_size'],
                random_state=configuration['test']['random_state'],
                results_path=results_path,
                test_name=test_name,
                multiprocessing= configuration['test'].get('multiprocessing', True)
            )

    N_synthetic_tests = configuration['test'].get('N_synthetic_tests', 3)
    synthetic_test_n_samples = configuration['test'].get('synthetic_test_n_samples', 1000)
    for i in range(N_synthetic_tests):
        X, y = make_classification(
            n_samples=synthetic_test_n_samples, class_sep=0.8
        )
        test_name = f"random-{i}"
        trainer.fit_and_evaluate(
            X, y,
            test_size=configuration['test']['test_size'],
            random_state=configuration['test']['random_state'],
            results_path=results_path,
            test_name=test_name,
            multiprocessing= configuration['test'].get('multiprocessing', True)
        )

    print("=== Benchmark Finished ===")
    return results_folder

def parse_results_to_df(json_path):
    with open(json_path, "r") as f:
        data = json.load(f) 
    rows = []
    for model_name, info in data.items():
        row = {
            "model": model_name,
            "accuracy": info["test_accuracy"], 
            "train_time_sec": info["train_time_sec"],
            "inference_time_sec": info["inference_time_sec"],
            "memory_usage_mb": info["memory_usage_mb"]
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main_cli():
    parser = argparse.ArgumentParser(description="BoostingTesting_Framework CLI")
    parser.add_argument("--mode", choices=["generate", "train", "plot", "trainplot"], required=True)
    parser.add_argument("--cfg")
    parser.add_argument("--plot_dirs", nargs='*', default=None, help="Список подпапок в results для построения графиков")
    parser.add_argument("--mppl", default="True")

    args = parser.parse_args()

    if args.mode == "generate":
        from utils.gen_synt_dataset import DataSetGenerator
        DataSetGenerator()

    elif args.mode == "train":
        cfg_file = args.cfg if args.cfg is not None else 'cfg.json'
        from main import run_benchmark
        run_benchmark(cfg_file)

    elif args.mode == "plot":
        from utils.plotting import plot_mode
        plot_mode(only_dirs=args.plot_dirs, multiprocessing=args.mppl)

    elif args.mode == "trainplot":
        cfg_file = args.cfg if args.cfg is not None else 'cfg.json'
        from main import run_benchmark
        results_folder = run_benchmark(cfg_file)
        from utils.plotting import plot_mode
        plot_mode(only_dirs=[results_folder], multiprocessing=int(args.mppl) if args.mppl.isdigit() else bool(args.mppl))
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
