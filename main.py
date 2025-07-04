import numpy as np
from utils.trainer import BoostingBenchmarkTrainer
from sklearn.datasets import make_classification
import json
from os import mkdir, makedirs
import os
from time import time
from utils.loader import load_algorithm, load_metric
import pandas as pd
import argparse



def run_benchmark(cfg_file):
    """Run boosting algorithm benchmark with specified configuration.

    Loads configuration, initializes algorithms, and evaluates them on predefined or synthetic datasets.
    Saves results to a timestamped folder. Supports multiprocessing and customizable noise levels.

    Args:
        cfg_file (str): Name of the JSON configuration file.

    Returns:
        str: Name of the results folder where benchmark outputs are saved.
    """

    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs", cfg_file
    )

    with open(config_path, "r") as file:
        configuration = json.load(file)

    print("=== Starting Boosting Benchmark ===")

    makedirs("results", exist_ok=True)
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

    N_synthetic_tests = configuration['test'].get('N_synthetic_tests', 3)
    synthetic_test_n_samples = configuration['test'].get('synthetic_test_n_samples', 1000)
    use_predefined = configuration['test'].get('use_predefined_datasets', False)
    feature_noise = configuration['test'].get('feature_noise', 0)
    label_noise = configuration['test'].get('label_noise', 0)
    test_size = configuration['test'].get('test_size', 0.15)
    validation_size = configuration['test'].get('validation_size', 0.15)
    random_state = configuration['test'].get('random_state', 42)
    multiprocessing = configuration['test'].get('multiprocessing', True)
    N_retrain = configuration['test'].get('retrain', 1)
    skip_datasets = configuration['test'].get('skip_datasets', [])
    metric = load_metric(configuration['test'].get('metric', "accuracy"))
    use_class_weights = configuration['test'].get('use_class_weights', False)
    
    if use_predefined:
        predefined_datasets = configuration['test'].get('predefined_datasets', [])

        if not predefined_datasets:
            for dataset in os.listdir('datasets'):
                predefined_datasets.append(dataset)

        for dataset_name in predefined_datasets:
            if dataset_name in skip_datasets:
                continue
            csv_full_path = os.path.join(os.path.dirname(__file__), "datasets", dataset_name)
            if not os.path.exists(csv_full_path):
                print(f"Not found: {csv_full_path}")
                continue
            data = np.genfromtxt(csv_full_path, delimiter=",")
            X, y = data[:, :-1], data[:, -1]

            test_name = f"{dataset_name}"

            trainer.fit_and_evaluate(
                X, y,
                metric_function=metric, 
                use_class_weights=False,
                validation_size=validation_size,
                test_size=test_size,
                N_retrain = N_retrain,
                label_noise = label_noise,
                feature_noise = feature_noise,
                random_state=random_state,
                results_path=results_path,
                test_name=test_name,
                multiprocessing= multiprocessing
            )


    for i in range(N_synthetic_tests):
        X, y = make_classification(
            n_samples=synthetic_test_n_samples, class_sep=0.8
        )
        test_name = f"random-{i}"
        trainer.fit_and_evaluate(
            X, y,
            metric_function=metric, 
            use_class_weights=False,
            validation_size=validation_size,
            test_size=test_size,
            N_retrain = N_retrain,
            label_noise = label_noise,
            feature_noise = feature_noise,
            random_state=random_state,
            results_path=results_path,
            test_name=test_name,
            multiprocessing= multiprocessing
        )

    print("=== Benchmark Finished ===")
    return results_folder


def main_cli():
    """Command-line interface for BoostingTesting_Framework.

    Parses arguments in one of the execute modes: generate synthetic dataset, train model,
    plot results, or train and then plot. Supports configuration file and multiprocessing options.
    """

    parser = argparse.ArgumentParser(description="BoostingTesting_Framework CLI")
    parser.add_argument("--mode", choices=["generate", "train", "plot", "trainplot"], required=True)
    parser.add_argument("--cfg")
    parser.add_argument("--dirs", nargs='*', default=None, help="Список подпапок в results для построения графиков")
    parser.add_argument("--mp", default="True")

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
        plot_mode(only_dirs=args.dirs, multiprocessing=int(args.mp) if args.mp.isdigit() else False)

    elif args.mode == "trainplot":
        cfg_file = args.cfg if args.cfg is not None else 'cfg.json'
        from main import run_benchmark
        results_folder = run_benchmark(cfg_file)
        from utils.plotting import plot_mode
        plot_mode(only_dirs=[results_folder], multiprocessing=int(args.mp) if args.mp.isdigit() else False)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()
