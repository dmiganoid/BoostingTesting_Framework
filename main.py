import numpy as np
from models.trainer import BoostingBenchmarkTrainer
from sklearn.datasets import make_classification
from utils.misc import parse_json_config
from os import mkdir
from time import time
from models.trainer import load_algorithm

def run_benchmark():
    configuration = parse_json_config("c:/Users/Need2BuySSD/Documents/GitHub/BoostingTesting_Framework/configs/cfg.json")
    
    print("=== Starting Boosting Benchmark ===")

    results_path = f'results/{time()}'
    mkdir(results_path)
    algorithms = []
    for algorithm in configuration['algorithms']:
        algorithms.append(load_algorithm(algorithm=algorithm, algorithm_config=configuration['model'], base_estimator_cfg=configuration['estimator']))

    trainer = BoostingBenchmarkTrainer(
            algorithms=algorithms
    )

    # Real datasets (not implemented)
    if False:
        if configuration['test']['realdatasets'] != "all":
            for dataset in configuration['test']['realdatasets']:
                data = np.genfromtxt(f"datasets/{dataset}.csv", delimiter=",")
                X, y = data[:,:-1], data[:, -1]
                trainer.fit_and_evaluate(
                    *train_test_split(X, y, test_size=configuration['test']['test_size'], random_state=configuration['test']['random_state']), 
                    test_name = f'test-random-{i}'
                )
        else:
            for dataset in os.listdir("datasets/"):
                data = np.genfromtxt(f"datasets/{dataset}", delimiter=",")
                X, y = data[:,:-1], data[:, -1]
                trainer.fit_and_evaluate(
                    *train_test_split(X, y, test_size=configuration['test']['test_size'], random_state=configuration['test']['random_state']),
                    test_name = f'test-random-{i}'
                )

    # Synthetic datasets
    for i in range(configuration['test']['N_synthetic_tests']):
        X, y = make_classification(
            n_samples=np.random.randint(1000, 1100)
        )
        trainer.fit_and_evaluate(
            X, y, test_size=configuration['test']['test_size'], random_state=configuration['test']['random_state'],
            test_name = f'{results_path}/test-random-{i}',
        )
    
    print("=== Boosting Benchmark Finished ===")

if __name__ == "__main__":
    run_benchmark()
