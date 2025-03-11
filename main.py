from sklearn.model_selection import train_test_split
import numpy as np
from models.trainer import BoostingBenchmarkTrainer
from sklearn.datasets import make_classification
from utils.misc import parse_json_config
import os

def run_benchmark():
    configuration = parse_json_config("c:/Users/Need2BuySSD/Documents/GitHub/BoostingTesting_Framework/cfg.json")
    selected_classes = []
    if "GradientBoosting" in configuration['algorithms']:
        from sklearn.ensemble import GradientBoostingClassifier
        selected_classes.append(GradientBoostingClassifier)

    if "AdaBoost" in configuration['algorithms']:
        from sklearn.ensemble import AdaBoostClassifier
        selected_classes.append(AdaBoostClassifier)

    # base estimator initialization
    match configuration['test']['estimator']:
        case "stump":
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(**configuration['test']['estimator_params'])
    
        
    
    print("=== Starting Boosting Benchmark ===")
    trainer = BoostingBenchmarkTrainer(
        base_estimator=base_estimator,
        algorithms=selected_classes,
        algorithm_configs=configuration['model']
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
            n_samples=np.random.randint(1000, 200000)
        )
        trainer.fit_and_evaluate(
            *train_test_split(X, y, test_size=configuration['test']['test_size'], random_state=configuration['test']['random_state']), 
            test_name = f'test-random-{i}'
        )

if __name__ == "__main__":
    run_benchmark()
