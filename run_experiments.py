import os
import yaml
from sklearn.model_selection import train_test_split
from src.data_loader import generate_synthetic_data, load_data_from_csv
from src.models.model_configs import build_model_config_map
from src.models.trainer import GBMBenchmarkTrainer
from src.utils.plotting import plot_benchmark_results
from src.utils.metadata import save_experiment_metadata
from src.models.model_configs import make_model_config

def run_experiments():
    with open("configs/base_config.yaml", "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    with open("configs/scenario_experiments.yaml", "r", encoding="utf-8") as f:
        scenario_cfg = yaml.safe_load(f)

    method_map = build_model_config_map(base_cfg["methods"])

    results_dir = base_cfg["output_settings"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    for exp in scenario_cfg["experiments"]:
        exp_name = exp["name"]

        data_info = exp["data"]

        if data_info["type"] == "synthetic":
            X, Y = generate_synthetic_data(data_info)
            X_train, X_test, y_train, y_test = train_test_split(X, Y)
        elif data_info["type"] == "csv":
            X_train, y_train = load_data_from_csv(data_info["train_path"])

        method_names = exp["methods"]

        model_configs_for_exp = {}
        for m in method_names:
            if m in method_map:
                model_type, mparams = method_map[m]
                model_configs_for_exp[m] = make_model_config(model_type, mparams)
            else:
                print(f"Warning: method {m} not found in base_config! Skip...")

        trainer = GBMBenchmarkTrainer(model_configs_for_exp, task_type="classification")  

        results_df, trained_models = trainer.fit_and_evaluate(X_train, y_train, X_test, y_test)

        exp_output_dir = os.path.join(results_dir, exp_name)
        os.makedirs(exp_output_dir, exist_ok=True)

        csv_path = os.path.join(exp_output_dir, "results.csv")
        results_df.to_csv(csv_path, index=False)

        plot_benchmark_results(results_df, task_type="classification", output_dir=exp_output_dir)

        metadata_path = os.path.join(exp_output_dir, "metadata.yaml")
        save_experiment_metadata(metadata_path, exp, results_df)

        print(f"Finished experiment: {exp_name}, results in {exp_output_dir}")


if __name__ == "__main__":
    run_experiments()
