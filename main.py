import argparse

from sklearn.model_selection import train_test_split

from data_loader import generate_synthetic_data
from models.model_configs import get_default_model_configs
from models.trainer import GBMBenchmarkTrainer
from utils.plotting import plot_benchmark_results


def main():
    parser = argparse.ArgumentParser(description="GBM Benchmark Framework")
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help='Тип задачи: classification или regression')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Доля выборки, отправляемая в тест.')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Число объектов (строк) для синтетического датасета.')
    parser.add_argument('--n_features', type=int, default=20,
                        help='Число признаков (столбцов) для синтетического датасета.')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Значение для воспроизводимости.')
    parser.add_argument('--plot_dir', type=str, default='plots',
                        help='Папка для сохранения графиков.')
    args = parser.parse_args()

    print("=== Запуск фреймворка GBM Benchmark ===")
    print(f"Task: {args.task}")
    print(f"Test size: {args.test_size}")
    print(f"Num samples: {args.n_samples}")
    print(f"Num features: {args.n_features}")
    print(f"Random state: {args.random_state}\n")

    X, y = generate_synthetic_data(
        task_type=args.task,
        n_samples=args.n_samples,
        n_features=args.n_features,
        random_state=args.random_state
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    model_configs = get_default_model_configs(
        task_type=args.task,
        random_state=args.random_state
    )

    trainer = GBMBenchmarkTrainer(
        model_configs=model_configs,
        task_type=args.task
    )

    results_df = trainer.fit_and_evaluate(X_train, y_train, X_test, y_test)
    print("=== Результаты бенчмарка ===")
    print(results_df)

    plot_benchmark_results(results_df, task_type=args.task, output_dir=args.plot_dir)
    print(f"Графики сохранены в папку: {args.plot_dir}")


if __name__ == "__main__":
    main()
