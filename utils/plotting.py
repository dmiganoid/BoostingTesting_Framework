# gbm_benchmark/utils/plotting.py

import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_benchmark_results(results_df, task_type, output_dir="plots"):
    """
    Принимает DataFrame вида:
      model | train_time_sec | memory_usage_mb | accuracy(или mse)
    и строит три бар-чарта:
      1) Метрика (accuracy или mse)
      2) Время обучения (train_time_sec)
      3) Память (memory_usage_mb)

    Результирующий график сохраняется в output_dir/benchmark_comparison.png
    """

    os.makedirs(output_dir, exist_ok=True)

    if task_type == 'classification':
        metric_name = 'accuracy'
        metric_title = 'Accuracy'
    else:
        metric_name = 'mse'
        metric_title = 'MSE'

    plt.figure(figsize=(18, 5))
    sns.set_style("whitegrid")

    # 1) Подграфик метрики
    plt.subplot(1, 3, 1)
    sns.barplot(data=results_df, x='model', y=metric_name, palette='Blues_r')
    plt.title(f"{metric_title} Comparison", fontsize=14)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # 2) Подграфик времени обучения
    plt.subplot(1, 3, 2)
    sns.barplot(data=results_df, x='model', y='train_time_sec', palette='Greens_r')
    plt.title("Training Time (sec)", fontsize=14)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # 3) Подграфик использования памяти
    plt.subplot(1, 3, 3)
    sns.barplot(data=results_df, x='model', y='memory_usage_mb', palette='Reds_r')
    plt.title("Memory Usage (MB)", fontsize=14)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # Сохраняем картинку
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=150)
    plt.close()
