import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_benchmark_results(results_df, task_type, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    if task_type == 'classification':
        metric_name = 'accuracy'
        metric_title = 'Accuracy'
    else:
        metric_name = 'mse'
        metric_title = 'MSE'

    plt.figure(figsize=(18, 5))
    sns.set_style("whitegrid")

    plt.subplot(1, 4, 1)
    sns.barplot(data=results_df, x='model', y=metric_name, palette='Blues_r')
    plt.title(f"{metric_title} Comparison", fontsize=14)
    plt.xticks(rotation=25, ha='right')

    plt.subplot(1, 4, 2)
    sns.barplot(data=results_df, x='model', y='train_time_sec', palette='Greens_r')
    plt.title("Training Time (sec)", fontsize=14)
    plt.xticks(rotation=25, ha='right')
    
    plt.subplot(1, 4, 3)
    sns.barplot(data=results_df, x='model', y='inference_time_sec', palette='Greens_r')
    plt.title("Inference Time (sec)", fontsize=14)
    plt.xticks(rotation=25, ha='right')

    plt.subplot(1, 4, 4)
    sns.barplot(data=results_df, x='model', y='memory_usage_mb', palette='Reds_r')
    plt.title("Memory Usage (MB)", fontsize=14)
    plt.xticks(rotation=25, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=150)
    plt.close()


def plot_roc_curves(
        trained_models_dict,
        X_test,
        y_test,
        output_dir="plots"
    ):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models_dict.items():
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.2f})")
            except:
                pass

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()


def plot_decision_boundaries_2d(
        trained_models_dict,
        X_train,
        y_train,
        output_dir="plots"
    ):

    if X_train.shape[1] != 2:
        print("Для построения decision boundaries нужно ровно 2 признака.")
        return

    os.makedirs(output_dir, exist_ok=True)


    X = X_train.to_numpy()
    y = y_train.to_numpy()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    fig, axes = plt.subplots(
        1, len(trained_models_dict),
        figsize=(5 * len(trained_models_dict), 4),
        squeeze=False
    )
    axes = axes.ravel()

    for idx, (model_name, model) in enumerate(trained_models_dict.items()):
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axes[idx].contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="coolwarm")
        axes[idx].set_title(model_name)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "decision_boundaries.png"), dpi=150)
    plt.close()
