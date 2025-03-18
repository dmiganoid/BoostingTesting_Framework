import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def parse_params_dict(param_str):
    if not isinstance(param_str, str):
        return {}
    try:
        return ast.literal_eval(param_str)
    except:
        return {}


def make_param_str(params_dict, max_len=100):
    items = [f"{k}={v}" for k, v in params_dict.items()]
    s = ", ".join(items)
    return s if len(s) <= max_len else s[:max_len] + "…"


def plot_mode(top_n=5, worst_n=5, top_k_per_algo=2):
    results_root = "results"
    if not os.path.exists(results_root):
        print("No results folder found.")
        return

    metrics = [
        ("test_accuracy", "Test Accuracy"),
        ("train_accuracy", "Train Accuracy"),
        ("train_time_sec", "Train Time (sec)"),
        ("inference_time_sec", "Inference Time (sec)"),
        ("memory_usage_mb", "Memory Usage (MB)"),
    ]
    for time_dir in os.listdir(results_root):
        time_dir_path = os.path.join(results_root, time_dir)

        if not os.path.isdir(time_dir_path):
            continue

        for test_name in os.listdir(time_dir_path):
            test_dir_path = os.path.join(time_dir_path, test_name)
            if not os.path.isdir(test_dir_path):
                continue
            csv_path = os.path.join(test_dir_path, "results.json")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, sep=",", index_col=0)
            if df.empty:
                continue
            df["model_params_dict"] = df["model_params"].apply(parse_params_dict)
            df["param_str"] = df["model_params_dict"].apply(
                lambda d: make_param_str(d, max_len=100)
            )
            plot_subdir = os.path.join(test_dir_path, "plots_split")
            os.makedirs(plot_subdir, exist_ok=True)
            algos = df["algorithm"].unique()

            for algo_name in algos:
                sub_df = df[df["algorithm"] == algo_name].copy()
                if sub_df.empty:
                    continue
                sub_df_sorted = sub_df.sort_values("test_accuracy", ascending=False)
                best_sub_df = sub_df_sorted.head(top_n).copy()
                worst_sub_df = sub_df_sorted.tail(worst_n).copy()
                for col, metric_title in metrics:
                    ascending = True if ("time" in col or "memory" in col) else False
                    temp_best = best_sub_df.sort_values(col, ascending=ascending)
                    plt.figure(figsize=(20, 8))
                    ax = plt.gca()
                    ax.set_axisbelow(True)
                    plt.grid(True, zorder=0)
                    sns.barplot(
                        data=temp_best,
                        x="algorithm",
                        y=col,
                        hue="param_str",
                        dodge=True,
                        width=0.8,
                        zorder=3,
                    )
                    plt.title(
                        f"{algo_name} (Top {top_n} by test_accuracy)\n{test_name}\n{metric_title}"
                    )
                    ax.set_xticks([])
                    ax.set_xlabel("")
                    if "accuracy" in col:
                        plt.yticks(np.linspace(0, 1, 21))
                    plt.legend(
                        bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters"
                    )
                    plt.tight_layout()
                    out_png = os.path.join(
                        plot_subdir, f"{algo_name}_{col}_best{top_n}.png"
                    )
                    plt.savefig(out_png, dpi=150)
                    plt.close()

                    temp_worst = worst_sub_df.sort_values(col, ascending=ascending)
                    plt.figure(figsize=(20, 8))
                    ax = plt.gca()
                    ax.set_axisbelow(True)
                    plt.grid(True, zorder=0)
                    sns.barplot(
                        data=temp_worst,
                        x="algorithm",
                        y=col,
                        hue="param_str",
                        dodge=True,
                        width=0.8,
                        zorder=3,
                    )
                    plt.title(
                        f"{algo_name} (Worst {worst_n} by test_accuracy)\n{test_name}\n{metric_title}"
                    )
                    ax.set_xticks([])
                    ax.set_xlabel("")
                    if "accuracy" in col:
                        plt.yticks(np.linspace(0, 1, 21))
                    plt.legend(
                        bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters"
                    )
                    plt.tight_layout()
                    out_png = os.path.join(
                        plot_subdir, f"{algo_name}_{col}_worst{worst_n}.png"
                    )
                    plt.savefig(out_png, dpi=150)
                    plt.close()

            df_sorted = df.sort_values("test_accuracy", ascending=False).copy()
            top_all = df_sorted.head(top_n)
            worst_all = df_sorted.tail(worst_n)
            plt.figure(figsize=(20, 8))
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.grid(True, zorder=0)
            sns.barplot(
                data=top_all,
                x="algorithm",
                y="test_accuracy",
                hue="param_str",
                dodge=True,
                width=0.8,
                zorder=3,
            )
            plt.title(f"Global Top {top_n} by test_accuracy\n{test_name}")
            ax.set_xticks([])
            ax.set_xlabel("")
            plt.yticks(np.linspace(0, 1, 21))
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters")
            plt.tight_layout()
            out_png = os.path.join(plot_subdir, f"global_top{top_n}_test_accuracy.png")
            plt.savefig(out_png, dpi=150)
            plt.close()

            plt.figure(figsize=(20, 8))
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.grid(True, zorder=0)
            sns.barplot(
                data=worst_all,
                x="algorithm",
                y="test_accuracy",
                hue="param_str",
                dodge=True,
                width=0.8,
                zorder=3,
            )
            plt.title(f"Global Worst {worst_n} by test_accuracy\n{test_name}")
            ax.set_xticks([])
            ax.set_xlabel("")
            plt.yticks(np.linspace(0, 1, 21))
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters")
            plt.tight_layout()
            out_png = os.path.join(
                plot_subdir, f"global_worst{worst_n}_test_accuracy.png"
            )
            plt.savefig(out_png, dpi=150)
            plt.close()

            list_rows = []
            for algo_name in algos:
                sub_df = df[df["algorithm"] == algo_name].copy()
                if sub_df.empty:
                    continue
                sub_df = sub_df.sort_values("test_accuracy", ascending=False)
                topX = sub_df.head(top_k_per_algo)
                list_rows.append(topX)
            comp_df = pd.concat(list_rows, ignore_index=True)
            plt.figure(figsize=(20, 8))
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.grid(True, zorder=0)
            sns.barplot(
                data=comp_df,
                x="algorithm",
                y="test_accuracy",
                hue="param_str",
                dodge=True,
                width=0.8,
                zorder=3,
            )
            plt.title(f"Comparison of best {top_k_per_algo} per algorithm\n{test_name}")
            plt.yticks(np.linspace(0, 1, 21))
            ax.set_xlabel("Algorithm")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Parameters")
            plt.tight_layout()
            out_png = os.path.join(
                plot_subdir, f"comparison_top{top_k_per_algo}_per_algo.png"
            )
            plt.savefig(out_png, dpi=150)
            plt.close()

            plt.figure(figsize=(24, 8))
            ax = plt.gca()
            ax.set_axisbelow(True)
            plt.grid(True, zorder=0)
            sns.barplot(
                data=comp_df,
                x="param_str",
                y="test_accuracy",
                hue="algorithm",
                dodge=True,
                width=0.8,
                zorder=3,
            )
            plt.title(f"Comparison (Algorithm in Legend)\n{test_name}")
            plt.yticks(np.linspace(0, 1, 21))
            plt.xticks(rotation=45, ha="right")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Algorithm")
            plt.tight_layout()
            out_png = os.path.join(plot_subdir, f"comparison_paramX_algoHue.png")
            plt.savefig(out_png, dpi=150)
            plt.close()

            params_to_compare = [
                "n_estimators",
                "learning_rate",
                "depth",
                "c",
                "convergence_criterion",
            ]
            for param in params_to_compare:
                df[param] = df["model_params_dict"].apply(
                    lambda d: d.get(param, np.nan)
                )
                if df[param].notna().sum() == 0:
                    continue
                plt.figure(figsize=(20, 8))
                ax = plt.gca()
                ax.set_axisbelow(True)
                plt.grid(True, zorder=0)
                sns.barplot(
                    data=df.dropna(subset=[param]),
                    x=param,
                    y="test_accuracy",
                    hue="algorithm",
                    dodge=True,
                    ci=None,
                    zorder=3,
                )
                plt.title(f"Test Accuracy vs {param}\n{test_name}")
                plt.xlabel(param)
                plt.ylabel("Test Accuracy")
                plt.yticks(np.linspace(0, 1, 21))
                plt.legend(
                    bbox_to_anchor=(1.05, 1), loc="upper left", title="Algorithm"
                )
                plt.tight_layout()
                out_png = os.path.join(plot_subdir, f"comparison_by_{param}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()

            train_file = os.path.join(test_dir_path, "train-dataset.csv")
            test_file = os.path.join(test_dir_path, "test-dataset.csv")
            if not (os.path.exists(train_file) and os.path.exists(test_file)):
                continue
            train_data = np.genfromtxt(train_file, delimiter=",")
            test_data = np.genfromtxt(test_file, delimiter=",")
            y_train = train_data[:, -1]
            y_test = test_data[:, -1]
            cm_df = pd.concat([top_all, worst_all]).drop_duplicates(
                subset=["algorithm", "file_postfix", "param_str"]
            )
            for i, row in cm_df.iterrows():
                postfix = row["file_postfix"]
                param_s = row["param_str"]
                pred_train_file = os.path.join(
                    test_dir_path, f"pred_train_{postfix}.csv"
                )
                pred_test_file = os.path.join(test_dir_path, f"pred_test_{postfix}.csv")
                if not (
                    os.path.exists(pred_train_file) and os.path.exists(pred_test_file)
                ):
                    continue
                y_train_pred = np.genfromtxt(pred_train_file, delimiter=",")
                y_test_pred = np.genfromtxt(pred_test_file, delimiter=",")
                cm_train = confusion_matrix(y_train, y_train_pred)
                disp = ConfusionMatrixDisplay(cm_train)
                disp.plot()
                plt.title(f"Train CM\n{param_s}\n{test_name}")
                plt.tight_layout()
                out_cm = os.path.join(plot_subdir, f"cm_train_{postfix}.png")
                plt.savefig(out_cm, dpi=150)
                plt.close()
                cm_test = confusion_matrix(y_test, y_test_pred)
                disp = ConfusionMatrixDisplay(cm_test)
                disp.plot()
                plt.title(f"Test CM\n{param_s}\n{test_name}")
                plt.tight_layout()
                out_cm = os.path.join(plot_subdir, f"cm_test_{postfix}.png")
                plt.savefig(out_cm, dpi=150)
                plt.close()
    print("Done.")
