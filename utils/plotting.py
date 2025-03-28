import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patches as mplpatches
sns.set_theme(style="whitegrid", palette="deep")
plt.rc("axes", titlesize=16, labelsize=14)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)


def compare_dicts(dict_1 : dict, dict_2 : dict, ignored : str):
    for key in dict_1.keys():
        if key != ignored and dict_1[key] != dict_2[key]:
            return False
    return True

def make_label(label:str) -> str:
    output = label.split("_")
    for ind, word in enumerate(output):
        if word not in ('sec', 'ms', 'n'):
            output[ind] = word[0].upper() + word[1:]
        else:
            output[ind] = f'({word})'
    return ' '.join(output)

def parse_params_dict(param_str):
    if not isinstance(param_str, str):
        return {}
    try:
        return ast.literal_eval(param_str)
    except:
        return {}

def make_param_str(params_dict):
    items = [f"{k}={v}" for k, v in params_dict.items()]
    return ", ".join(items)

def plot_mode(best_n=5, worst_n=5, best_k_per_algo=2):
    results_root = "results"
    if not os.path.exists(results_root):
        print("No results folder found.")
        return

    metrics = [
        "test_accuracy",
        "train_accuracy",
        "train_time_sec",
        "inference_time_sec",
        ]
    for time_dir in os.listdir(results_root):
        time_dir_path = os.path.join(results_root, time_dir)
        if not os.path.isdir(time_dir_path):
            continue

        for test_name in os.listdir(time_dir_path):
            test_dir_path = os.path.join(time_dir_path, test_name)
            if not os.path.isdir(test_dir_path):
                continue
            
            csv_path = os.path.join(test_dir_path, "results.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path, sep=",", index_col=0)
            if df.empty:
                continue

            df["model_params_dict"] = df["model_params"].apply(parse_params_dict)
            df["param_str"] = df["model_params_dict"].apply(make_param_str)
            
            plot_subdir = os.path.join(test_dir_path, "plots")
            os.makedirs(plot_subdir, exist_ok=True)
            other_plots_subdir = os.path.join(test_dir_path, "plots", "other")
            os.makedirs(other_plots_subdir, exist_ok=True)

            algorithms = df["algorithm"].unique()


            # PARAMETERS
            best_test_accuracy_models = df.loc[[df[df['algorithm'] == algorithm]['test_accuracy'].idxmax() for algorithm in algorithms]]
            best_test_accuracy_models.set_index('algorithm', inplace=True)
            best_train_accuracy_models = df.loc[[df[df['algorithm'] == algorithm]['train_accuracy'].idxmax() for algorithm in algorithms]]
            best_train_accuracy_models.set_index('algorithm', inplace=True)
            params_to_compare = ["n_estimators", "learning_rate", "c", "convergence_criterion", "delta", "epsilon", "tau", ]


            for algorithm in algorithms:
                ### line plots for the best test accuracy models
                for param in params_to_compare:
                    best_model = best_test_accuracy_models.loc[algorithm]
                    if param not in best_model["model_params_dict"].keys():
                        continue
                    x_data = []
                    y_test_data = []
                    y_train_data = []
                    for _, algorithm_data in df[df['algorithm']==algorithm].iterrows():
                        if compare_dicts(algorithm_data["model_params_dict"], best_model["model_params_dict"], param):
                            x_data.append(algorithm_data['model_params_dict'][param])
                            y_train_data.append(algorithm_data['train_accuracy'])
                            y_test_data.append(algorithm_data['test_accuracy'])

                    # sorting not required if config params are in non-descending order 
                    sort_ind = np.argsort(x_data, axis=0)
                    x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
                    y_train_data = np.take_along_axis(np.array(y_train_data), sort_ind, axis=0)
                    y_test_data = np.take_along_axis(np.array(y_test_data), sort_ind, axis=0)

                    fig = plt.figure(figsize=(12, 8))
                    plt.axis('off')
                    plt.title(f"Test Accuracy vs {param}")
                    gs = fig.add_gridspec(nrows=10, ncols=10)
                    axes = [fig.add_subplot(gs[0:9, :]), fig.add_subplot(gs[9:, :])]

                    axes[0].set_axisbelow(True)
                    axes[0].grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
                    axes[0].plot(x_data, y_train_data, 'r', marker="o", label="Train")
                    axes[0].plot(x_data, y_test_data, 'b', marker="o", label="Test")

                    if x_data.max() / x_data.min() > 100:
                        axes[0].set_xscale('log')
                    axes[0].set_xlabel(make_label(param))
                    axes[0].set_ylabel("Train Accuracy")
                    axes[0].set_ylim(bottom=y_test_data.min()*0.95, top=y_train_data.max()*1.05)
                    axes[0].legend()

                    textstr = f"{algorithm}: {make_param_str(best_model["model_params_dict"])}"
                    axes[1].text(0, .95, textstr, transform=axes[1].transAxes, linespacing=1.75, verticalalignment='top', horizontalalignment='left',)
                    axes[1].axis('off')

                    plt.tight_layout()
                    out_png = os.path.join(plot_subdir, f"{algorithm}-line_accuracy-vs-{param}.png")
                    plt.savefig(out_png, dpi=150)
                    plt.close()


                    sub_df = df[df["algorithm"] == algorithm]
                    sub_df_sorted = sub_df.sort_values("test_accuracy", ascending=False)
                    best_sub_df = sub_df_sorted.head(best_n).copy()
                    worst_sub_df = sub_df_sorted.tail(worst_n).copy()
                    for col in metrics:
                        ascending = True if ("time" in col) else False
                        
                        temp_best = best_sub_df.sort_values(col, ascending=ascending)
                        plt.figure(figsize=(20, 8))
                        ax = plt.gca()
                        ax.set_axisbelow(True)
                        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
                        sns.barplot(data=temp_best, x="algorithm", y=col, hue="param_str",
                                    dodge=True, width=0.8, zorder=3)
                        plt.title(f"{algorithm} (Top {best_n} by test_accuracy) - {make_label(col)}\n{test_name}")
                        ax.set_xticks([])
                        ax.set_xlabel("")
                        if "accuracy" in col:
                            plt.yticks(np.linspace(0, 1, 21))
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Parameters")
                        plt.tight_layout()
                        out_png = os.path.join(other_plots_subdir, f"{algorithm}_{col}_best{best_n}.png")
                        plt.savefig(out_png, dpi=150)
                        plt.close()


                        temp_worst = worst_sub_df.sort_values(col, ascending=ascending)
                        plt.figure(figsize=(20, 8))
                        ax = plt.gca()
                        ax.set_axisbelow(True)
                        plt.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
                        sns.barplot(data=temp_worst, x="algorithm", y=col, hue="param_str",
                                    dodge=True, width=0.8, zorder=3)
                        plt.title(f"{algorithm} (Worst {worst_n} by test_accuracy) - {make_label(col)}\n{test_name}")
                        ax.set_xticks([])
                        ax.set_xlabel("")
                        if "accuracy" in col:
                            plt.yticks(np.linspace(0, 1, 21))
                        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Parameters")
                        plt.tight_layout()
                        out_png = os.path.join(other_plots_subdir, f"{algorithm}_{col}_worst{worst_n}.png")
                        plt.savefig(out_png, dpi=150)
                        plt.close()            

            ### compare top test accuracy models from each class
            fig = plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.title("Best Models By Test Accuracy")
            gs = fig.add_gridspec(nrows=10, ncols=10)
            axes = [fig.add_subplot(gs[0:7, :]), fig.add_subplot(gs[7:, :])]

            axes[0].set_axisbelow(True)
            axes[0].grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
            sns.barplot(data=best_test_accuracy_models, x="algorithm", y="train_accuracy", color='darkblue',
                        zorder=3, ax=axes[0])
            sns.barplot(data=best_test_accuracy_models, x="algorithm", y="test_accuracy", color='lightblue',
                        zorder=4, ax=axes[0])
            axes[0].legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
            axes[0].set_xlabel("Algorithm")
            axes[0].set_ylabel("Test Accuracy")
            axes[0].set_ylim(bottom=0.95*best_test_accuracy_models['test_accuracy'].min(), top=1)

            textstr = '\n'.join([f"{algorithm}: {make_param_str(data['model_params_dict'])}" for algorithm, data in best_test_accuracy_models[['model_params_dict']].iterrows()])
            axes[1].text(0, .95, textstr, transform=axes[1].transAxes, linespacing=1.75, verticalalignment='top', horizontalalignment='left',)
            axes[1].axis('off')

            plt.tight_layout()
            out_png = os.path.join(plot_subdir, f"global_top_test_accuracy.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            

            ### compare top train accuracy models from each class
            fig = plt.figure(figsize=(12, 8))
            plt.axis('off')
            plt.title(f"Best Models By Train Accuracy")
            gs = fig.add_gridspec(nrows=10, ncols=10)
            axes = [fig.add_subplot(gs[0:7, :]), fig.add_subplot(gs[7:, :])]

            axes[0].set_axisbelow(True)
            axes[0].grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
            sns.barplot(data=best_train_accuracy_models, x="algorithm", y="train_accuracy", color='darkblue',
                        zorder=3, ax=axes[0])
            sns.barplot(data=best_train_accuracy_models, x="algorithm", y="test_accuracy", color='lightblue',
                        zorder=4, ax=axes[0])
            axes[0].legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
            axes[0].set_xlabel("Algorithm")
            axes[0].set_ylabel("Train Accuracy")
            axes[0].set_ylim(bottom=0.95*best_train_accuracy_models['test_accuracy'].min(), top=1)

            textstr = '\n'.join([f"{algorithm}: {make_param_str(data['model_params_dict'])}" for algorithm, data in best_train_accuracy_models[['model_params_dict']].iterrows()])
            axes[1].text(0, .95, textstr, transform=axes[1].transAxes, linespacing=1.75, verticalalignment='top', horizontalalignment='left',)
            axes[1].axis('off')
            
            plt.tight_layout()
            out_png = os.path.join(plot_subdir, f"global_top_train_accuracy.png")
            plt.savefig(out_png, dpi=150)
            plt.close()

            
            ### Confusion Matrices for top train accuracy and top test accuracy models from each class
            if os.path.exists(os.path.join(test_dir_path, 'pred')):
                train_file = os.path.join(test_dir_path, "train-dataset.csv")
                test_file = os.path.join(test_dir_path, "test-dataset.csv")
                if not (os.path.exists(train_file) and os.path.exists(test_file)):
                    continue
                train_data = np.genfromtxt(train_file, delimiter=",")
                test_data = np.genfromtxt(test_file, delimiter=",")
                y_train = train_data[:, -1]
                y_test = test_data[:, -1]
                for algorithm, algorithm_data in best_test_accuracy_models[["file_postfix", "param_str"]].iterrows():
                    postfix = algorithm_data["file_postfix"]
                    param_s = algorithm_data["param_str"]
                    pred_train_file = os.path.join(test_dir_path, 'pred', f"train_{postfix}.csv")
                    pred_test_file = os.path.join(test_dir_path, 'pred', f"test_{postfix}.csv")
                    if not (os.path.exists(pred_train_file) and os.path.exists(pred_test_file)):
                        continue
                    y_train_pred = np.genfromtxt(pred_train_file, delimiter=",")
                    cm_train = confusion_matrix(y_train, y_train_pred)
                    disp = ConfusionMatrixDisplay(cm_train)

                    fig = plt.figure(figsize=(12, 8))
                    plt.axis('off')
                    plt.title(f"{algorithm} Train Confusion Matrix")
                    gs = fig.add_gridspec(nrows=10, ncols=10)
                    axes = [fig.add_subplot(gs[0:9, :]), fig.add_subplot(gs[9:, :])]
                    axes[0].set_axisbelow(True)
                    disp.plot(ax=axes[0])
                    axes[0].grid(visible=False)

                    textstr = f"{algorithm}: {make_param_str(best_model["model_params_dict"])}"
                    axes[1].text(0, .95, textstr, transform=axes[1].transAxes, linespacing=1.75, verticalalignment='top', horizontalalignment='left',)
                    axes[1].axis('off')

                    plt.tight_layout()
                    out_cm = os.path.join(plot_subdir, f"{algorithm}_cm_train.png")
                    plt.savefig(out_cm, dpi=150)
                    plt.close()


                    y_test_pred = np.genfromtxt(pred_test_file, delimiter=",")
                    cm_test = confusion_matrix(y_test, y_test_pred)
                    disp = ConfusionMatrixDisplay(cm_test)

                    fig = plt.figure(figsize=(12, 8))
                    plt.axis('off')
                    plt.title(f"{algorithm} Test Confusion Matrix")
                    gs = fig.add_gridspec(nrows=10, ncols=10)
                    axes = [fig.add_subplot(gs[0:9, :]), fig.add_subplot(gs[9:, :])]
                    axes[0].set_axisbelow(True)
                    disp.plot(ax=axes[0])
                    axes[0].grid(visible=False)

                    textstr = f"{algorithm}: {make_param_str(best_model["model_params_dict"])}"
                    axes[1].text(0, .95, textstr, transform=axes[1].transAxes, linespacing=1.75, verticalalignment='top', horizontalalignment='left',)
                    axes[1].axis('off')

                    plt.tight_layout()
                    out_cm = os.path.join(plot_subdir, f"{algorithm}_cm_test.png")
                    plt.savefig(out_cm, dpi=150)
                    plt.close()
    print("Done.")