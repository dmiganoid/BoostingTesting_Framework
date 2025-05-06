import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.patches as mplpatches
from matplotlib import cm
from psutil import cpu_count
from multiprocessing import Pool


sns.set_theme(style="whitegrid", palette="deep")
plt.rc("axes", titlesize=16, labelsize=14)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)


def compare_dicts(dict_1 : dict, dict_2 : dict, ignored : list):
    for key in dict_1.keys():
        if key not in ignored and dict_1[key] != dict_2[key]:
            return False
    return True

def make_label(label:str) -> str:
    output = label.split("_")
    for ind, word in enumerate(output):
        if word not in ('sec', 'ms'):
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

def plot_mode(only_dirs=None, multiprocessing=True):
    results_root = "results"
    if not os.path.exists(results_root):
        print("No results folder found.")
        return

    if only_dirs is not None and len(only_dirs) > 0:
        top_level_dirs = [d for d in only_dirs if os.path.isdir(os.path.join(results_root, d))]
    else:
        top_level_dirs = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]

    metrics = ["test_accuracy","train_accuracy","train_time_sec","inference_time_sec"]

    for time_dir in top_level_dirs:
        time_dir_path = os.path.join(results_root, time_dir)
        if not os.path.isdir(time_dir_path):
            continue


        if multiprocessing:
            cpus = cpu_count(logical=False)
            if type(multiprocessing) is int:
                cpus = multiprocessing
            with Pool(processes=cpus) as pool:
                for test_name in os.listdir(time_dir_path):
                    test_dir_path = os.path.join(time_dir_path, test_name)
                    if not os.path.isdir(test_dir_path):
                        continue
                    
                    csv_path = os.path.join(test_dir_path, "results.csv")
                    if not os.path.exists(csv_path):
                        continue
                    pool.apply_async(plot_results, args=[csv_path, test_dir_path, metrics])
                pool.close()
                pool.join()
                print("=== Finished plots ===")
        else: # NOT TESTED
            for test_name in os.listdir(time_dir_path):
                test_dir_path = os.path.join(time_dir_path, test_name)
                if not os.path.isdir(test_dir_path):
                    continue
                
                csv_path = os.path.join(test_dir_path, "results.csv")
                if not os.path.exists(csv_path):
                    continue
                plot_results(csv_path, test_dir_path, metrics)
                print(f"== Finished plots for {test_name} ==")

            print("== Finished plots ==")


def plot_results(csv_path, test_dir_path, metrics):

    df = pd.read_csv(csv_path, sep=",")
    if df.empty or "algorithm" not in df.columns:
        return

    if "model_params" in df.columns:
        df["model_params_dict"] = df["model_params"].apply(parse_params_dict)
    else:
        df["model_params_dict"] = [{}]*len(df)

    plot_subdir = os.path.join(test_dir_path, "plots")
    os.makedirs(plot_subdir, exist_ok=True)
    other_plots_subdir = os.path.join(plot_subdir, "other")
    os.makedirs(other_plots_subdir, exist_ok=True)

    algorithms = df["algorithm"].unique()

    # PARAMETERS
    best_test_accuracy_models = df.loc[[df[df['algorithm'] == algorithm]['test_accuracy'].idxmax() for algorithm in algorithms]]
    best_test_accuracy_models.set_index('algorithm', inplace=True)
    best_train_accuracy_models = df.loc[[df[df['algorithm'] == algorithm]['train_accuracy'].idxmax() for algorithm in algorithms]]
    best_train_accuracy_models.set_index('algorithm', inplace=True)
    params_to_compare = ["n_estimators", "learning_rate", "c", "delta", "epsilon", "tau",'iterations' ]


    for algorithm in algorithms:
        best_model = best_test_accuracy_models.loc[algorithm]
        df_algorithm = df[df["algorithm"] == algorithm]
        ### line plots for the best test accuracy models
        for param in params_to_compare:
            if param not in best_model["model_params_dict"].keys():
                continue
            x_data = []
            y_test_data = []
            y_train_data = []
            for _, algorithm_data in df_algorithm.iterrows():
                if compare_dicts(algorithm_data["model_params_dict"], best_model["model_params_dict"], [param]):
                    x_data.append(algorithm_data['model_params_dict'][param])
                    y_train_data.append(algorithm_data['train_accuracy'])
                    y_test_data.append(algorithm_data['test_accuracy'])

            sort_ind = np.argsort(x_data, axis=0)
            x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
            y_train_data = np.take_along_axis(np.array(y_train_data), sort_ind, axis=0)
            y_test_data = np.take_along_axis(np.array(y_test_data), sort_ind, axis=0)
            if x_data.shape[0] < 2:
                continue
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            plt.title(f"Test Accuracy vs {param}")

            ax.set_axisbelow(True)
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
            ax.plot(x_data, y_train_data, 'r', marker="o", label="Train")
            ax.plot(x_data, y_test_data, 'b', marker="o", label="Test")

            if x_data.max() / x_data.min() > 100:
                ax.set_xscale('log')
            ax.set_xlabel(make_label(param))
            ax.set_ylabel("Train Accuracy")

            y_min = np.min((y_test_data, y_train_data))
            y_max = np.max((y_test_data, y_train_data))
            y_range = y_max - y_min

            ax.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)
            ax.legend()

            plt.tight_layout()
            out_png = os.path.join(plot_subdir, f"{algorithm}-line_accuracy-vs-{param}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()

        # <needs work> 3d plot test_accuracy vs (learning_rate x n_estimators) for best_models
        for i in range(len(params_to_compare)-1):
            for j in range(i+1, len(params_to_compare)):
                param_1 = params_to_compare[i]
                param_2 = params_to_compare[j]
                if param_1 not in best_model["model_params_dict"].keys() or param_2 not in best_model["model_params_dict"].keys():
                    continue
                plot_data = []
                for _, algorithm_data in df_algorithm.iterrows():
                    if compare_dicts(algorithm_data["model_params_dict"], best_model["model_params_dict"], 
                                    [param_1,
                                    param_2]):
                        plot_data.append(np.array([algorithm_data['model_params_dict'][param_1], algorithm_data['model_params_dict'][param_2], algorithm_data['train_accuracy'], algorithm_data['test_accuracy']]))
                plot_data = np.array(plot_data)
                X = np.unique(plot_data[:, 0])
                X.sort()
                Y = np.unique(plot_data[:, 1])
                Y.sort()
                if X.shape[0] < 3 or Y.shape[0] < 3:
                    continue
                Z_train = np.zeros((Y.shape[0], X.shape[0]))
                Z_test = np.zeros((Y.shape[0], X.shape[0]))
                for point in plot_data:
                    Z_train[np.where(Y==point[1]), np.where(X == point[0])] = point[2]
                    Z_test[np.where(Y==point[1]), np.where(X == point[0])] = point[3]
                x_logscale=False
                y_logscale=False
                if np.abs(np.log2(X[0] / X[-1])) > 5:
                    X = np.log2(X)
                    x_logscale=True
                if np.abs(np.log2(Y[0] / Y[-1])) > 5:
                    Y = np.log2(Y)
                    y_logscale=True
                
                X, Y = np.meshgrid(X, Y)
                fig, ax = plt.subplots(figsize=(12,8), subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(X, Y, Z_test,
                                    linewidth=0, cmap=cm.coolwarm)
                plt.xticks(ticks=plt.xticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.xticks()[0]])
                plt.yticks(ticks=plt.yticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.yticks()[0]])
                plt.xlabel(make_label(param_1))
                plt.ylabel(make_label(param_2))
                plt.title(f"Test Accuracy vs {make_label(param_1)} x {make_label(param_2)}")
                out_png = os.path.join(plot_subdir, f"{algorithm}-3d-test_accuracy-vs-{param_1}-x-{param_2}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()

                fig, ax = plt.subplots(figsize=(12,8), subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(X, Y, Z_test,
                                    linewidth=0, cmap=cm.coolwarm)
                plt.xticks(ticks=plt.xticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.xticks()[0]])
                plt.yticks(ticks=plt.yticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.yticks()[0]])
                ax.view_init(elev=0, azim=45)
                plt.xlabel(make_label(param_1))
                plt.ylabel(make_label(param_2))
                plt.title(f"Projection of Test Accuracy vs {make_label(param_1)} x {make_label(param_2)}")
                out_png = os.path.join(plot_subdir, f"{algorithm}-3d-proj-test_accuracy-vs-{param_1}-x-{param_2}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()

                fig, ax = plt.subplots(figsize=(12,8), subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(X, Y, Z_train,
                                    linewidth=0, cmap=cm.coolwarm)
                plt.xticks(ticks=plt.xticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.xticks()[0]])
                plt.yticks(ticks=plt.yticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.yticks()[0]])
                plt.xlabel(make_label(param_1))
                plt.ylabel(make_label(param_2))
                plt.title(f"Train Accuracy vs {make_label(param_1)} x {make_label(param_2)}")
                out_png = os.path.join(plot_subdir, f"{algorithm}-3d-train_accuracy-vs-{param_1}-x-{param_2}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()

                fig, ax = plt.subplots(figsize=(12,8), subplot_kw={"projection": "3d"})
                surf = ax.plot_surface(X, Y, Z_train,
                                    linewidth=0, cmap=cm.coolwarm)
                ax.view_init(elev=0, azim=45)
                plt.title(f"Projection of Train Accuracy vs {make_label(param_1)} x {make_label(param_2)}")
                plt.tight_layout()
                out_png = os.path.join(plot_subdir, f"{algorithm}-3d-proj-train_accuracy-vs-{param_1}-x-{param_2}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()


    ### compare top test accuracy models from each class
    fig = plt.figure(figsize=(16, 10))
    axis = plt.gca()
    plt.title("Best Models By Test Accuracy")
    axis.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    sns.barplot(data=best_test_accuracy_models, x="algorithm", y="train_accuracy", color='darkblue',
                zorder=3, ax=axis)
    sns.barplot(data=best_test_accuracy_models, x="algorithm", y="test_accuracy", color='lightblue',
                zorder=4, ax=axis)
    axis.legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
    axis.set_xlabel("")
    axis.set_ylabel("Test Accuracy")
    axis.tick_params('x', rotation=20)
    y_min = np.min([best_test_accuracy_models['train_accuracy'], best_test_accuracy_models['test_accuracy']])
    y_max = np.max([best_test_accuracy_models['train_accuracy'], best_test_accuracy_models['test_accuracy']])
    y_range = y_max - y_min
    axis.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f"global_top_test_accuracy.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    

    ### compare top train accuracy models from each class
    fig = plt.figure(figsize=(16, 10))
    axis = plt.gca()
    plt.title(f"Best Models By Train Accuracy")
    axis.set_axisbelow(True)
    axis.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    sns.barplot(data=best_train_accuracy_models, x="algorithm", y="train_accuracy", color='darkblue',
                zorder=3, ax=axis)
    sns.barplot(data=best_train_accuracy_models, x="algorithm", y="test_accuracy", color='lightblue',
                zorder=4, ax=axis)
    axis.legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
    axis.tick_params('x', rotation=20)
    axis.set_xlabel("")
    axis.set_ylabel("Train Accuracy") 
    y_min = np.min([best_train_accuracy_models['train_accuracy'], best_train_accuracy_models['test_accuracy']])
    y_max = np.max([best_train_accuracy_models['train_accuracy'], best_train_accuracy_models['test_accuracy']])
    y_range = y_max - y_min
    axis.set_ylim(bottom=y_min - 0.1*y_range, 
                        top=y_max + 0.1*y_range)
    
    out_png = os.path.join(plot_subdir, f"global_top_train_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    ### save best models params
    train_params_str = [f"{algorithm}: {make_param_str(data['model_params_dict'])}" for algorithm, data in best_train_accuracy_models[['model_params_dict']].iterrows()]
    test_params_str = [f"{algorithm}: {make_param_str(data['model_params_dict'])}" for algorithm, data in best_test_accuracy_models[['model_params_dict']].iterrows()]

    with open(os.path.join(plot_subdir, f"global_top_params.csv"), "w") as f:
        f.write("Top Train Models:\n")
        for st in train_params_str:
            f.write(f"{st}\n")
        f.write("\nTop Test Models:\n")
        for st in test_params_str:
            f.write(f"{st}\n")
        




    ### Train test scatter plots for top train accuracy and top test accuracy models from each class
    pred_dir=os.path.join(test_dir_path,"pred")
    
    if os.path.isdir(pred_dir):
        train_file = os.path.join(test_dir_path, "train-dataset.csv")
        test_file = os.path.join(test_dir_path, "test-dataset.csv")
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            return

        train_data = np.genfromtxt(train_file, delimiter=",")
        test_data = np.genfromtxt(test_file, delimiter=",")

        y_train = train_data[:, -1]
        y_test = test_data[:, -1]
        X_train = train_data[:, :-1]
        X_test = test_data[:, :-1]

        if X_test.shape[1] == 2:
            for algo_,row_ in best_test_accuracy_models[["file_postfix","model_params","model_params_dict"]].iterrows():
                postfix=row_["file_postfix"]

                pred_train_file=os.path.join(pred_dir,f"train_{postfix}.csv")
                pred_test_file=os.path.join(pred_dir,f"test_{postfix}.csv")
                if not(os.path.exists(pred_train_file) and os.path.exists(pred_test_file)):
                    continue

                y_train_pred=np.genfromtxt(pred_train_file,delimiter=",")
                y_test_pred=np.genfromtxt(pred_test_file,delimiter=",")

                fig, axes = plt.subplots(1, 2, figsize=(24,8))
                fig.suptitle(f"{algo_} Train and Test Predictions")

                true_train_preds = np.where(y_train==y_train_pred)
                true_test_preds = np.where(y_test==y_test_pred)
                false_train_preds = np.where(y_train!=y_train_pred)
                false_test_preds =  np.where(y_test!=y_test_pred)

                axes[0].scatter(x=X_train[true_train_preds, 0], y=X_train[true_train_preds, 1], c=np.where(y_train_pred[true_train_preds], 'r', 'b'), marker='+', s=12, label="Correct")
                axes[0].scatter(x=X_train[false_train_preds, 0], y=X_train[false_train_preds, 1], c=np.where(y_train_pred[false_train_preds], 'darkred',  'darkblue'), marker='x', label="Incorrect")
                axes[0].grid(False)
                axes[0].set_title("Train")

                axes[1].scatter(x=X_test[true_test_preds, 0], y=X_test[true_test_preds, 1], c=np.where(y_test_pred[true_test_preds], 'r', 'b'), marker='+', s=12, label="Correct")
                axes[1].scatter(x=X_test[false_test_preds, 0], y=X_test[false_test_preds, 1], c=np.where(y_test_pred[false_test_preds],  'darkred', 'darkblue'), marker='x', label="Incorrect")
                axes[1].grid(False)
                axes[1].set_title("Test")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_subdir,f"{algo_}_train-test-preds.png"),dpi=150)
                plt.close()



if __name__ == "__main__":
    plot_mode()