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

def plot_3d_2params(df_algorithm, selected_model, param_1, param_2, algorithm, plot_subdir):
    xyz_data = []
    for _, algorithm_data in df_algorithm.iterrows():
        if compare_dicts(algorithm_data["model_params_dict"], selected_model["model_params_dict"], 
                        [param_1, param_2]):
            xyz_data.append(np.array([algorithm_data['model_params_dict'][param_1], algorithm_data['model_params_dict'][param_2], algorithm_data['mean_results_train_accuracy'], algorithm_data['mean_results_test_accuracy']]))
                
    xyz_data = np.array(xyz_data)
    X = np.unique(xyz_data[:, 0])
    X.sort()
    Y = np.unique(xyz_data[:, 1])
    Y.sort()
    if X.shape[0] < 3 or Y.shape[0] < 3:
        return
    Z_test = np.zeros((Y.shape[0], X.shape[0]))
    for point in xyz_data:
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
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "3d"})
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

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z_test,
                        linewidth=0, cmap=cm.coolwarm)
    if x_logscale:
        plt.xticks(ticks=plt.xticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.xticks()[0]])
    if y_logscale:
        plt.yticks(ticks=plt.yticks()[0], labels= ["$2^{"+str(int(i))+"}$" for i in plt.yticks()[0]])
    ax.view_init(elev=0, azim=225)
    plt.xlabel(make_label(param_1))
    plt.ylabel(make_label(param_2))
    plt.title(f"Projection of Test Accuracy vs {make_label(param_1)} x {make_label(param_2)}")
    out_png = os.path.join(plot_subdir, f"{algorithm}-3d-proj-test_accuracy-vs-{param_1}-x-{param_2}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_best_models_in_class(selected_models, plot_subdir, readable=True):
    fig = plt.figure(figsize=(10, 5))
    axis = plt.gca()
    plt.title("Best Models By Validation Accuracy")
    axis.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    X = np.arange(len(selected_models))  # bar positions

    # Plot settings
    bar_width = 0.4  # Half width for each half

    if readable:
        # Plot left halves (dataset 1)
        axis.bar(X - bar_width/2, [ data['worst_results_train_accuracy'] for _, data in selected_models.iterrows()], width=bar_width, color='darkblue', align='center')
        axis.bar(X - bar_width/2, [ data['worst_results_test_accuracy'] for _, data in selected_models.iterrows()], width=bar_width, color='lightblue', align='center')

        # Plot right halves (dataset 2)
        axis.bar(X + bar_width/2, [ data['best_results_train_accuracy'] for _, data in selected_models.iterrows()], width=bar_width, color='darkblue', align='center')
        axis.bar(X + bar_width/2, [ data['best_results_test_accuracy'] for _, data in selected_models.iterrows()], width=bar_width, color='lightblue', align='center')

    else:
        worst_train = [ data['worst_results_train_accuracy'] for _, data in selected_models.iterrows()]
        worst_test = [ data['worst_results_test_accuracy'] for _, data in selected_models.iterrows()]
        best_train = [ data['best_results_train_accuracy'] for _, data in selected_models.iterrows()]
        best_test = [ data['best_results_test_accuracy'] for _, data in selected_models.iterrows()]

        for x in X:
            
            # Plot left halves (worst)
            if worst_train[x] > worst_test[x]:
                axis.bar(x - bar_width/2, worst_train[x], width=bar_width, color='darkblue', align='center')
                axis.bar(x - bar_width/2, worst_test[x], width=bar_width, color='lightblue', align='center')
            else:
                axis.bar(x - bar_width/2, worst_test[x], width=bar_width, color='lightblue', align='center')
                axis.bar(x - bar_width/2, worst_train[x], width=bar_width, color='darkblue', align='center')
            
            # Plot right halves (best)
            if best_train[x] > best_test[x]:
                axis.bar(x + bar_width/2, best_train[x], width=bar_width, color='darkblue', align='center')
                axis.bar(x + bar_width/2, best_test[x], width=bar_width, color='lightblue', align='center')
            else:
                axis.bar(x + bar_width/2, best_test[x], width=bar_width, color='lightblue', align='center')
                axis.bar(x + bar_width/2, best_train[x], width=bar_width, color='darkblue', align='center')


    axis.legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
    axis.set_xlabel("")
    axis.set_xticks(ticks=X, labels=[alg.replace('Classifier', '') for alg, data in selected_models.iterrows()])
    axis.set_ylabel("Test Accuracy")
    axis.tick_params('x', rotation=20)
    y_min = np.min([selected_models['worst_results_train_accuracy'], selected_models['worst_results_test_accuracy'], selected_models['best_results_train_accuracy'], selected_models['best_results_test_accuracy']])
    y_max = np.max([selected_models['worst_results_train_accuracy'], selected_models['worst_results_test_accuracy'], selected_models['best_results_train_accuracy'], selected_models['best_results_test_accuracy']])
    y_range = y_max - y_min
    axis.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.2*y_range)

    plt.tight_layout()
    if readable:
        out_png = os.path.join(plot_subdir, f"global_top_test_accuracy.png")
    else:
        out_png = os.path.join(plot_subdir, f"global_top_test_accuracy_unreadable.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_all_algorithms_lines_test_accuracy_param(df, algorithms, selected_models, plot_subdir):
    logscale = False
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    param = "n_estimators"
    for algorithm in algorithms:
        best_model = selected_models.loc[algorithm]
        df_algorithm = df[df["algorithm"] == algorithm]
        if param not in best_model["model_params_dict"].keys():
            continue
        x_data = []
        y_test_data = []
        for _, algorithm_data in df_algorithm.iterrows():
            if compare_dicts(algorithm_data["model_params_dict"], best_model["model_params_dict"], [param]):
                x_data.append(algorithm_data['model_params_dict'][param])
                y_test_data.append(algorithm_data['mean_results_test_accuracy'])

        sort_ind = np.argsort(x_data, axis=0)
        x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
        y_test_data = np.take_along_axis(np.array(y_test_data), sort_ind, axis=0)
        if x_data.shape[0] < 2:
            continue
        logscale = (x_data.max() / x_data.min()) > 100
        ax.plot(x_data, 1-y_test_data, label=algorithm, linewidth=1.25)
        sc = ax.scatter(best_model['model_params_dict'][param], 1-best_model['mean_results_test_accuracy'], marker="x")
        ax.scatter(x_data[np.argmin(1-y_test_data)], (1-y_test_data).min(), marker="o", c=sc.get_facecolor())

    plt.title(f"Error Rate vs {make_label(param)}")

    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', linewidth=0.5, linestyle="--")

    if logscale:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel("Error Rate")
    leg = ax.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f"all_algorithms-error_rate-vs-{make_label(param)}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_best_models_params(selected_models, plot_subdir):
    test_params_str = [f"{algorithm}, {make_param_str(data['model_params_dict'])}" for algorithm, data in selected_models[['model_params_dict']].iterrows()]

    with open(os.path.join(plot_subdir, f"global_top_params.csv"), "w") as f:
        f.write("\nTop Test Models:\n")
        for st in test_params_str:
            f.write(f"{st}\n")

def plot_train_test_accuracy_vs_param(df_algorithm, selected_model, algorithm, param, plot_subdir):
    x_data = []
    y_test = []
    y_train = []
    y_major_test = []
    y_minor_test = []
    for _, algorithm_data in df_algorithm.iterrows():
        if compare_dicts(algorithm_data["model_params_dict"], selected_model["model_params_dict"], [param]):
            x_data.append(algorithm_data['model_params_dict'][param])
            y_train.append(algorithm_data['mean_results_train_accuracy'])
            y_test.append(algorithm_data['mean_results_test_accuracy'])
            y_major_test.append(algorithm_data['mean_results_major_class_test_accuracy'])
            y_minor_test.append(algorithm_data['mean_results_minor_class_test_accuracy'])
    sort_ind = np.argsort(x_data, axis=0)
    x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
    y_train_data = np.take_along_axis(np.array(y_train), sort_ind, axis=0)
    y_test_data = np.take_along_axis(np.array(y_test), sort_ind, axis=0)
    if x_data.shape[0] < 2:
        return
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.title(f"Train and test accuracy vs {make_label(param)}")

    ax.set_axisbelow(True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
    ax.plot(x_data, y_train_data, 'r', marker="o", label="Train")
    ax.plot(x_data, y_test_data, 'b', marker="o", label="Test")

    if x_data.max() / x_data.min() > 100:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel("Accuracy")

    y_min = np.min((y_test_data, y_train_data))
    y_max = np.max((y_test_data, y_train_data))
    y_range = y_max - y_min

    ax.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f"{algorithm}-line_accuracy-vs-{param}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_major_minor_accuracy_vs_param(df_algorithm, selected_model, algorithm, param, plot_subdir):
    x_data = []
    y_major_test = []
    y_minor_test = []
    for _, algorithm_data in df_algorithm.iterrows():
        if compare_dicts(algorithm_data["model_params_dict"], selected_model["model_params_dict"], [param]):
            x_data.append(algorithm_data['model_params_dict'][param])
            y_major_test.append(algorithm_data['mean_results_major_class_test_accuracy'])
            y_minor_test.append(algorithm_data['mean_results_minor_class_test_accuracy'])
    sort_ind = np.argsort(x_data, axis=0)
    x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
    y_major_test_data = np.take_along_axis(np.array(y_major_test), sort_ind, axis=0)
    y_minor_test_data = np.take_along_axis(np.array(y_minor_test), sort_ind, axis=0)
    if x_data.shape[0] < 2:
        return
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.title(f"Test accuracy on major and minor class vs {make_label(param)}")

    ax.set_axisbelow(True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
    ax.plot(x_data, y_minor_test_data, 'r', marker="o", label="Accuracy on minor class")
    ax.plot(x_data, y_major_test_data, 'b', marker="o", label="Accuracy on major class")

    if x_data.max() / x_data.min() > 100:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel("Accuracy")

    y_min = np.min((y_minor_test_data, y_major_test_data))
    y_max = np.max((y_minor_test_data, y_major_test_data))
    y_range = y_max - y_min

    ax.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f"{algorithm}-line_class_accuracy-vs-{param}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_dataset(test_dir_path, datasets_dir_path, dataset_name, plot_subdir):
    pred_dir=os.path.join(test_dir_path,"pred")
    
    if os.path.isdir(pred_dir):
        dataset_file = os.path.join(datasets_dir_path, dataset_name)
        if not (os.path.exists(dataset_file)):
            return
        dataset = np.genfromtxt(dataset_file, delimiter=",")
        X = dataset[:,:-1]
        y = dataset[:,-1]
        if X.shape[1] != 2:
            return
        fig = plt.figure(figsize=(10,5))
        axis=plt.gca()

        axis.scatter(x=X[:, 0], y=X[:,1], c=np.where(y==+1, 'r', 'b'), marker='.', s=12)
        axis.grid(False)
        axis.set_xticks([])
        axis.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_subdir,f"dataset.png"),dpi=150)
        plt.close()

def plot_model_preds_scatter(selected_models, test_dir_path, datasets_dir_path, dataset_name, plot_subdir):
    pred_dir=os.path.join(test_dir_path,"pred")
    
    if os.path.isdir(pred_dir):
        dataset_file = os.path.join(datasets_dir_path, dataset_name)
        if not (os.path.exists(dataset_file)):
            return
        dataset = np.genfromtxt(dataset_file, delimiter=",")
        X = dataset[:,:-1]
        y = dataset[:,-1]
        if X.shape[1] != 2:
            return

        for algo_,row_ in selected_models[["file_postfix","model_params","model_params_dict"]].iterrows():
            postfix=row_["file_postfix"]

            pred_file=os.path.join(pred_dir,f"{postfix}_pred.csv")
            if not(os.path.exists(pred_file)):
                continue

            pred=np.genfromtxt(pred_file,delimiter=",")

            fig = plt.figure(figsize=(10,5))
            axis = plt.gca()
            #fig.suptitle(f"{algo_} Predictions")
            
            true_preds = np.where(y==pred)
            false_preds =  np.where(y!=pred)

            axis.scatter(x=X[true_preds, 0], y=X[true_preds, 1], c=np.where(pred[true_preds], 'r', 'b'), marker='+', s=12)
            axis.scatter(x=X[false_preds, 0], y=X[false_preds, 1], c=np.where(pred[false_preds], 'darkred',  'darkblue'), marker='x')
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{algo_} Predictions")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_subdir,f"{algo_}_preds.png"),dpi=150)
            plt.close()

def plot_mode(only_dirs=None, multiprocessing=True):
    results_root = "results"
    datasets_root = "datasets"
    datasets_dir_path = os.path.join(datasets_root)

    if not os.path.exists(results_root):
        print("No results folder found.")
        return
    if only_dirs is not None and len(only_dirs) > 0:
        top_level_dirs = [d for d in only_dirs if os.path.isdir(os.path.join(results_root, d))]
    else:
        top_level_dirs = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]

    metrics = ["mean_results_test_accuracy","mean_results_train_accuracy","mean_results_train_time_sec","mean_results_inference_time_sec", "mean_results_major_class_test_accuracy", "mean_results_minor_class_test_accuracy"]
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
                    pool.apply_async(plot_results, args=[csv_path, test_dir_path, metrics], kwds={"datasets_dir_path" : datasets_dir_path, })
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
                plot_results(csv_path, test_dir_path, metrics, datasets_dir_path=datasets_dir_path)
                print(f"== Finished plots for {test_name} ==")

            print("== Finished plots ==")


def plot_results(csv_path,test_dir_path, metrics,  datasets_dir_path=None):
    dataset_name = os.path.basename(test_dir_path)

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
    ### mean_results -> best_results
    best_validation_accuracy_models = df.loc[[df[df['algorithm'] == algorithm]['bestvalid_results_validation_accuracy'].idxmax() for algorithm in algorithms]]
    best_validation_accuracy_models.set_index('algorithm', inplace=True)
    params_to_compare = ["n_estimators", "learning_rate", "c", "delta", "epsilon", "tau",'iterations' ]

    for algorithm in algorithms:
        best_model = best_validation_accuracy_models.loc[algorithm]
        df_algorithm = df[df["algorithm"] == algorithm]
        
        ### line plots for the best test accuracy models
        for param in params_to_compare:
            if param not in best_model["model_params_dict"].keys():
                continue

            # Train and Test Accuracy vs param
            plot_train_test_accuracy_vs_param(df_algorithm, best_model, algorithm, param, plot_subdir)

            # Major and Minor Accuracy vs param
            plot_major_minor_accuracy_vs_param(df_algorithm, best_model, algorithm, param, plot_subdir)

        # <needs work> 3d plot test_accuracy vs (learning_rate x n_estimators) for best validation model
        for i in range(len(params_to_compare)-1):
            for j in range(i+1, len(params_to_compare)):
                param_1 = params_to_compare[i]
                param_2 = params_to_compare[j]
                if param_1 not in best_model["model_params_dict"].keys() or param_2 not in best_model["model_params_dict"].keys():
                    continue
                plot_3d_2params(df_algorithm, best_model, param_1, param_2, algorithm, plot_subdir)


    ### compare top test accuracy models from each class
    plot_best_models_in_class(best_validation_accuracy_models, plot_subdir)
    plot_best_models_in_class(best_validation_accuracy_models, plot_subdir, readable=False)

    #  All algos Test Accuracy vs param
    plot_all_algorithms_lines_test_accuracy_param(df, algorithms, best_validation_accuracy_models, plot_subdir)

    ### Save best validation models params
    save_best_models_params(best_validation_accuracy_models, plot_subdir)
        
    ### Scatter plot for 2d datasets
    plot_dataset(test_dir_path, datasets_dir_path, dataset_name, plot_subdir)

    ### Scatter plots for best validation accuracy model from each class
    plot_model_preds_scatter(best_validation_accuracy_models, test_dir_path, datasets_dir_path, dataset_name, plot_subdir)



if __name__ == "__main__":
    plot_mode()