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


def ind_max(s : pd.Series, k : str) -> int:
    '''Find the last index of the maximum value associated with a given key in a Series of dictionaries.

    Parameters:
        s (pd.Series): A pandas Series where each element is a dictionary.
        k (str): The key whose associated value is used for comparison.

    Returns:
        The index of the last occurrence of the maximum value for the specified key.
    '''

    ind = -1
    maxval = 1e-16
    for i in s.keys():
        i_val = s[i][k]
        maxval = max(maxval, i_val)
        if i_val == maxval:
            ind = i
    return ind

def compare_dicts(dict_1 : dict, dict_2 : dict, ignored : list) -> bool:
    '''Compare two dictionaries for equality, ignoring specific keys.

    Parameters:
    dict_1 (dict): The first dictionary.
    dict_2 (dict): The second dictionary.
    ignored (list): A list of keys to ignore during comparison.

    Returns:
    bool: True if all non-ignored key-value pairs in dict_1 match those in dict_2; False otherwise.
    '''

    for key in dict_1.keys():
        if key not in ignored and dict_1[key] != dict_2[key]:
            return False
    return True

def make_label(label:str) -> str:
    '''Parse a string representation of a dictionary into an actual dictionary.

    Parameters:
        param_str (str): A string expected to represent a Python dictionary.

    Returns:
        dict: A dictionary parsed from the input string, or an empty dictionary if parsing fails or input is not a string.
    '''

    output = label.split('_')
    for ind, word in enumerate(output):
        if word not in ('sec', 'ms'):
            output[ind] = word[0].upper() + word[1:]
        else:
            output[ind] = f'({word})'
    return ' '.join(output)

def parse_params_dict(param_str):
    '''Convert a string representation of a dictionary into a dictionary object.

    Parameters:
        param_str (str): A string containing a Python-style dictionary.

    Returns:
        dict: The evaluated dictionary if parsing is successful and input is a string; otherwise, an empty dictionary.
    '''

    if not isinstance(param_str, str):
        return {}
    try:
        return ast.literal_eval(param_str)
    except:
        return {}

def make_param_str(params_dict : dict) -> str:
    '''Convert a dictionary into a formatted parameter string.

    Parameters:
        params_dict (dict): A dictionary of key-value pairs to format.

    Returns:
        str: A string where each key-value pair is represented as "key=value", separated by commas.
    '''

    items = [f'{k}={v}' for k, v in params_dict.items()]
    return ', '.join(items)

def plot_3d_2params(algorithm_results_df : pd.DataFrame, selected_model : pd.Series, param_1 : str, param_2 : str, plot_subdir : str):
    '''Plot and save 3D surface plots of test accuracy for two selected hyperparameters.

    Parameters:
        algorithm_results_df (pandas.DataFrame) A pandas.DataFrame containing model results.
        selected_model (pandas.Series): A pandas.Series containing the reference model's parameters, used to filter comparable results.
        param_1 (str): Name of the first hyperparameter to plot on the x-axis.
        param_2 (str): Name of the second hyperparameter to plot on the y-axis.
        plot_subdir (str): Directory path where the output plots will be saved.

    Returns:
        None: The function saves a plots as a PNG file and does not return a value.
    '''

    xyz_data = []
    for _, algorithm_data in algorithm_results_df.iterrows():
        if compare_dicts(algorithm_data['params_dict'], selected_model['params_dict'], 
                        [param_1, param_2]):
            xyz_data.append(np.array([algorithm_data['params_dict'][param_1], algorithm_data['params_dict'][param_2], algorithm_data['mean_results_dict']['train_metric'], algorithm_data['mean_results_dict']['test_metric']]))
                
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
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(X, Y, Z_test,
                        linewidth=0, cmap=cm.coolwarm)
    plt.xticks(ticks=plt.xticks()[0], labels= ['$2^{'+str(int(i))+'}$' for i in plt.xticks()[0]])
    plt.yticks(ticks=plt.yticks()[0], labels= ['$2^{'+str(int(i))+'}$' for i in plt.yticks()[0]])
    plt.xlabel(make_label(param_1))
    plt.ylabel(make_label(param_2))
    plt.title(f'Test Accuracy vs {make_label(param_1)} x {make_label(param_2)}')
    print(selected_model)
    out_png = os.path.join(plot_subdir, f'{selected_model["algorithm"]}-3d-test_metric-vs-{param_1}-x-{param_2}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': '3d'})
    surf = ax.plot_surface(X, Y, Z_test,
                        linewidth=0, cmap=cm.coolwarm)
    if x_logscale:
        plt.xticks(ticks=plt.xticks()[0], labels= ['$2^{'+str(int(i))+'}$' for i in plt.xticks()[0]])
    if y_logscale:
        plt.yticks(ticks=plt.yticks()[0], labels= ['$2^{'+str(int(i))+'}$' for i in plt.yticks()[0]])
    ax.view_init(elev=0, azim=225)
    plt.xlabel(make_label(param_1))
    plt.ylabel(make_label(param_2))
    plt.title(f'Projection of Test Accuracy vs {make_label(param_1)} x {make_label(param_2)}')
    out_png = os.path.join(plot_subdir, f'{selected_model["algorithm"]}-3d-proj-test_metric-vs-{param_1}-x-{param_2}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_best_worst_and_cases_for_selected_models_in_class(selected_models, plot_subdir, readable=True):
    '''Plot and save a bar chart comparing train and test accuracy of best and worst test case models for each algorithm.

    Parameters:
        selected_models (pandas.DataFrame):  A pandas.DataFrame containing results for selected models.
        plot_subdir (str): Directory path where the output plots will be saved.
        readable (bool): If True, generates a simplified (readable) version of the plot. Otherwise, uses a train and test bars will overlap in a way so both of them are visible.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    fig = plt.figure(figsize=(10, 5))
    axis = plt.gca()
    plt.title('Best Models By Validation Accuracy')
    axis.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=0)
    X = np.arange(len(selected_models))  # bar positions

    # Plot settings
    bar_width = 0.4  # Half width for each half
    worst_case_train = [ data['worst_test_metric_results_dict']['train_metric'] for _, data in selected_models.iterrows()]
    worst_case_test = [ data['worst_test_metric_results_dict']['test_metric'] for _, data in selected_models.iterrows()]
    best_case_train = [ data['best_test_metric_results_dict']['train_metric'] for _, data in selected_models.iterrows()]
    best_case_test = [ data['best_test_metric_results_dict']['test_metric'] for _, data in selected_models.iterrows()]
    if readable:
        # Plot left halves (dataset 1)
        axis.bar(X - bar_width/2, worst_case_train, width=bar_width, color='darkblue', align='center')
        axis.bar(X - bar_width/2, worst_case_test, width=bar_width, color='lightblue', align='center')

        # Plot right halves (dataset 2)
        axis.bar(X + bar_width/2, best_case_train, width=bar_width, color='darkblue', align='center')
        axis.bar(X + bar_width/2, best_case_test, width=bar_width, color='lightblue', align='center')

    else:
        for x in X:
            # Plot left halves (worst case)
            if worst_case_train[x] > worst_case_test[x]:
                axis.bar(x - bar_width/2, worst_case_train[x], width=bar_width, color='darkblue', align='center')
                axis.bar(x - bar_width/2, worst_case_test[x], width=bar_width, color='lightblue', align='center')
            else:
                axis.bar(x - bar_width/2, worst_case_test[x], width=bar_width, color='lightblue', align='center')
                axis.bar(x - bar_width/2, worst_case_train[x], width=bar_width, color='darkblue', align='center')
            
            # Plot right halves (best case)
            if best_case_train[x] > best_case_test[x]:
                axis.bar(x + bar_width/2, best_case_train[x], width=bar_width, color='darkblue', align='center')
                axis.bar(x + bar_width/2, best_case_test[x], width=bar_width, color='lightblue', align='center')
            else:
                axis.bar(x + bar_width/2, best_case_test[x], width=bar_width, color='lightblue', align='center')
                axis.bar(x + bar_width/2, best_case_train[x], width=bar_width, color='darkblue', align='center')


    axis.legend(handles=[mplpatches.Patch(color='darkblue', label='Train Accuracy'), mplpatches.Patch(color='lightblue', label='Test Accuracy')])
    axis.set_xlabel('')
    axis.set_xticks(ticks=X, labels=[data['algorithm'].replace('Classifier', '') for _, data in selected_models.iterrows()])
    axis.set_ylabel('Test Accuracy')
    axis.tick_params('x', rotation=20)
    y_min = np.min([worst_case_train, worst_case_test, best_case_train, best_case_test])
    y_max = np.max([worst_case_train, worst_case_test, best_case_train, best_case_test])
    y_range = y_max - y_min
    axis.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.2*y_range)
    str.replace
    plt.tight_layout()
    if readable:
        out_png = os.path.join(plot_subdir, f'best_params_best_worst_case_metric.png')
    else:
        out_png = os.path.join(plot_subdir, f'best_params_best_worst_case_metric_unreadable.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_all_algorithms_lines_test_metric_param(all_results_df, algorithms, selected_models, plot_subdir, param):
    '''Plot error rate against a specific hyperparameter (with other hyperparameters fixed) across multiple algorithms and save the result.

    Parameters:
        all_results_df (pandas.DataFrame): A pandas.DataFrame containing results for all models.
        selected_models (pandas.DataFrame): A pandas.DataFrame containing results for selected models.
        plot_subdir (str): Directory path where the output plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    algorithms = all_results_df['algorithm'].unique()
    logscale = False
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    
    for algorithm in algorithms:
        algorithm_results_df = all_results_df[all_results_df['algorithm'] == algorithm]
        if param not in selected_models['params_dict'].keys():
            continue
        x_data = []
        y_test_data = []
        for _, algorithm_data in algorithm_results_df.iterrows():
            if compare_dicts(algorithm_data['params_dict'], selected_models['params_dict'], [param]):
                x_data.append(algorithm_data['params_dict'][param])
                y_test_data.append(algorithm_data['mean_results_dict']['test_metric'])

        sort_ind = np.argsort(x_data, axis=0)
        x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
        y_test_data = np.take_along_axis(np.array(y_test_data), sort_ind, axis=0)
        if x_data.shape[0] < 2:
            continue
        logscale = (x_data.max() / x_data.min()) > 100
        ax.plot(x_data, 1-y_test_data, label=algorithm, linewidth=1.25)
        sc = ax.scatter(selected_models['params_dict'][param], 1-selected_models['mean_results_dict']['test_metric'], marker='x')
        ax.scatter(x_data[np.argmin(1-y_test_data)], (1-y_test_data).min(), marker='o', c=sc.get_facecolor())

    plt.title(f'Error Rate vs {make_label(param)}')

    ax.set_axisbelow(True)
    ax.grid(visible=True, which='major', linewidth=0.5, linestyle='--')

    if logscale:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel('Error Rate')
    leg = ax.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f'all_algorithms-error_rate-vs-{make_label(param)}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

def save_selected_models_params(selected_models, plot_subdir):
    '''Save selected models' parameter sets to a CSV file.

    Parameters:
        selected_models (pandas.DataFrame): A pandas.DataFrame containing results for the selected models.
        plot_subdir (str): Directory path where the output CSV file will be saved.

    Returns:
        None: The function writes to a CSV file and does not return a value.
    '''

    test_params_str = [f'{algorithm}, {make_param_str(data['params_dict'])}' for algorithm, data in selected_models[['params_dict']].iterrows()]

    with open(os.path.join(plot_subdir, f'best_params_best_worst_case_metric_params.csv'), 'w') as f:
        f.write('\nTop Test Models:\n')
        for st in test_params_str:
            f.write(f'{st}\n')

def plot_train_test_metric_vs_param(algorithm_results_df, selected_model, param, plot_subdir):
    '''Plot and save a line plot of train and test accuracy against a single hyperparameter (with other hyperparameters fixed).

    Parameters:
        algorithm_results_df (pandas.DataFrame): Results for all models for a specific algorithm.
        selected_model (pandas.Series): A pandas.Series containing the results for selected model (to fix other hyperparameters).
        param (str): Name of the hyperparameter to plot.
        plot_subdir (str): Directory path where the output plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    x_data = []
    y_test = []
    y_train = []
    y_major_test = []
    y_minor_test = []
    for _, algorithm_data in algorithm_results_df.iterrows():
        if compare_dicts(algorithm_data['params_dict'], selected_model['params_dict'], [param]):
            x_data.append(algorithm_data['params_dict'][param])
            y_train.append(algorithm_data['mean_results_dict']['train_metric'])
            y_test.append(algorithm_data['mean_results_dict']['test_metric'])
            y_major_test.append(algorithm_data['mean_results_dict']['major_class_test_metric'])
            y_minor_test.append(algorithm_data['mean_results_dict']['minor_class_test_metric'])
    sort_ind = np.argsort(x_data, axis=0)
    x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
    y_train_data = np.take_along_axis(np.array(y_train), sort_ind, axis=0)
    y_test_data = np.take_along_axis(np.array(y_test), sort_ind, axis=0)
    if x_data.shape[0] < 2:
        return
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.title(f'Train and test accuracy vs {make_label(param)}')

    ax.set_axisbelow(True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
    ax.plot(x_data, y_train_data, 'r', marker='o', label='Train')
    ax.plot(x_data, y_test_data, 'b', marker='o', label='Test')

    if x_data.max() / x_data.min() > 100:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel('Accuracy')

    y_min = np.min((y_test_data, y_train_data))
    y_max = np.max((y_test_data, y_train_data))
    y_range = y_max - y_min

    ax.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f'{selected_model["algorithm"]}-line_metric-vs-{param}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_major_minor_metric_vs_param(algorithm_results_df, selected_model, param, plot_subdir):
    '''Plot and save a line plot of test accuracy on major and minor classes against a specific hyperparameter (with other hyperparameters fixed).

    Parameters:
        algorithm_results_df (pandas.DataFrame): Results for all models for a specific algorithm.
        selected_model (pandas.Series): A pandas.Series containing the results for selected model (to fix other hyperparameters).
        param (str): Name of the hyperparameter to plot.
        plot_subdir (str): Directory path where the output plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    x_data = []
    y_major_test = []
    y_minor_test = []
    for _, algorithm_data in algorithm_results_df.iterrows():
        if compare_dicts(algorithm_data['params_dict'], selected_model['params_dict'], [param]):
            x_data.append(algorithm_data['params_dict'][param])
            y_major_test.append(algorithm_data['mean_results_dict']['major_class_test_metric'])
            y_minor_test.append(algorithm_data['mean_results_dict']['minor_class_test_metric'])
    sort_ind = np.argsort(x_data, axis=0)
    x_data = np.take_along_axis(np.array(x_data), sort_ind, axis=0)
    y_major_test_data = np.take_along_axis(np.array(y_major_test), sort_ind, axis=0)
    y_minor_test_data = np.take_along_axis(np.array(y_minor_test), sort_ind, axis=0)
    if x_data.shape[0] < 2:
        return
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.title(f'Test accuracy on major and minor class vs {make_label(param)}')

    ax.set_axisbelow(True)
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, zorder=0)
    ax.plot(x_data, y_minor_test_data, 'r', marker='o', label='Accuracy on minor class')
    ax.plot(x_data, y_major_test_data, 'b', marker='o', label='Accuracy on major class')

    if x_data.max() / x_data.min() > 100:
        ax.set_xscale('log')
    ax.set_xlabel(make_label(param))
    ax.set_ylabel('Accuracy')

    y_min = np.min((y_minor_test_data, y_major_test_data))
    y_max = np.max((y_minor_test_data, y_major_test_data))
    y_range = y_max - y_min

    ax.set_ylim(bottom=y_min - 0.1*y_range, top=y_max + 0.1*y_range)
    ax.legend()

    plt.tight_layout()
    out_png = os.path.join(plot_subdir, f'{selected_model['algorithm']}-line_class_metric-vs-{param}.png')
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_2d_dataset(datasets_dir_path, dataset_name, plot_subdir):
    '''Plot and save a 2D scatter plot of the dataset (if 2D features), colored by class label.

    Parameters:
        datasets_dir_path (str): Directory where datasets are stored.
        dataset_name (str): Filename of the dataset (CSV format).
        plot_subdir (str): Directory path where the plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    dataset_file = os.path.join(datasets_dir_path, dataset_name)
    if not (os.path.exists(dataset_file)):
        return
    dataset = np.genfromtxt(dataset_file, delimiter=',')
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
    plt.savefig(os.path.join(plot_subdir,f'dataset.png'),dpi=150)
    plt.close()

def plot_model_preds_scatter(selected_models, test_dir_path, datasets_dir_path, dataset_name, plot_subdir):
    '''Plot and save 2D scatter plots of model predictions for each selected model, highlighting correct and incorrect classifications.

    Parameters:
        selected_models (pandas.DataFrame): A pandas.DataFrame containing results for the selected models.
        test_dir_path (str): Path to directory containing prediction result files.
        datasets_dir_path (str): Path to directory where the datasets are stored.
        dataset_name (str): Name of the dataset.
        plot_subdir (str): Directory path where the output plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and does not return a value.
    '''

    pred_dir=os.path.join(test_dir_path,'pred')
    
    if os.path.isdir(pred_dir):
        dataset_file = os.path.join(datasets_dir_path, dataset_name)
        if not (os.path.exists(dataset_file)):
            return
        dataset = np.genfromtxt(dataset_file, delimiter=',')
        X = dataset[:,:-1]
        y = dataset[:,-1]
        if X.shape[1] != 2:
            return

        for algo_,row_ in selected_models[['file_postfix','params','params_dict']].iterrows():
            postfix=row_['file_postfix']

            pred_file=os.path.join(pred_dir,f'{postfix}_pred.csv')
            if not(os.path.exists(pred_file)):
                continue

            pred=np.genfromtxt(pred_file,delimiter=',')

            fig = plt.figure(figsize=(10,5))
            axis = plt.gca()
            #fig.suptitle(f'{algo_} Predictions')
            
            true_preds = np.where(y==pred)
            false_preds =  np.where(y!=pred)

            axis.scatter(x=X[true_preds, 0], y=X[true_preds, 1], c=np.where(pred[true_preds], 'r', 'b'), marker='+', s=12)
            axis.scatter(x=X[false_preds, 0], y=X[false_preds, 1], c=np.where(pred[false_preds], 'darkred',  'darkblue'), marker='x')
            axis.grid(False)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f'{algo_} Predictions')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_subdir,f'{algo_}_preds.png'),dpi=150)
            plt.close()

def plot_mode(only_dirs=None, multiprocessing=True):
    '''Launches plot mode for each experiment in multiprocessing mode or a single process mode.

    Parameters:
        only_dirs (list[str]): If not None plot results for selected directories.
        multiprocessing (bool): If True, uses number of processes equal to physical cpu count. If int, uses that number of processes. If False, does not use multiprocessing.

    Returns:
        None
    '''
    
    results_root = 'results'
    datasets_root = 'datasets'
    datasets_dir_path = os.path.join(datasets_root)

    if not os.path.exists(results_root):
        print('No results folder found.')
        return
    if only_dirs is not None and len(only_dirs) > 0:
        top_level_dirs = [d for d in only_dirs if os.path.isdir(os.path.join(results_root, d))]
    else:
        top_level_dirs = [d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))]

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
                    
                    csv_path = os.path.join(test_dir_path, 'results.csv')
                    if not os.path.exists(csv_path):
                        continue
                    pool.apply_async(plot_results, args=[csv_path, test_dir_path], kwds={'datasets_dir_path' : datasets_dir_path, })
                pool.close()
                pool.join()
                print('=== Finished plots ===')
        else: # NOT TESTED
            for test_name in os.listdir(time_dir_path):
                test_dir_path = os.path.join(time_dir_path, test_name)
                if not os.path.isdir(test_dir_path):
                    continue
                
                csv_path = os.path.join(test_dir_path, 'results.csv')
                if not os.path.exists(csv_path):
                    continue
                plot_results(csv_path, test_dir_path, datasets_dir_path=datasets_dir_path)
                print(f'== Finished plots for {test_name} ==')

            print('== Finished plots ==')

def plot_results(csv_path,test_dir_path, datasets_dir_path=None):
    '''Generates performance visualizations for ML experiments.

    Loads results from a CSV file, identifies top-performing models per algorithm based on validation metrics,
    and creates and saves plots comparing parameter effects, algorithm performance, and prediction outputs. 

    Parameters:
        csv_path (str): Path to CSV file containing experiment results.
        test_dir_path (str): Directory where plots and results will be saved.
        datasets_dir_path (str, optional): Path to datasets directory for generating 2D dataset visualizations.

    Returns:
        None
    '''

    sns.set_theme(style='whitegrid', palette='deep')
    plt.rc('axes', titlesize=16, labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    dataset_name = os.path.basename(test_dir_path)

    all_results_df = pd.read_csv(csv_path, sep=',')
    if all_results_df.empty or 'algorithm' not in all_results_df.columns:
        return

    # transform dictionaries stored in strings to normal dictionaries
    if 'params' in all_results_df.columns:
        all_results_df['params_dict'] = all_results_df['params'].apply(parse_params_dict)
    else:
        all_results_df['params_dict'] = [{}]*len(all_results_df)
    
    if 'mean_results' in all_results_df.columns:
        all_results_df['mean_results_dict'] = all_results_df['mean_results'].apply(parse_params_dict)
    else:
        all_results_df['mean_results_dict'] = [{}]*len(all_results_df)

    if 'best_test_metric_results' in all_results_df.columns:
        all_results_df['best_test_metric_results_dict'] = all_results_df['best_test_metric_results'].apply(parse_params_dict)
    else:
        all_results_df['best_test_metric_results_dict'] = [{}]*len(all_results_df)

    if 'worst_test_metric_results' in all_results_df.columns:
        all_results_df['worst_test_metric_results_dict'] = all_results_df['worst_test_metric_results'].apply(parse_params_dict)
    else:
        all_results_df['worst_test_metric_results_dict'] = [{}]*len(all_results_df)    
    
    if 'best_validation_metric_results' in all_results_df.columns:
        all_results_df['best_validation_metric_results_dict'] = all_results_df['best_validation_metric_results'].apply(parse_params_dict)
    else:
        all_results_df['best_validation_metric_results_dict'] = [{}]*len(all_results_df)

    plot_subdir = os.path.join(test_dir_path, 'plots')
    os.makedirs(plot_subdir, exist_ok=True)
    other_plots_subdir = os.path.join(plot_subdir, 'other')
    os.makedirs(other_plots_subdir, exist_ok=True)

    algorithms = all_results_df['algorithm'].unique()

    
    # PARAMETERS
    best_validation_metric_models = all_results_df.loc[[ ind_max(all_results_df[all_results_df['algorithm'] == algorithm]['best_validation_metric_results_dict'], 'validation_metric') for algorithm in algorithms]]
    params_to_compare = ['n_estimators', 'learning_rate', 'c', 'delta', 'epsilon', 'tau','iterations' ]

    for algorithm in algorithms:
        best_validation_metric_model = best_validation_metric_models.loc[best_validation_metric_models['algorithm'] == algorithm]
        algorithm_results_df = all_results_df[all_results_df['algorithm'] == algorithm]
        
        ### line plots for the best test accuracy models
        for param in params_to_compare:
            if param not in best_validation_metric_model['params_dict'].keys():
                continue

            # Train and Test Accuracy vs param
            plot_train_test_metric_vs_param(algorithm_results_df, best_validation_metric_model, algorithm, param, plot_subdir)

            # Major and Minor Accuracy vs param
            plot_major_minor_metric_vs_param(algorithm_results_df, best_validation_metric_model, algorithm, param, plot_subdir)

        # 3d plot test_metric vs (learning_rate x n_estimators) for best validation model
        for i in range(len(params_to_compare)-1):
            for j in range(i+1, len(params_to_compare)):
                param_1 = params_to_compare[i]
                param_2 = params_to_compare[j]
                if param_1 not in best_validation_metric_model['params_dict'].keys() or param_2 not in best_validation_metric_model['params_dict'].keys():
                    continue
                plot_3d_2params(algorithm_results_df, best_validation_metric_model, param_1, param_2, plot_subdir)


    ### compare top test accuracy models from each class
    plot_best_worst_and_cases_for_selected_models_in_class(best_validation_metric_models, plot_subdir)
    plot_best_worst_and_cases_for_selected_models_in_class(best_validation_metric_models, plot_subdir, readable=False)

    #  All algos Test Accuracy vs param
    plot_all_algorithms_lines_test_metric_param(all_results_df, algorithms, best_validation_metric_models, plot_subdir, param='n_estimators')

    ### Save best validation models params
    save_selected_models_params(best_validation_metric_models, plot_subdir)
        
    ### Scatter plot for 2d datasets
    plot_2d_dataset(datasets_dir_path, dataset_name, plot_subdir)

    ### Scatter plots for best validation accuracy model from each class
    plot_model_preds_scatter(best_validation_metric_models, test_dir_path, datasets_dir_path, dataset_name, plot_subdir)

if __name__ == '__main__':
    plot_mode()