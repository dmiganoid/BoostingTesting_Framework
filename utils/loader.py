from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_metric(metric):
    match metric:
        case "accuracy":
            return accuracy_score
        case "f1":
            return f1_score
        case "precision":
            return precision_score
        case "recall":
            return recall_score
        case _:
            raise ValueError(f"Unknown metric: {metric}")



def load_algorithm(algorithm, algorithm_config, base_estimator_cfg, random_state):
    base_estimators = []
    base_regressor_estimators = []

    match base_estimator_cfg['estimator_type']:
        case "stump":
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.tree import DecisionTreeRegressor
            for params in ParameterGrid(base_estimator_cfg['estimator_params']):
                base_estimators.append(DecisionTreeClassifier(**params))
                base_regressor_estimators.append(DecisionTreeRegressor(**params))
                
        case "neural_network":
            raise NotImplementedError
            from models.neural_classifier import NeuralBinaryClassifier
            for params in ParameterGrid(base_estimator_cfg['estimator_params']):
                base_estimators.append(NeuralBinaryClassifier(**params))

    param_grid = dict()
    algorithm_class = None
    match algorithm:
        case "AdaBoost":
            from models.adaboost import AdaBoostClassifier
            algorithm_class = AdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']

        case "GradientBoost":
            from models.gradientboost import GradientBoostingClassifier
            algorithm_class = GradientBoostingClassifier
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["loss"] = algorithm_config["GradientBoost"]["loss"]

        case "BrownBoost":
            from models.brownboost import BrownBoostClassifier
            algorithm_class = BrownBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["c"] = algorithm_config["BrownBoost"]["c"]
            param_grid["convergence_criterion"] = algorithm_config["BrownBoost"]["convergence_criterion"]
            param_grid["n_estimators"] = [ max(algorithm_config["common"]["n_estimators"])]
        
        case "MadaBoost":
            from models.madaboost import MadaBoostClassifier
            algorithm_class = MadaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "RealAdaBoost":
            from models.realadaboost import RealAdaBoostClassifier
            algorithm_class = RealAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "ModestAdaBoost":
            from models.modestadaboost import ModestAdaBoostClassifier
            algorithm_class = ModestAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "LogitBoost":
            from models.logitboost import LogitBoost
            algorithm_class = LogitBoost
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "SOWAdaBoost":
            from models.sowadaboost import SOWAdaBoostClassifier
            algorithm_class = SOWAdaBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "WaterBoost":
            from models.waterboost import WaterBoostClassifier
            algorithm_class = WaterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "FilterBoost":
            from models.filterboost import FilterBoostClassifier
            algorithm_class = FilterBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            if algorithm_config.get('FilterBoost', False):
                param_grid["epsilon"] = algorithm_config['FilterBoost'].get('epsilon', 0.9)
                param_grid["delta"] = algorithm_config['FilterBoost'].get('delta', 0.9)
                param_grid["tau"] = algorithm_config['FilterBoost'].get('tau', 0.1)

        # No estimator for CatBoost
        case "CatBoost":
            from catboost import CatBoostClassifier
            algorithm_class = CatBoostClassifier
            param_grid['allow_writing_files'] = [False]
            param_grid['silent'] = [True]
            param_grid["iterations"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["depth"] = algorithm_config['CatBoost']['depth']

        # No estimator for XGBoost
        case "XGBoost":
            from xgboost import XGBClassifier
            algorithm_class = XGBClassifier
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["max_depth"] = algorithm_config['XGBoost']['max_depth']
            param_grid["use_label_encoder"] = [False]
            param_grid["eval_metric"] = ["logloss"]
            param_grid['verbosity'] = [0]

        # No estimator for LightGBM
        case "LightGBM":
            from lightgbm import LGBMClassifier
            algorithm_class = LGBMClassifier
            param_grid['verbose'] = [-1]
            param_grid["n_estimators"] = algorithm_config['common']['n_estimators']
            param_grid["learning_rate"] = algorithm_config['common']['learning_rate']
            param_grid["max_depth"] = algorithm_config['LightGBM']['max_depth']

        case "NGBoost":
            from models.ngboost import NGBoostClassifier
            algorithm_class = NGBoostClassifier
            param_grid["estimator"] = base_regressor_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]

        case "SMOTEBoost":
            from models.smoteboost import SMOTEBoostClassifier
            algorithm_class = SMOTEBoostClassifier
            param_grid["estimator"] = base_estimators
            param_grid["n_estimators"] = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"] = algorithm_config["common"]["learning_rate"]
        
        case "RUSBoost":
            from models.rusboost import RUSBoostClassifier
            algorithm_class = RUSBoostClassifier
            param_grid["estimator"]      = base_estimators
            param_grid["n_estimators"]   = algorithm_config["common"]["n_estimators"]
            param_grid["learning_rate"]  = algorithm_config["common"]["learning_rate"]
        
        case _:
            raise ValueError(f"Unknown algorithm: {algorithm}")


    return (algorithm_class, param_grid)
