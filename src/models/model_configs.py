import copy
import psutil
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
try:
    from .custom_boosting import MadaBoostClassifier, BrownBoostClassifier, FilterBoostClassifier
except ImportError:
    MadaBoostClassifier = None
    BrownBoostClassifier = None
    FilterBoostClassifier = None

def build_model_config_map(methods_cfg):
    out_map = {}
    for name, info in methods_cfg.items():
        if not info.get("enabled", False):
            continue
        out_map[name] = (info["model_type"], info["params"])
    return out_map

def make_model_config(model_type, mparams):
    if model_type == "catboost_classifier":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoostClassifier not installed.")
        return {
            "model_class": CatBoostClassifier,
            "params": copy.deepcopy(mparams)
        }
    elif model_type == "xgb_classifier":
        if XGBClassifier is None:
            raise RuntimeError("XGBClassifier not installed.")
        return {
            "model_class": XGBClassifier,
            "params": copy.deepcopy(mparams)
        }
    elif model_type == "adaboost_classifier":
        final_params = copy.deepcopy(mparams)
        if "base_estimator" in final_params:
            base_type = final_params.pop("base_estimator")
            base_params = final_params.pop("base_estimator_params", {})
            if base_type == "decision_tree":
                dt = DecisionTreeClassifier(**base_params)
                final_params["base_estimator"] = dt
        return {
            "model_class": AdaBoostClassifier,
            "params": final_params
        }
    elif model_type == "madaboost":
        final_params = copy.deepcopy(mparams)
        if "base_estimator" in final_params:
            base_type = final_params.pop("base_estimator")
            base_params = final_params.pop("base_estimator_params", {})
            if base_type == "decision_tree":
                dt = DecisionTreeClassifier(**base_params)
                final_params["base_estimator"] = dt
        return {
            "model_class": MadaBoostClassifier,
            "params": final_params
        }
    elif model_type == "brownboost":
        final_params = copy.deepcopy(mparams)
        if "base_estimator" in final_params:
            base_type = final_params.pop("base_estimator")
            base_params = final_params.pop("base_estimator_params", {})
            if base_type == "decision_tree":
                dt = DecisionTreeClassifier(**base_params)
                final_params["base_estimator"] = dt
        return {
            "model_class": BrownBoostClassifier,
            "params": final_params
        }
    elif model_type == "filterboost":
        final_params = copy.deepcopy(mparams)
        if "base_estimator" in final_params:
            base_type = final_params.pop("base_estimator")
            base_params = final_params.pop("base_estimator_params", {})
            if base_type == "decision_tree":
                dt = DecisionTreeClassifier(**base_params)
                final_params["base_estimator"] = dt
        return {
            "model_class": FilterBoostClassifier,
            "params": final_params
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_memory_usage_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024