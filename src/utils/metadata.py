import yaml

def save_experiment_metadata(path, exp_info, results_df):
    """
    <TODO> Add essential data.
    """
    metadata = {
        "experiment_name": exp_info.get("name"),
        "data_info": exp_info.get("data"),
        "methods": exp_info.get("methods"),
        "results_summary": results_df.to_dict(orient="records")
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, sort_keys=False, allow_unicode=True)
