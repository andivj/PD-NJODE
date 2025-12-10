import os
import pandas as pd
import matplotlib.pyplot as plt


def _find_best_epoch(saved_models_path, model_id, metric_overview_col):
    """
    Read metric_id-<id>.csv and return (best_epoch, best_metric_value)
    using the closest matching metric column.
    """
    metric_file = os.path.join(
        saved_models_path, f"id-{int(model_id)}", f"metric_id-{int(model_id)}.csv"
    )
    if not os.path.exists(metric_file):
        return None, None

    dfm = pd.read_csv(metric_file)
    # derive a non-min metric name from the overview column
    base_from_overview = metric_overview_col
    if base_from_overview.endswith("_min"):
        base_from_overview = base_from_overview[: -len("_min")]

    metric_candidates = [
        base_from_overview,
        "evaluation_mean_diff",
        "evaluation_mse",
        "eval_metric",
        "val_loss",
        "test_loss",
    ]
    metric_col = None
    for c in metric_candidates:
        if c in dfm.columns and dfm[c].notna().any():
            metric_col = c
            dfm = dfm[dfm[c].notna()]
            break
    if metric_col is None:
        return None, None

    idx = dfm[metric_col].idxmin()
    row = dfm.loc[idx]
    epoch = row["epoch"] if "epoch" in row else None
    return epoch, row[metric_col]


def print_best_models(saved_models_path, n_models=3, plot=False):
    """
    Prints the best models based on evaluation_mean_diff_min from the
    training_overview CSV inside the given models directory. Optionally plots
    a bar chart of the top models' evaluation_mean_diff_min.

    Parameters:
        saved_models_path (str): Path to the directory containing model checkpoints.
        n_models (int): Number of top models to print/plot.
        plot (bool): If True, produce a bar plot of the top models.
    """
    if not os.path.isdir(saved_models_path):
        print(f"{saved_models_path} not found, skipping.")
        return

    csv_file = None
    for filename in os.listdir(saved_models_path):
        if filename.startswith("training_overview-ids-") and filename.endswith(".csv"):
            csv_file = os.path.join(saved_models_path, filename)
            break

    if csv_file is None:
        print(f"No training overview CSV found in {saved_models_path}, skipping.")
        return

    df = pd.read_csv(csv_file)

    # choose the best available metric column with non-NaN values
    metric_candidates = [
        "evaluation_mean_diff_min",
        "evaluation_mse_min",
        "eval_metric_min",
        "val_loss_min",
        "val_loss",
    ]
    metric_col = None
    for c in metric_candidates:
        if c in df.columns:
            # drop NaNs for this metric
            if df[c].notna().any():
                metric_col = c
                df = df[df[c].notna()]
                break
    if metric_col is None:
        print(f"No usable metric column in {csv_file}; columns={list(df.columns)}")
        return

    df_sorted = df.sort_values(metric_col, ascending=True)
    top = df_sorted[["id", metric_col]].head(n_models).copy()

    # attach best epoch per model id
    best_epochs = []
    for mid in top["id"]:
        be, _ = _find_best_epoch(saved_models_path, mid, metric_col)
        best_epochs.append(be)
    top["best_epoch"] = best_epochs

    print(f"\nBest {n_models} models in: {saved_models_path}")
    print(top[["id", "best_epoch", metric_col]])

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(top["id"].astype(str), top[metric_col])
        plt.xlabel("Model ID")
        plt.ylabel(metric_col)
        plt.title(f"Top {n_models} models in {os.path.basename(saved_models_path.rstrip('/'))}")
        plt.tight_layout()
        plt.show()

def run_examples(plot=False):
    examples = {
        "DepObs": "../data/saved_models_DepObservations/",
        "BM_NoisyObs": "../data/saved_models_BMNoisyObs/",
        "BS_dep_obs_skew_noise": "../data/saved_models_BS_dep_obs_skew_noise/",
        "BS_dep_obs_noisy": "../data/saved_models_BS_dep_obs_noisy/",
        "BS_dep_obs_heavy_noise": "../data/saved_models_BS_dep_obs_heavy_noise/",
        "BS_dep_obs_heavy_noise_robust": "../data/saved_models_BS_dep_obs_heavy_noise_robust/",
        "BS_dep_obs_heavy_noise_df2": "../data/saved_models_BS_dep_obs_heavy_noise_df2/",
        "BS_dep_obs_heavy_noise_df3": "../data/saved_models_BS_dep_obs_heavy_noise_df3/",
        "BS_dep_obs_heavy_noise_df5": "../data/saved_models_BS_dep_obs_heavy_noise_df5/",
        "BS_dep_obs_hetero_state": "../data/saved_models_BS_dep_obs_hetero_state/",
        "BS_dep_obs_hetero_state_weighted": "../data/saved_models_BS_dep_obs_hetero_state_weighted/",
        "BS_dep_obs_hetero_time": "../data/saved_models_BS_dep_obs_hetero_time/",
        "BS_dep_obs_hetero_time_weighted": "../data/saved_models_BS_dep_obs_hetero_time_weighted/",
    }
    for name, path in examples.items():
        print_best_models(path, n_models=3, plot=plot)


if __name__ == "__main__":
    # Run predefined examples without plotting
    run_examples(plot=False)
    # Or call print_best_models directly, e.g.:
    # print_best_models("../data/saved_models_BS_dep_obs_hetero_time_weighted/", n_models=5, plot=False)
