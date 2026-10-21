"""Post-evaluation model explanations; outputs remain local by default."""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud.features import build_features
from fraud.io import dump_json, read_json
from fraud.models import canonical_frame


def explain(config):
    from fraud.pipeline import verify_frozen
    verify_frozen(config)
    run = Path(config["run_dir"])
    evaluation = read_json(run / "evaluation.json")
    from fraud.io import digest
    for file, expected_hash in evaluation["source_outputs"].items():
        if digest(run / file) != expected_hash:
            raise ValueError(f"Evaluation output changed: {file}")
    name = evaluation["chosen_model"]
    bundle = joblib.load(run / "models" / f"{name}.joblib")
    output = run / "explanations"
    output.mkdir(exist_ok=True)
    if name == "logistic":
        names = bundle.pipeline.named_steps["transform"].get_feature_names_out()
        values = bundle.pipeline.named_steps["model"].coef_[0]
        table = pd.DataFrame({"feature": names, "coefficient_standardized_or_encoded": values})
        table["absolute_value"] = table.iloc[:, 1].abs()
        table.sort_values("absolute_value", ascending=False).to_csv(output / "linear_coefficients.csv", index=False)
        status = {"method": "linear_coefficients", "model": name,
                  "interpretation": "Conditional log-odds associations for scaled/encoded inputs, not causal effects"}
    else:
        import shap
        frame = pd.read_parquet(Path(config["processed_dir"]) / "transactions.parquet")
        index = pd.read_parquet(run / "split_index.parquet")
        ids = index.loc[index.split == "test", "TransactionID"]
        positions = frame.index[frame.TransactionID.isin(ids)].to_numpy()
        rng = np.random.default_rng(config["seed"])
        positions = np.sort(rng.choice(positions, min(config["shap_sample"], len(positions)), replace=False))
        features = build_features(frame.drop(columns=["isFraud"]), bundle.include_anonymous, bundle.include_relations)
        x = canonical_frame(features.loc[positions, bundle.columns])
        transformed = bundle.pipeline.named_steps["transform"].transform(x)
        model = bundle.pipeline.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        explanation = explainer(transformed, check_additivity=True)
        names = bundle.pipeline.named_steps["transform"].get_feature_names_out()
        table = pd.DataFrame({"feature": names, "mean_absolute_shap_log_odds": np.abs(explanation.values).mean(axis=0)})
        table = table.sort_values("mean_absolute_shap_log_odds", ascending=False)
        table.to_csv(output / "shap_importance.csv", index=False)
        # Preserve local contributions plus base value for an auditable log-odds sum.
        local = pd.DataFrame(explanation.values, columns=names)
        local.insert(0, "TransactionID", frame.loc[positions, "TransactionID"].to_numpy())
        local.insert(1, "base_log_odds", np.broadcast_to(explanation.base_values, len(positions)))
        local.to_parquet(output / "local_shap.parquet", index=False)
        top = table.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top.feature.str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False), top.mean_absolute_shap_log_odds)
        ax.set_xlabel("Mean absolute SHAP contribution to raw model log-odds")
        fig.tight_layout()
        fig.savefig(output / "shap_importance.png", dpi=150)
        plt.close(fig)
        status = {"method": "TreeSHAP_tree_path_dependent", "model": name, "sample_rows": len(positions),
                  "interpretation": "Raw XGBoost log-odds, before sigmoid calibration; observational associations; anonymous fields do not establish a causal explanation"}
    dump_json(output / "explanation.json", status)
    return status
