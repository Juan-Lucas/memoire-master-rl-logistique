"""Étude d'ablation sur w4, le poids de la pénalité d'attente.

Pour chaque valeur, le benchmark est relancé et les quatre agents sont
réentraînés une fois, avec la même graine.

Valeurs testées : w4 = 0.0 / 0.1 / 0.3 (valeur retenue) / 0.5
Scénario fixé  : nominal (12 camions, pb=2 %)

Les résultats sont sauvegardés dans data/results/ablation/ablation_w4.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

from memoire_master_rl_logistique.experiments.run_benchmark import run_benchmark
from memoire_master_rl_logistique.experiments.scenarios import SCENARIOS

# w4 = 0.3 correspond au scénario nominal existant ; les autres sont dans SCENARIOS
ABLATION_KEYS = ["ablation_w4_00", "ablation_w4_01", "nominal", "ablation_w4_05"]
W4_LABELS = {
    "ablation_w4_00": 0.0,
    "ablation_w4_01": 0.1,
    "nominal": 0.3,
    "ablation_w4_05": 0.5,
}


def run_ablation(output_dir: str = "data/results/ablation") -> None:
    """Lance le benchmark pour chaque valeur de w4 et sauvegarde les résultats."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for key in ABLATION_KEYS:
        if key not in SCENARIOS:
            print(f"  Scénario '{key}' introuvable — ignoré.")
            continue

        scenario = SCENARIOS[key]
        w4 = W4_LABELS[key]
        print(f"\n{'='*60}")
        print(f"Ablation w4={w4}  (scénario clé : {key})")
        print("=" * 60)

        rows = run_benchmark(scenario, out_path / key, train_ppo_flag=True)
        for row in rows:
            row["w4"] = w4
        all_rows.extend(rows)

    csv_path = out_path / "ablation_w4.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nRésultats ablation sauvegardés : {csv_path}")


def print_ablation_table(csv_path: str | Path = "data/results/ablation/ablation_w4.csv") -> None:
    """Affiche le tableau récapitulatif des performances par w4 et politique."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    summary = (
        df.groupby(["w4", "policy"])["productivity_tph"]
        .agg(["mean", "std"])
        .round(2)
        .rename(columns={"mean": "mean_tph", "std": "std_tph"})
    )
    print("\n=== Tableau d'ablation — productivité par w4 ===")
    print(summary.to_string())


if __name__ == "__main__":
    run_ablation()
    print_ablation_table()
