"""Script pour lancer le benchmark uniquement pour le scénario nominal_asymétrique."""

from memoire_master_rl_logistique.experiments.run_benchmark import run_all_benchmarks

if __name__ == "__main__":
    print("Lancement du benchmark pour le scénario nominal_asymétrique uniquement...")
    run_all_benchmarks(
        output_dir="data/results",
        train_ppo_flag=True,
        scenario_names=["nominal_asymétrique"],
    )
    print("Benchmark terminé !")
