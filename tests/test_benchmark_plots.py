from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from memoire_master_rl_logistique.plots.benchmark_comparisons import (
    main,
    pareto_front,
    summarise,
)

REFERENCES_CSV = (
    Path(__file__).parent.parent / "data" / "results_v1_article" / "benchmark_results.csv"
)


def test_pareto_front_points_domines():
    """On minimise le coût et on maximise le gain : seuls les compromis non battus restent."""
    cost = np.array([10, 20, 30, 25])
    gain = np.array([100, 200, 300, 150])  # le 4e point est battu par le 2e (moins cher, plus de gain)

    assert pareto_front(cost, gain).tolist() == [True, True, True, False]


def test_pareto_front_egalite_non_dominee():
    """Deux points identiques ne se dominent pas l'un l'autre."""
    assert pareto_front([5, 5], [50, 50]).tolist() == [True, True]


def test_summarise_moyenne_et_ic():
    """Une moyenne et un IC par (politique, régime) ; valeurs constantes → IC de largeur nulle."""
    df = pd.DataFrame(
        {
            "policy": ["A"] * 3 + ["B"] * 3,
            "scenario": ["nominal"] * 6,
            "productivity_tph": [10, 10, 10, 1, 2, 3],
        }
    )
    summary = summarise(df, "productivity_tph").set_index("policy")

    assert summary.loc["A", "mean"] == pytest.approx(10)
    assert (summary.loc["A", "ci_low"], summary.loc["A", "ci_high"]) == (10, 10)
    assert summary.loc["B", "ci_low"] <= 2 <= summary.loc["B", "ci_high"]


@pytest.mark.skipif(not REFERENCES_CSV.exists(), reason="CSV v1 absent")
def test_main_ecrit_les_trois_figures(tmp_path):
    """Sur le CSV v1, les trois figures sont écrites ; un second lancement refuse d'écraser."""
    outputs = main(REFERENCES_CSV, tmp_path)

    assert [p.name for p in outputs] == [
        "barplot_productivite_scenarios.png",
        "wait_time_comparison.png",
        "pareto_productivity_waiting.png",
    ]
    assert all(p.exists() for p in outputs)
    # version vectorielle à côté de chaque PNG (figures de l'article en PDF)
    assert all(p.with_suffix(".pdf").read_bytes().startswith(b"%PDF") for p in outputs)
    with pytest.raises(FileExistsError):
        main(REFERENCES_CSV, tmp_path)


def test_courbes_de_la_campagne(tmp_path):
    """Une figure, trois panneaux (régimes), à partir du CSV des courbes de la campagne ;
    jamais d'écrasement sans --overwrite."""
    from memoire_master_rl_logistique.plots.campaign_curves import main as curves_main

    rows = [dict(x=p, progress_pct=p, n_runs=5, mean=4000 + p, ci_low=3990 + p,
                 ci_high=4010 + p, algorithm=a, scenario=s)
            for a in ("PPO", "DQN", "Q-Learning", "SARSA")
            for s in ("nominal", "high_load", "high_breakdown") for p in (0.5, 50.5, 99.5)]
    csv = tmp_path / "courbes.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = curves_main(csv, tmp_path / "fig")
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        curves_main(csv, tmp_path / "fig")


def test_figures_avec_plusieurs_graines_d_entrainement(tmp_path):
    """Campagne : un agent a 5 modèles. La barre est la moyenne de ses modèles, et l'IC
    porte sur les 30 réplications (moyennées sur les modèles), pas sur 150 épisodes."""
    from memoire_master_rl_logistique.analysis import statistical_tests as st
    from memoire_master_rl_logistique.plots import benchmark_comparisons as bc

    rows = []
    for scenario, trucks in (("nominal", 12), ("high_load", 18), ("high_breakdown", 12)):
        for s in range(42, 72):
            rows.append(dict(scenario=scenario, policy="Fixed", mode=None, train_seed=None,
                             seed=s, truck_count=trucks, productivity_tph=4000.0 + s,
                             queue_wait_min_per_truck=10.0))
            for t in range(1, 6):
                rows.append(dict(scenario=scenario, policy="PPO", mode="spécialisé",
                                 train_seed=t, seed=s, truck_count=trucks,
                                 productivity_tph=4100.0 + s + 10 * t,
                                 queue_wait_min_per_truck=12.0))
    csv = tmp_path / "campagne.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = st.average_over_training_seeds(st.load_results(csv))
    ppo = df[(df["policy"] == "PPO") & (df["scenario"] == "nominal")]
    assert len(ppo) == 30 and ppo["productivity_tph"].mean() == pytest.approx(4100 + 56.5 + 30)
    assert len(bc.main(csv, tmp_path / "fig")) == 3


def test_schema_des_regimes(tmp_path):
    """Les trois régimes avec la probabilité de panne par cycle (plus de camions « en
    panne » dessinés) ; jamais d'écrasement sans --overwrite."""
    from memoire_master_rl_logistique.plots import scenarios_schematic as sc

    assert [s["name"] for s in sc.SCENARIOS] == ["Nominal", "High load", "Frequent breakdowns"]
    assert [s["breakdown_pct"] for s in sc.SCENARIOS] == [2, 2, 10]
    assert all("broken_trucks" not in s for s in sc.SCENARIOS)
    out = sc.main(tmp_path)
    assert out.exists() and out.stat().st_size > 0
    assert out.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    with pytest.raises(FileExistsError):
        sc.main(tmp_path)


def test_figures_sans_titre_dans_l_image(tmp_path, monkeypatch):
    """Convention de l'article : le titre est dans la légende LaTeX, pas dans l'image. Les
    barres de productivité, la répartition par pelle et la heatmap n'ont pas de titre
    général ; la légende des régimes des barres est au-dessus du graphique."""
    import matplotlib.pyplot as plt

    from memoire_master_rl_logistique.plots import behaviour as bh
    from memoire_master_rl_logistique.plots import benchmark_comparisons as bc
    from memoire_master_rl_logistique.plots import robustness as rb

    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)  # garder les figures
    plt.close("all")
    try:
        summary = pd.DataFrame([dict(policy=p, scenario=s, mean=4000.0, ci_low=3990.0,
                                     ci_high=4010.0)
                                for p in ("Fixed", "PPO")
                                for s in ("nominal", "high_load", "high_breakdown")])
        bc.plot_grouped_bars(summary, "t/h", tmp_path / "bars.png")
        bars = plt.gcf()
        assert bars.axes[0].get_title() == ""
        assert bars.axes[0].get_legend().get_bbox_to_anchor().y0 >= bars.axes[0].bbox.y1 - 1

        shares = pd.DataFrame([dict(policy="Fixed", scenario=s, share_S1_pct=34.0,
                                    share_S2_pct=33.0, share_S3_pct=33.0, wait_share_pct=0.0)
                               for s in ("nominal", "high_load", "high_breakdown")])
        bh.plot_behaviour(shares, tmp_path / "behaviour.png")
        assert plt.gcf()._suptitle is None

        ratios = pd.DataFrame({"high_load": [0.9], "high_breakdown": [0.95]}, index=["Fixed"])
        rb.plot_robustness_heatmap(ratios, ratios * 10, tmp_path / "heatmap.png")
        assert plt.gcf()._suptitle is None
    finally:
        monkeypatch.undo()
        plt.close("all")


def test_libelles_des_figures_en_anglais(tmp_path):
    """La version anglaise de l'article a ses propres figures : mêmes données, textes en
    anglais ; revenir au français remet les libellés français."""
    from memoire_master_rl_logistique.plots import common

    try:
        common.set_language("en")
        assert common.policy_label("Fixed") == "Fixed Assignment"
        assert common.policy_label("LP") == "Static LP"
        assert common.SCENARIO_LABELS["high_breakdown"] == "Frequent breakdowns"
        assert common.tr("Politique", "Policy") == "Policy"
        common.set_language("fr")
        assert common.policy_label("Fixed") == "Affectation fixe"
        assert common.policy_label("LP") == "LP statique"
        assert common.tr("Politique", "Policy") == "Politique"
        with pytest.raises(ValueError):
            common.set_language("de")
    finally:
        common.set_language("fr")


def test_chronologie_du_cycle_suit_les_regles_du_simulateur():
    """Schéma du cycle (fig:cycle) : les durées sont illustratives, mais les règles sont
    celles de env/mine_env.py. Phases contiguës, ATTENDRE de 0,2 à 5 min et terminée par un
    événement, décision à la fin du déversement, récompense calculée à l'arrivée à la verse
    et versée au pas (tau_l, tau_l+1], tonnage nul si le déversement finit après 480 min."""
    from memoire_master_rl_logistique.plots import common
    from memoire_master_rl_logistique.plots import cycle_timeline as ct

    for phases in ct.TRUCKS.values():
        assert all(a[1] == b[0] for a, b in zip(phases, phases[1:]))
        assert all(start < end for start, end, _ in phases)
    events = {s for phases in ct.TRUCKS.values() for s, _, k in phases if k in ("queue", "load")}
    for truck, phases in ct.TRUCKS.items():
        for start, end, kind in phases:
            if kind == "wait":
                assert ct.WAIT_BOUNDS_MIN[0] <= end - start <= ct.WAIT_BOUNDS_MIN[1]
                assert end in events  # ATTENDRE s'arrête au prochain événement
    times = [t for t, _ in ct.DECISIONS]
    assert times == sorted(times) and times[-1] < ct.SHIFT_END_MIN
    for time_min, truck in ct.DECISIONS:
        phases = ct.TRUCKS[truck]
        starts_cycle = any(s == time_min for s, _, _ in phases)
        after_dump = time_min == phases[0][0] or any(
            e == time_min and k == "dump" for _, e, k in phases)
        after_wait = any(e == time_min and k == "wait" for _, e, k in phases)
        assert starts_cycle and (after_dump or after_wait)
    for time_min, truck in ct.DUMP_ARRIVALS:
        assert any(e == time_min and k == "travel_loaded" for _, e, k in ct.TRUCKS[truck])

    # à égalité, l'arrivée compte dans le pas qui se termine à cet instant
    assert ct.step_of(times[1]) == 0 and ct.step_of(times[1] + 0.01) == 1
    assert ct.step_of(ct.SHIFT_END_MIN - 0.1) == len(times) - 1
    try:
        common.set_language("en")
        rewards = ct.step_rewards()
    finally:
        common.set_language("fr")
    assert rewards[0] == "0"
    assert "WAIT cost" in rewards[1]
    assert "truck 1" in rewards[2] and "production 0" not in rewards[2]
    assert "truck 2" in rewards[3] and "production 0" in rewards[3]


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_chronologie_du_cycle_png_et_pdf(tmp_path, lang):
    from memoire_master_rl_logistique.plots import common
    from memoire_master_rl_logistique.plots import cycle_timeline as ct

    try:
        out = ct.main(tmp_path, lang=lang)
        assert out.name == "cycle_timeline.png" and out.stat().st_size > 0
        assert out.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
        with pytest.raises(FileExistsError):
            ct.main(tmp_path, lang=lang)
    finally:
        common.set_language("fr")


def test_figures_lisibles_en_noir_et_blanc():
    """En niveaux de gris, couleurs proches : régimes et pelles se distinguent aussi par leurs
    hachures, les agents par leur style de trait."""
    from memoire_master_rl_logistique.plots import behaviour, common

    hatches = [common.SCENARIO_HATCHES[s] for s in common.MAIN_SCENARIOS]
    assert len(set(hatches)) >= 2 and common.SCENARIO_HATCHES["high_breakdown"]
    assert len(set(behaviour.SHOVEL_HATCHES[:3])) == 3
    assert set(common.AGENT_LINESTYLES) == set(common.AGENT_COLORS)
    assert len(set(common.AGENT_LINESTYLES.values())) == len(common.AGENT_LINESTYLES)
