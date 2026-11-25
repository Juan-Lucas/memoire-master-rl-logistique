"""Diagnostics des politiques (T2, T3, T9, T17 du rapport du 17/11/2026)."""

import pandas as pd
import pytest

from memoire_master_rl_logistique.analysis import policy_diagnostics as pdg
from memoire_master_rl_logistique.simulation.entities import CycleRecord


def _record(truck, start, kind, wait=0.0):
    return CycleRecord(kind=kind, truck_id=truck, start_min=start, end_min=start + 1.0,
                       voluntary_wait_min=wait)


def test_suites_d_attentes():
    """Plus longue suite d'ATTENDRE d'un même camion (interrompue par un cycle) et part des
    attentes de moins de 0,5 min ; les autres camions ne coupent pas la suite."""
    log = [_record("T1", 0, "wait", 0.2), _record("T2", 0.5, "cycle"), _record("T1", 1, "wait", 0.3),
           _record("T1", 2, "wait", 2.0), _record("T1", 3, "cycle"), _record("T1", 4, "wait", 0.4)]
    res = pdg.wait_runs(log)
    assert res["waits"] == 4
    assert res["longest_wait_run"] == 3
    assert res["short_wait_share"] == pytest.approx(3 / 4)
    assert pd.isna(pdg.wait_runs([_record("T1", 0, "cycle")])["short_wait_share"])


def test_resume_accord_et_saturation():
    """Accord sur la route et sur la pelle, pelle plus proche quand elles diffèrent, part
    d'ATTENDRE ; saturation par bloc."""
    rows = [
        {"regime": "nominal", "action": 0, "mtwt_action": 0, "wait": False, "same_route": True,
         "same_shovel": True, "ppo_closer": False},
        {"regime": "nominal", "action": 2, "mtwt_action": 4, "wait": False, "same_route": False,
         "same_shovel": False, "ppo_closer": True},
        {"regime": "nominal", "action": 6, "mtwt_action": 0, "wait": True},
    ]
    for r in rows:
        for b in pdg.BLOCKS:
            r[f"sat_{b}"] = b == "route_wait" and r["action"] == 2
    episodes = pd.DataFrame([{"regime": "nominal", "wait_share_pct": 33.3, "longest_wait_run": 1,
                              "short_wait_share": 1.0}])
    out = pdg.summarise(pd.DataFrame(rows), episodes)
    a = out["agreement"].set_index("regime").loc["nominal"]
    assert a["same_route"] == 0.5 and a["same_shovel"] == 0.5
    assert a["closer_when_different"] == 1.0
    assert a["wait_share"] == pytest.approx(1 / 3)
    sat = out["saturation"].set_index("regime").loc["nominal"]
    assert sat["sat_route_wait"] == pytest.approx(1 / 3) and sat["sat_fuel"] == 0.0


def test_machine():
    info = pdg.machine()
    assert int(info["logical_cores"]) >= int(info["physical_cores"]) >= 1
