"""Chronologie d'un cycle dans le processus de décision (figure ``fig:cycle`` de l'article).

Deux camions de la flotte en fin de poste, et les instants de décision de l'agent :
trajets, files, chargement, déversement, action ATTENDRE, versement de la récompense et
fin du poste. Les durées sont illustratives, mais suivent les règles du simulateur
(``env/mine_env.py``) :

- un camion décide quand il est libéré à une verse (après déversement, et réparation
  éventuelle) ; ATTENDRE dure jusqu'au prochain événement, entre 0,2 et 5 min ;
- la récompense d'un cycle est calculée à l'arrivée du camion à la verse, où la file et le
  déversement sont connus ; elle entre dans la récompense ``R`` du pas pendant lequel cette
  arrivée a lieu, ``(tau_t, tau_t+1]`` (à égalité, les arrivées passent avant les décisions) ;
- le tonnage ne compte que si le déversement se termine avant 480 min ; à la fin du poste,
  l'épisode est tronqué et les cycles en cours sont menés à terme.

Figure en niveaux de gris avec hachures, lisible en noir et blanc. Le dossier de sortie est
obligatoire ; ``reports/figures`` (v1) n'est jamais réécrit.

Lancement :
    python -m memoire_master_rl_logistique.plots.cycle_timeline DOSSIER [--overwrite] [--lang fr|en]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from memoire_master_rl_logistique.plots.common import (
    LANGUAGES, prepare_outputs, save_raster_and_vector, set_language, tr,
)

OUTPUT_FILE = "cycle_timeline.png"
SHIFT_END_MIN = 480.0
X_RANGE = (457.3, 484.3)
WAIT_BOUNDS_MIN = (0.2, 5.0)

# Phases d'un camion : (début, fin, nature). Les durées sont illustratives.
TRUCKS = {
    1: [
        (458.0, 463.5, "travel_empty"), (463.5, 464.5, "queue"), (464.5, 466.5, "load"),
        (466.5, 474.5, "travel_loaded"), (474.5, 475.0, "queue"), (475.0, 476.0, "dump"),
        (476.0, 481.5, "travel_empty"),
    ],
    2: [
        (461.0, 463.5, "wait"), (463.5, 468.0, "travel_empty"), (468.0, 470.0, "load"),
        (470.0, 478.8, "travel_loaded"), (478.8, 480.3, "dump"),
    ],
}
# Décisions de l'agent : (instant, camion qui décide).
DECISIONS = [(458.0, 1), (461.0, 2), (463.5, 2), (476.0, 1)]
# Arrivées à la verse, où la récompense du cycle est calculée : (instant, camion).
DUMP_ARRIVALS = [(474.5, 1), (478.8, 2)]

STYLE = {  # nature -> (couleur de fond, hachures)
    "travel_empty": ("#E0E0E0", ""),
    "travel_loaded": ("#9E9E9E", ""),
    "queue": ("white", "xxx"),
    "load": ("#424242", ""),
    "dump": ("#000000", ""),
    "wait": ("white", "///"),
}


def phase_labels() -> dict[str, str]:
    return {
        "travel_empty": tr("Trajet à vide", "Empty travel"),
        "travel_loaded": tr("Trajet chargé", "Loaded travel"),
        "queue": tr("File", "Queue"),
        "load": tr("Chargement", "Loading"),
        "dump": tr("Déversement", "Dumping"),
        "wait": tr("ATTENDRE", "WAIT"),
    }


def step_of(time_min: float) -> int:
    """Indice du pas (0 pour tau_t) dont la récompense reçoit un événement à ``time_min`` :
    le pas ``l`` couvre ``(tau_l, tau_l+1]``, le dernier va jusqu'à la fin du poste."""
    times = [t for t, _ in DECISIONS]
    for l in range(len(times) - 1):
        if times[l] < time_min <= times[l + 1]:
            return l
    if times[-1] < time_min:
        return len(times) - 1
    raise ValueError(f"Événement avant la première décision : {time_min}")


def step_rewards() -> list[str]:
    """Contenu de la récompense de chaque pas, dans la langue des figures."""
    content: list[list[str]] = [[] for _ in DECISIONS]
    for l, (time_min, truck) in enumerate(DECISIONS):
        first = next(p for p in TRUCKS[truck] if p[0] == time_min)
        if first[2] == "wait":
            content[l].append(tr("coût d'ATTENDRE", "WAIT cost"))
    for time_min, truck in DUMP_ARRIVALS:
        dump_end = next(e for s, e, k in TRUCKS[truck] if k == "dump" and s >= time_min)
        label = f"$r_c$ ({tr('camion', 'truck')} {truck})"
        if dump_end >= SHIFT_END_MIN:
            label += tr(", production 0", ", production 0")
        content[step_of(time_min)].append(label)
    content[-1].append(tr("cycles finis après 480", "cycles finished after 480"))
    return ["\n".join(c) if c else "0" for c in content]


def draw(ax: plt.Axes) -> None:
    labels = phase_labels()
    rows = {1: 2.0, 2: 1.0}
    height = 0.5
    for truck, phases in TRUCKS.items():
        y = rows[truck]
        for start, end, kind in phases:
            face, hatch = STYLE[kind]
            alpha = 0.35 if start >= 476.0 and truck == 1 else 1.0
            ax.add_patch(mpatches.Rectangle((start, y - height / 2), end - start, height,
                                            facecolor=face, hatch=hatch, edgecolor="black",
                                            linewidth=0.6, alpha=alpha))
    for time_min, truck in DECISIONS:
        ax.plot(time_min, rows[truck] + height / 2 + 0.12, marker="v", color="black",
                markersize=7)
    for time_min, truck in DUMP_ARRIVALS:
        ax.plot(time_min, rows[truck] - height / 2 - 0.12, marker="*", color="black",
                markersize=10)

    # Instants de décision, durées des pas et récompenses
    y0 = 0.0
    times = [t for t, _ in DECISIONS] + [SHIFT_END_MIN]
    ax.plot([times[0], SHIFT_END_MIN], [y0, y0], color="black", linewidth=0.8)
    names = [r"$\tau_t$", r"$\tau_{t+1}$", r"$\tau_{t+2}$", r"$\tau_{t+3}$"]
    for time_min, name in zip(times, names):
        ax.plot([time_min, time_min], [y0 - 0.08, y0 + 0.08], color="black", linewidth=1)
        ax.text(time_min, y0 + 0.14, name, ha="center", va="bottom", fontsize=11)
    rewards = step_rewards()
    for l in range(len(DECISIONS)):
        a, b = times[l], times[l + 1]
        ax.annotate("", xy=(a, y0 - 0.25), xytext=(b, y0 - 0.25),
                    arrowprops=dict(arrowstyle="<->", linewidth=0.7))
        dt = r"$\Delta t_{t}$" if l == 0 else rf"$\Delta t_{{t+{l}}}$"
        ax.text((a + b) / 2, y0 - 0.33, dt, ha="center", va="top", fontsize=10)
        rname = r"$R_t$" if l == 0 else rf"$R_{{t+{l}}}$"
        ax.text((a + b) / 2, y0 - 0.62, f"{rname}{tr(' :', ':')} {rewards[l]}", ha="center", va="top",
                fontsize=9)

    ax.axvline(SHIFT_END_MIN, color="black", linestyle="--", linewidth=1)
    ax.text(SHIFT_END_MIN + 0.2, 2.62, tr("Fin du poste\n(480 min) :\ntroncature",
                                         "End of shift\n(480 min):\ntruncation"),
            fontsize=9.5, va="top", ha="left")

    handles = [mpatches.Patch(facecolor=STYLE[k][0], hatch=STYLE[k][1], edgecolor="black",
                              label=labels[k]) for k in STYLE]
    handles += [
        plt.Line2D([], [], marker="v", color="black", linestyle="", markersize=7,
                   label=tr("Décision (camion libre à la verse)",
                            "Decision (truck free at a dump)")),
        plt.Line2D([], [], marker="*", color="black", linestyle="", markersize=10,
                   label=tr("Arrivée à la verse : récompense $r_c$ calculée",
                            "Arrival at the dump: reward $r_c$ computed")),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=9.5, frameon=False)
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(-1.35, 2.75)
    ax.set_yticks([y0, rows[2], rows[1]])
    ax.set_yticklabels([tr("Décisions", "Decisions"), f"{tr('Camion', 'Truck')} 2",
                        f"{tr('Camion', 'Truck')} 1"], fontsize=11, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(tr("Temps du poste (min)", "Shift time (min)"))
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)


def main(out_dir: str | Path, overwrite: bool = False, lang: str = "fr") -> Path:
    """Écrit ``cycle_timeline.png`` (et ``.pdf``) dans ``out_dir`` ; refuse d'écraser sans
    ``overwrite``."""
    set_language(lang)
    (out_path,) = prepare_outputs(out_dir, [OUTPUT_FILE], overwrite)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    draw(ax)
    fig.tight_layout()
    save_raster_and_vector(fig, out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Écrit : {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", type=Path, help="dossier de sortie (créé si besoin)")
    parser.add_argument("--overwrite", action="store_true", help="autoriser l'écrasement")
    parser.add_argument("--lang", choices=LANGUAGES, default="fr", help="langue des libellés")
    args = parser.parse_args()
    main(args.out_dir, args.overwrite, args.lang)
