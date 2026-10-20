"""Callback d'entraînement (PPO et DQN) qui enregistre les KPIs de chaque épisode."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class KPILoggerCallback(BaseCallback):
    """Enregistre tonnage, attente et carburant de chaque épisode terminé.

    Les moyennes cumulées sont envoyées au logger tous les ``log_freq`` pas ;
    le détail par épisode est écrit dans ``log_path`` en fin d'entraînement.

    Depuis T30, chaque épisode garde aussi le nombre de pas d'entraînement écoulés à sa fin
    (``timesteps`` : l'axe commun des courbes de plusieurs graines, car les épisodes n'ont
    pas tous la même longueur) et sa récompense cumulée (``reward``, fournie par ``Monitor``).
    """

    def __init__(
        self,
        log_path: Path,
        log_freq: int = 1000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        self.log_freq = log_freq
        self._episode_rewards: list[float] = []
        self._episode_tonnages: list[float] = []
        self._episode_waits: list[float] = []
        self._episode_fuels: list[float] = []
        self._episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                self._episode_tonnages.append(info.get("total_tonnage_t", 0.0))
                self._episode_waits.append(info.get("total_wait_min", 0.0))
                self._episode_fuels.append(info.get("total_fuel_l", 0.0))
                self._episode_timesteps.append(self.num_timesteps)
                self._episode_rewards.append(info.get("episode", {}).get("r", float("nan")))

        if self.num_timesteps % self.log_freq == 0 and self._episode_tonnages:
            avg_tonnage = sum(self._episode_tonnages) / len(self._episode_tonnages)
            avg_wait = sum(self._episode_waits) / len(self._episode_waits)
            avg_fuel = sum(self._episode_fuels) / len(self._episode_fuels)

            self.logger.record("mine/avg_tonnage_t", avg_tonnage)
            self.logger.record("mine/avg_wait_min", avg_wait)
            self.logger.record("mine/avg_fuel_l", avg_fuel)
            self.logger.record("mine/episodes_logged", len(self._episode_tonnages))

            if self.verbose > 0:
                print(
                    f"[Step {self.num_timesteps}] "
                    f"tonnage={avg_tonnage:.0f}t, "
                    f"wait={avg_wait:.1f}min, "
                    f"fuel={avg_fuel:.1f}L"
                )

        return True

    @property
    def episodes(self) -> int:
        """Nombre d'épisodes terminés enregistrés."""
        return len(self._episode_tonnages)

    def restore(self, path: Path) -> None:
        """Reprend les épisodes d'un journal écrit par ``write`` (reprise d'un entraînement)."""
        with path.open(encoding="utf-8") as f:
            next(f)  # en-tête
            for line in f:
                _, steps, ton, wait, fuel, reward = line.strip().split(",")
                self._episode_timesteps.append(int(steps))
                self._episode_tonnages.append(float(ton))
                self._episode_waits.append(float(wait))
                self._episode_fuels.append(float(fuel))
                self._episode_rewards.append(float(reward))

    def _on_training_end(self) -> None:
        self.write(self.log_path)

    def write(self, path: Path) -> None:
        """Écrit le détail par épisode dans ``path`` (CSV)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("episode,timesteps,tonnage_t,wait_min,fuel_l,reward\n")
            rows = zip(
                self._episode_timesteps,
                self._episode_tonnages,
                self._episode_waits,
                self._episode_fuels,
                self._episode_rewards,
                strict=True,
            )
            for i, (steps, ton, wait, fuel, reward) in enumerate(rows):
                f.write(f"{i},{steps},{ton:.2f},{wait:.2f},{fuel:.2f},{reward:.4f}\n")
