"""Callbacks personnalisés pour l'entraînement PPO.

Enregistre les métriques clés (KPIs) à intervalles réguliers
pour produire les learning curves (Section 4.6 du mémoire).
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class KPILoggerCallback(BaseCallback):
    """Enregistre les KPIs de l'environnement minier dans un CSV."""

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

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]
                self._episode_tonnages.append(info.get("total_tonnage_t", 0.0))
                self._episode_waits.append(info.get("total_wait_min", 0.0))
                self._episode_fuels.append(info.get("total_fuel_l", 0.0))

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

    def _on_training_end(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            f.write("episode,tonnage_t,wait_min,fuel_l\n")
            for i, (ton, wait, fuel) in enumerate(
                zip(
                    self._episode_tonnages,
                    self._episode_waits,
                    self._episode_fuels,
                )
            ):
                f.write(f"{i},{ton:.2f},{wait:.2f},{fuel:.2f}\n")
