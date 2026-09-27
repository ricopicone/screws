"""Log: what a Scene records at every step."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["Log"]


class Log:
    """Per-step records of simulated time, joint state, command and end-effector frame.

    Arrays: t (N,), theta (N, n), dtheta (N, n), tau (N, n), command (N, n) and T_sb
    (N, 4, 4). Empty arrays before the first record.
    """

    def __init__(self):
        self._t: list[float] = []
        self._theta: list[np.ndarray] = []
        self._dtheta: list[np.ndarray] = []
        self._tau: list[np.ndarray] = []
        self._command: list[np.ndarray] = []
        self._T_sb: list[np.ndarray] = []

    def record(self, t, theta, dtheta, tau, command, T_sb) -> None:
        self._t.append(float(t))
        self._theta.append(np.asarray(theta, dtype=float))
        self._dtheta.append(np.asarray(dtheta, dtype=float))
        self._tau.append(np.asarray(tau, dtype=float))
        n = len(self._theta[-1])
        self._command.append(
            np.full(n, np.nan) if command is None else np.asarray(command, dtype=float)
        )
        self._T_sb.append(np.asarray(T_sb, dtype=float))

    def __len__(self) -> int:
        return len(self._t)

    @property
    def t(self) -> np.ndarray:
        return np.array(self._t)

    @property
    def theta(self) -> np.ndarray:
        return np.array(self._theta)

    @property
    def dtheta(self) -> np.ndarray:
        return np.array(self._dtheta)

    @property
    def tau(self) -> np.ndarray:
        return np.array(self._tau)

    @property
    def command(self) -> np.ndarray:
        return np.array(self._command)

    @property
    def T_sb(self) -> np.ndarray:
        return np.array(self._T_sb)

    def to_csv(self, path) -> None:
        """Write t, theta_i, dtheta_i, tau_i, command_i and the 12 entries of T_sb per row."""
        n = self.theta.shape[1] if len(self) else 0
        header = (
            ["t"]
            + [f"theta_{i + 1}" for i in range(n)]
            + [f"dtheta_{i + 1}" for i in range(n)]
            + [f"tau_{i + 1}" for i in range(n)]
            + [f"command_{i + 1}" for i in range(n)]
            + [f"T_{r}{c}" for r in range(3) for c in range(4)]
        )
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for k in range(len(self)):
                w.writerow(
                    [self._t[k], *self._theta[k], *self._dtheta[k], *self._tau[k],
                     *self._command[k], *self._T_sb[k][:3, :].reshape(-1)]
                )

    def to_mr_csv(self, path) -> None:
        """Write joint angles only, one row per step, no header: the MR wiki scenes' format."""
        np.savetxt(path, self.theta, delimiter=",")

    def plot(self):
        """One axis per quantity (theta, dtheta, tau, command). Needs matplotlib."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))
        for ax, (name, data) in zip(
            axes, [("theta", self.theta), ("dtheta", self.dtheta), ("tau", self.tau),
                   ("command", self.command)]
        ):
            if len(self):
                ax.plot(self.t, data)
            ax.set_ylabel(name)
            ax.grid(True)
        axes[-1].set_xlabel("t (s)")
        fig.tight_layout()
        return fig
