
#!/usr/bin/env python3
"""
CTMC comparison with explicit plotting: ITER-like vs STEP-like TF coil reliability

- Runs both scenarios and writes:
  * ctmc_outputs/ITER_like_transient.csv
  * ctmc_outputs/STEP_like_transient.csv
  * ctmc_outputs/comparison_Pfail.csv
  * ctmc_outputs/summary.csv

- Saves figures (PNG):
  * ctmc_outputs/comparison_Pfail.png      — overlay of P_fail(t)
  * ctmc_outputs/comparison_R.png          — overlay of Reliability R(t)
  * ctmc_outputs/ITER_like_states.png      — stacked area of state probabilities (ITER-like)
  * ctmc_outputs/STEP_like_states.png      — stacked area of state probabilities (STEP-like)

Author: (you)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import csv
import os

# --------------------------- Data classes ---------------------------

@dataclass
class SystemConfig:
    n_tf_coils: int = 12
    n_joints_total: int = 2880
    joints_per_coil: Optional[List[int]] = None  # if None, split evenly


@dataclass
class HazardParams:
    # Per-joint base hazards (per hour)
    lambda_quench_per_joint: float = 1e-7
    lambda_misalignment_per_joint: float = 5e-8

    # Cooling hazards
    lambda_cooling_per_coil: float = 1e-6       # per coil per hour
    lambda_cooling_system_cc: float = 2e-6      # system-level (common-cause) per hour

    # Progression to failure (within degraded states)
    mu_quench_to_fail: float = 3.6              # ~quench completion rate, h^-1
    mu_misalignment_to_fail: float = 1e-5
    mu_cooling_to_fail: float = 2e-4

    # Repairs (set to 0 to disable for pure reliability)
    repair_from_quench: float = 0.0
    repair_from_misalignment: float = 0.0
    repair_from_cooling: float = 0.0

    # Dependency multipliers (how much M or C increase quench initiation)
    alpha_MQ: float = 5.0
    alpha_CQ: float = 8.0
    alpha_common_cause_quench: float = 2.0  # added multiplier in MC state


@dataclass
class ModelParams:
    use_per_coil_cooling: bool = True
    use_system_cc_cooling: bool = True
    enable_combined_states: bool = True

    mission_hours: float = 1000.0
    dt_hours: float = 0.5

    # Monte Carlo settings
    mc_trials: int = 5000
    random_seed: int = 42


# --------------------------- CTMC Model ---------------------------

class CTMCModel:
    def __init__(self, config: SystemConfig, hazards: HazardParams, mp: ModelParams):
        self.config = config
        self.hz = hazards
        self.mp = mp

        # Distribute joints if not provided
        if self.config.joints_per_coil is None:
            avg = self.config.n_joints_total // self.config.n_tf_coils
            rem = self.config.n_joints_total % self.config.n_tf_coils
            jpc = [avg] * self.config.n_tf_coils
            for i in range(rem):
                jpc[i] += 1
            self.config.joints_per_coil = jpc

        # Build state space and generator
        self.states = self._define_states()
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.Q = self._build_generator()

    def _define_states(self) -> List[str]:
        if self.mp.enable_combined_states:
            return ["H", "M", "C", "Q", "MC", "QM", "QC", "F"]
        else:
            return ["H", "M", "C", "Q", "F"]

    def _aggregate_hazards(self) -> Dict[str, float]:
        n_coils = self.config.n_tf_coils
        joints_total = self.config.n_joints_total

        # Aggregate per-joint hazards to system level
        lambda_Q_sys = joints_total * self.hz.lambda_quench_per_joint
        lambda_M_sys = joints_total * self.hz.lambda_misalignment_per_joint

        # Cooling: per-coil + optional system-level CC
        lambda_C = 0.0
        if self.mp.use_per_coil_cooling:
            lambda_C += n_coils * self.hz.lambda_cooling_per_coil
        if self.mp.use_system_cc_cooling:
            lambda_C += self.hz.lambda_cooling_system_cc

        return {
            "lambda_Q_sys": lambda_Q_sys,
            "lambda_M_sys": lambda_M_sys,
            "lambda_C_sys": lambda_C,
        }

    def _build_generator(self) -> np.ndarray:
        S = len(self.states)
        Q = np.zeros((S, S), dtype=float)
        idx = self.state_index
        agg = self._aggregate_hazards()

        lQ = agg["lambda_Q_sys"]
        lM = agg["lambda_M_sys"]
        lC = agg["lambda_C_sys"]

        mu_QF = self.hz.mu_quench_to_fail
        mu_MF = self.hz.mu_misalignment_to_fail
        mu_CF = self.hz.mu_cooling_to_fail

        rQ = self.hz.repair_from_quench
        rM = self.hz.repair_from_misalignment
        rC = self.hz.repair_from_cooling

        a_MQ = self.hz.alpha_MQ
        a_CQ = self.hz.alpha_CQ
        a_ccQ = self.hz.alpha_common_cause_quench

        def add_rate(src: str, dst: str, rate: float):
            if rate <= 0:
                return
            i, j = idx[src], idx[dst]
            Q[i, j] += rate

        # Healthy
        add_rate("H", "Q", lQ)
        add_rate("H", "M", lM)
        add_rate("H", "C", lC)

        # Misalignment
        add_rate("M", "Q", lQ * a_MQ)
        add_rate("M", "F", mu_MF)
        add_rate("M", "H", rM)
        if "MC" in idx:
            add_rate("M", "MC", lC)

        # Cooling loss
        add_rate("C", "Q", lQ * a_CQ)
        add_rate("C", "F", mu_CF)
        add_rate("C", "H", rC)
        if "MC" in idx:
            add_rate("C", "MC", lM)

        # Quench
        add_rate("Q", "F", mu_QF)
        add_rate("Q", "H", rQ)
        if "QM" in idx:
            add_rate("Q", "QM", lM * 0.1)
        if "QC" in idx:
            add_rate("Q", "QC", lC * 0.1)

        if self.mp.enable_combined_states:
            # Misalignment + Cooling
            add_rate("MC", "Q", lQ * a_MQ * a_CQ * a_ccQ)
            add_rate("MC", "F", mu_MF + mu_CF)
            add_rate("MC", "M", rC)
            add_rate("MC", "C", rM)

            # Quench + Misalignment
            add_rate("QM", "F", mu_QF * 1.5)
            add_rate("QM", "M", rQ)

            # Quench + Cooling
            add_rate("QC", "F", mu_QF * 1.7)
            add_rate("QC", "C", rQ)

        # Make rows sum to zero
        for i in range(S):
            Q[i, i] = -np.sum(Q[i, :])

        return Q

    # --------------- Transient solver (RK4) ---------------

    def rk4_transient(self, t_end_hours: float, dt_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        S = len(self.states)
        Q = self.Q
        t = 0.0
        times = [t]
        p = np.zeros(S)
        p[self.state_index["H"]] = 1.0
        Ps = [p.copy()]

        def dp_dt(pvec):
            return pvec @ Q

        n_steps = int(np.ceil(t_end_hours / dt_hours))
        for _ in range(n_steps):
            h = dt_hours
            k1 = dp_dt(p)
            k2 = dp_dt(p + 0.5 * h * k1)
            k3 = dp_dt(p + 0.5 * h * k2)
            k4 = dp_dt(p + h * k3)
            p = p + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Numerical cleanup
            p = np.maximum(p, 0.0)
            s = np.sum(p)
            if s > 0:
                p /= s
            t += h
            times.append(t)
            Ps.append(p.copy())

        return np.array(times), np.vstack(Ps)

    def reliability(self, pvec: np.ndarray) -> float:
        return 1.0 - pvec[self.state_index["F"]]

    # --------------- Monte Carlo sanity check ---------------

    def _sample_exponential(self, rate: float, rng: np.random.Generator) -> float:
        if rate <= 0.0:
            return np.inf
        return rng.exponential(1.0 / rate)

    def monte_carlo_failure_prob(self) -> float:
        T = self.mp.mission_hours
        trials = self.mp.mc_trials
        rng = np.random.default_rng(self.mp.random_seed)

        agg = self._aggregate_hazards()
        lQ, lM, lC = agg["lambda_Q_sys"], agg["lambda_M_sys"], agg["lambda_C_sys"]

        failed = 0
        for _ in range(trials):
            t_now = 0.0
            state = "H"
            while t_now < T and state != "F":
                if state == "H":
                    tQ = self._sample_exponential(lQ, rng)
                    tM = self._sample_exponential(lM, rng)
                    tC = self._sample_exponential(lC, rng)
                    t_event = min(tQ, tM, tC)
                    if t_now + t_event > T:
                        break
                    state = "Q" if t_event == tQ else ("M" if t_event == tM else "C")
                    t_now += t_event

                elif state == "M":
                    tQ = self._sample_exponential(lQ * self.hz.alpha_MQ, rng)
                    tF = self._sample_exponential(self.hz.mu_misalignment_to_fail, rng)
                    tR = self._sample_exponential(self.hz.repair_from_misalignment, rng)
                    tC = self._sample_exponential(lC, rng)
                    t_event = min(tQ, tF, tR, tC)
                    if t_now + t_event > T:
                        break
                    if t_event == tQ:
                        state = "Q"
                    elif t_event == tF:
                        state = "F"
                    elif t_event == tR:
                        state = "H"
                    else:
                        state = "MC"
                    t_now += t_event

                elif state == "C":
                    tQ = self._sample_exponential(lQ * self.hz.alpha_CQ, rng)
                    tF = self._sample_exponential(self.hz.mu_cooling_to_fail, rng)
                    tR = self._sample_exponential(self.hz.repair_from_cooling, rng)
                    tM = self._sample_exponential(lM, rng)
                    t_event = min(tQ, tF, tR, tM)
                    if t_now + t_event > T:
                        break
                    if t_event == tQ:
                        state = "Q"
                    elif t_event == tF:
                        state = "F"
                    elif t_event == tR:
                        state = "H"
                    else:
                        state = "MC"
                    t_now += t_event

                elif state == "MC":
                    tQ = self._sample_exponential(
                        lQ * self.hz.alpha_MQ * self.hz.alpha_CQ * self.hz.alpha_common_cause_quench, rng)
                    tF_M = self._sample_exponential(self.hz.mu_misalignment_to_fail, rng)
                    tF_C = self._sample_exponential(self.hz.mu_cooling_to_fail, rng)
                    tR_M = self._sample_exponential(self.hz.repair_from_misalignment, rng)
                    tR_C = self._sample_exponential(self.hz.repair_from_cooling, rng)
                    t_event = min(tQ, tF_M, tF_C, tR_M, tR_C)
                    if t_now + t_event > T:
                        break
                    if t_event == tQ:
                        state = "Q"
                    elif t_event == tF_M or t_event == tF_C:
                        state = "F"
                    elif t_event == tR_M:
                        state = "C"
                    else:
                        state = "M"
                    t_now += t_event

                elif state == "Q":
                    tF = self._sample_exponential(self.hz.mu_quench_to_fail, rng)
                    tR = self._sample_exponential(self.hz.repair_from_quench, rng)
                    tM = self._sample_exponential(lM * 0.1, rng)
                    tC = self._sample_exponential(lC * 0.1, rng)
                    t_event = min(tF, tR, tM, tC)
                    if t_now + t_event > T:
                        break
                    if t_event == tF:
                        state = "F"
                    elif t_event == tR:
                        state = "H"
                    elif t_event == tM:
                        state = "QM"
                    else:
                        state = "QC"
                    t_now += t_event

                elif state == "QM":
                    tF = self._sample_exponential(self.hz.mu_quench_to_fail * 1.5, rng)
                    tR = self._sample_exponential(self.hz.repair_from_quench, rng)
                    t_event = min(tF, tR)
                    if t_now + t_event > T:
                        break
                    state = "F" if t_event == tF else "M"
                    t_now += t_event

                elif state == "QC":
                    tF = self._sample_exponential(self.hz.mu_quench_to_fail * 1.7, rng)
                    tR = self._sample_exponential(self.hz.repair_from_quench, rng)
                    t_event = min(tF, tR)
                    if t_now + t_event > T:
                        break
                    state = "F" if t_event == tF else "C"
                    t_now += t_event

                else:
                    break

            if state == "F":
                failed += 1

        return failed / trials


# --------------------------- Utilities ---------------------------

def save_transient_csv(path: str, times: np.ndarray, Ps: np.ndarray, state_names: List[str]):
    idxF = state_names.index("F")
    P_fail = Ps[:, idxF]
    R = 1.0 - P_fail

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["time_hours", "Reliability", "P_fail"] + [f"P_{s}" for s in state_names]
        w.writerow(header)
        for t, pvec, r, pf in zip(times, Ps, R, P_fail):
            row = [float(t), float(r), float(pf)] + [float(x) for x in pvec]
            w.writerow(row)


def run_scenario(name: str, config: SystemConfig, hazards: HazardParams, mp: ModelParams, outdir: str):
    model = CTMCModel(config, hazards, mp)
    times, Ps = model.rk4_transient(mp.mission_hours, mp.dt_hours)
    pT = Ps[-1, :]
    R_T = model.reliability(pT)
    P_fail_T = 1.0 - R_T
    mc_pf = model.monte_carlo_failure_prob()

    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{name}_transient.csv")
    save_transient_csv(csv_path, times, Ps, model.states)

    # Print a concise summary
    print(f"\n=== {name} ===")
    print(f"Mission time (h): {mp.mission_hours:.1f}")
    print(f"P_fail(T)  CTMC : {P_fail_T:.6e}")
    print(f"P_fail(T)   MC  : {mc_pf:.6e}")
    print(f"|Δ| (abs)       : {abs(P_fail_T - mc_pf):.6e}")
    print("State probabilities at T:")
    for s in model.states:
        print(f"  {s}: {pT[model.state_index[s]]:.6e}")

    return {
        "name": name,
        "times": times,
        "Ps": Ps,
        "states": model.states,
        "P_fail_T_ctmc": P_fail_T,
        "P_fail_T_mc": mc_pf,
        "R_T": R_T,
        "csv_path": csv_path,
    }


def write_comparison_csv(path: str, t1: np.ndarray, Pf1: np.ndarray, t2: np.ndarray, Pf2: np.ndarray,
                         name1: str, name2: str):
    assert len(t1) == len(t2), "Time grids differ; align before writing comparison."
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_hours", f"P_fail_{name1}", f"P_fail_{name2}"])
        for t, pf1, pf2 in zip(t1, Pf1, Pf2):
            w.writerow([float(t), float(pf1), float(pf2)])


def write_summary_csv(path: str, rows: List[Dict[str, float]]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "mission_hours", "P_fail_CTMC", "P_fail_MC", "Reliability_R(T)"])
        for r in rows:
            w.writerow([r["name"], r["mission_hours"], r["P_fail_T_ctmc"], r["P_fail_T_mc"], r["R_T"]])


# --------------------------- Plotting ---------------------------

def plot_comparison(outdir: str, res_iter: Dict, res_step: Dict):
    import matplotlib.pyplot as plt

    # Comparison overlay: P_fail(t)
    idxF_iter = res_iter["states"].index("F")
    idxF_step = res_step["states"].index("F")
    Pf_iter = res_iter["Ps"][:, idxF_iter]
    Pf_step = res_step["Ps"][:, idxF_step]

    plt.figure(figsize=(8, 4.8), dpi=120)
    plt.plot(res_iter["times"], Pf_iter, label="ITER-like P_fail(t)", lw=2)
    plt.plot(res_step["times"], Pf_step, label="STEP-like P_fail(t)", lw=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Failure probability")
    plt.title("CTMC: P_fail(t) Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png_pf = os.path.join(outdir, "comparison_Pfail.png")
    plt.savefig(out_png_pf)
    print(f"Saved plot: {out_png_pf}")

    # Comparison overlay: Reliability R(t)
    R_iter = 1.0 - Pf_iter
    R_step = 1.0 - Pf_step
    plt.figure(figsize=(8, 4.8), dpi=120)
    plt.plot(res_iter["times"], R_iter, label="ITER-like R(t)", lw=2)
    plt.plot(res_step["times"], R_step, label="STEP-like R(t)", lw=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Reliability")
    plt.title("CTMC: Reliability Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png_R = os.path.join(outdir, "comparison_R.png")
    plt.savefig(out_png_R)
    print(f"Saved plot: {out_png_R}")

def plot_states(outdir: str, res: Dict):
    import matplotlib.pyplot as plt
    times = res["times"]
    Ps = res["Ps"]
    labels = res["states"]
    colors = ["#2ca02c", "#ff7f0e", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#d62728"][:len(labels)]
    plt.figure(figsize=(9, 5.2), dpi=120)
    plt.stackplot(times, Ps.T, labels=labels, colors=colors, alpha=0.9)
    plt.xlabel("Time (hours)")
    plt.ylabel("State probability")
    plt.title(f"CTMC State Occupancy — {res['name']}")
    plt.legend(loc="upper left", ncol=2, frameon=True)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png = os.path.join(outdir, f"{res['name']}_states.png")
    plt.savefig(out_png)
    print(f"Saved plot: {out_png}")


# --------------------------- Main: scenarios ---------------------------

if __name__ == "__main__":
    OUTDIR = "ctmc_outputs"
    PLOT = True  # <-- plotting ON explicitly
    os.makedirs(OUTDIR, exist_ok=True)

    # Common system config
    config_common = SystemConfig(
        n_tf_coils=12,
        n_joints_total=2880,
        joints_per_coil=None
    )

    # Common model params
    mp_common = ModelParams(
        use_per_coil_cooling=True,
        use_system_cc_cooling=True,
        enable_combined_states=True,
        mission_hours=1000.0,   # <-- set your mission time
        dt_hours=0.5,
        mc_trials=5000,
        random_seed=42
    )

    # ---------------- Scenario A: ITER-like (illustrative) ----------------
    hazards_iter = HazardParams(
        lambda_quench_per_joint=2e-8,
        lambda_misalignment_per_joint=1e-8,
        lambda_cooling_per_coil=3e-7,
        lambda_cooling_system_cc=8e-7,
        mu_quench_to_fail=3.6,
        mu_misalignment_to_fail=5e-6,
        mu_cooling_to_fail=1e-4,
        repair_from_quench=0.0,
        repair_from_misalignment=0.0,
        repair_from_cooling=0.0,
        alpha_MQ=4.0,
        alpha_CQ=6.0,
        alpha_common_cause_quench=2.0
    )

    # ---------------- Scenario B: STEP-like (illustrative) ----------------
    hazards_step = HazardParams(
        lambda_quench_per_joint=6e-8,
        lambda_misalignment_per_joint=2e-8,
        lambda_cooling_per_coil=4e-7,
        lambda_cooling_system_cc=1.2e-6,
        mu_quench_to_fail=2.0,
        mu_misalignment_to_fail=8e-6,
        mu_cooling_to_fail=1.5e-4,
        repair_from_quench=0.0,
        repair_from_misalignment=0.0,
        repair_from_cooling=0.0,
        alpha_MQ=5.5,
        alpha_CQ=8.5,
        alpha_common_cause_quench=2.5
    )

    # Run both scenarios
    res_iter = run_scenario("ITER_like", config_common, hazards_iter, mp_common, OUTDIR)
    res_step = run_scenario("STEP_like", config_common, hazards_step, mp_common, OUTDIR)

    # Write comparison CSV (P_fail over time)
    idxF_iter = res_iter["states"].index("F")
    idxF_step = res_step["states"].index("F")
    Pf_iter = res_iter["Ps"][:, idxF_iter]
    Pf_step = res_step["Ps"][:, idxF_step]
    cmp_path = os.path.join(OUTDIR, "comparison_Pfail.csv")
    write_comparison_csv(cmp_path, res_iter["times"], Pf_iter, res_step["times"], Pf_step,
                         res_iter["name"], res_step["name"])
    print(f"\nSaved comparison time series: {cmp_path}")

    # Write end-of-mission summary
    summary_rows = [
        {
            "name": res_iter["name"],
            "mission_hours": mp_common.mission_hours,
            "P_fail_T_ctmc": res_iter["P_fail_T_ctmc"],
            "P_fail_T_mc": res_iter["P_fail_T_mc"],
            "R_T": res_iter["R_T"],
        },
        {
            "name": res_step["name"],
            "mission_hours": mp_common.mission_hours,
            "P_fail_T_ctmc": res_step["P_fail_T_ctmc"],
            "P_fail_T_mc": res_step["P_fail_T_mc"],
            "R_T": res_step["R_T"],
        },
    ]
    sum_path = os.path.join(OUTDIR, "summary.csv")
    write_summary_csv(sum_path, summary_rows)
    print(f"Saved end-of-mission summary: {sum_path}")

    # --------------- Explicit plotting calls ---------------
    if PLOT:
        try:
            plot_comparison(OUTDIR, res_iter, res_step)
            plot_states(OUTDIR, res_iter)
            plot_states(OUTDIR, res_step)
        except Exception as e:
            print(f"[PLOT] Skipped plotting due to error: {e}")
