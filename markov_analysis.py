
"""
CTMC reliability model for a TF coil with event-dependent failure modes.

System overview:
- 12 TF coils (configurable), each with N_joints (e.g., 2880 total joints).
- Failure modes: QUENCH, MISALIGNMENT, COOLING LOSS.
- States: Healthy (H), Misalignment (M), Cooling loss (C), Quench (Q), combined intermediate states,
          and Failed (F, absorbing).
- Dependencies: In M or C, the quench hazard is multiplied by factors (alpha_MQ, alpha_CQ).
- Repairs: Optional transitions back to healthier states with repair rates.

Key outputs:
- Transient probability vector p(t) over mission time.
- Probability of failure by time T (p_F(T)).
- Reliability R(T) = 1 - p_F(T).
- Monte Carlo sanity check.

Assumptions:
- Hazards are exponential (memoryless) → CTMC is appropriate.
- Independence at joint level unless captured via multipliers/common-cause rates.
- Cooling-pipe hazards may be coil-level rather than per-joint (configurable).

Author: (you)
"""

from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional


@dataclass
class SystemConfig:
    n_tf_coils: int = 12
    n_joints_total: int = 2880  # total joints across all coils
    # Distribution of joints per coil (optional). If None, assume equal split.
    joints_per_coil: Optional[List[int]] = None


@dataclass
class HazardParams:
    # Per-joint base hazards (per hour, example units; set as needed)
    lambda_quench_per_joint: float = 1e-7      # quench initiation per joint
    lambda_misalignment_per_joint: float = 5e-8  # misalignment initiation per joint

    # Cooling pipes may be per coil or system-level. Use BOTH and choose via flags in ModelParams.
    lambda_cooling_per_coil: float = 1e-6     # cooling-loss initiation per coil
    lambda_cooling_system_cc: float = 2e-6    # common-cause cooling-loss rate affecting many coils at once

    # Escalation to failure (within a degraded state), e.g., quench → failed
    mu_quench_to_fail: float = 1e-3           # failure completion rate from quench
    mu_misalignment_to_fail: float = 1e-5     # failure completion rate from misalignment
    mu_cooling_to_fail: float = 2e-4          # failure completion rate from cooling loss

    # Repairs (optional; set to 0 to disable)
    repair_from_quench: float = 5e-4
    repair_from_misalignment: float = 1e-3
    repair_from_cooling: float = 8e-4

    # Dependency multipliers (hazard coupling)
    alpha_MQ: float = 5.0     # multiplier: misalignment increases quench rate
    alpha_CQ: float = 8.0     # multiplier: cooling loss increases quench rate

    # Optional common-cause coupling factors for quench when cooling or misalignment present
    alpha_common_cause_quench: float = 2.0


@dataclass
class ModelParams:
    # Whether cooling hazard is per coil or a system-level common-cause in addition to per-coil
    use_per_coil_cooling: bool = True
    use_system_cc_cooling: bool = True

    # Whether to allow combined degraded states (M+C, Q+M, Q+C). If False, simplify the chain.
    enable_combined_states: bool = True

    # Mission time horizon and integrator settings
    mission_hours: float = 1000.0
    dt_hours: float = 0.5   # time step for RK4 integrator
    # Monte Carlo settings
    mc_trials: int = 5000
    random_seed: int = 42


class CTMCModel:
    """
    Build and solve a CTMC for event-dependent failure modes of a TF coil system.
    """

    def __init__(self, config: SystemConfig, hazards: HazardParams, mp: ModelParams):
        self.config = config
        self.hz = hazards
        self.mp = mp

        # Joints per coil allocation
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
        """
        Define system-level states.
        We track system condition at the aggregate level. States:
        H  : Healthy
        M  : Misalignment present (system-level indication)
        C  : Cooling loss present
        Q  : Quench in progress
        MC : Misalignment + Cooling loss
        QM : Quench + Misalignment
        QC : Quench + Cooling loss
        F  : Failed (absorbing)
        """
        if self.mp.enable_combined_states:
            return ["H", "M", "C", "Q", "MC", "QM", "QC", "F"]
        else:
            # Simplified chain without combined states
            return ["H", "M", "C", "Q", "F"]

    def _aggregate_hazards(self) -> Dict[str, float]:
        """
        Compute aggregate system-level transition rates from base hazards and configuration.
        """
        n_coils = self.config.n_tf_coils
        joints_total = self.config.n_joints_total
        joints_per_coil = self.config.joints_per_coil

        # Aggregate per-joint hazards
        lambda_Q_sys = joints_total * self.hz.lambda_quench_per_joint
        lambda_M_sys = joints_total * self.hz.lambda_misalignment_per_joint

        # Cooling hazards
        lambda_C_per_coil_total = 0.0
        if self.mp.use_per_coil_cooling:
            # Sum per coil hazard
            lambda_C_per_coil_total = n_coils * self.hz.lambda_cooling_per_coil

        lambda_C_cc_sys = self.hz.lambda_cooling_system_cc if self.mp.use_system_cc_cooling else 0.0

        return {
            "lambda_Q_sys": lambda_Q_sys,
            "lambda_M_sys": lambda_M_sys,
            "lambda_C_sys": lambda_C_per_coil_total + lambda_C_cc_sys,
        }

    def _build_generator(self) -> np.ndarray:
        """
        Build the CTMC generator matrix Q (rows sum to zero).
        State order defined by self.states.
        """
        S = len(self.states)
        Q = np.zeros((S, S), dtype=float)
        idx = self.state_index
        agg = self._aggregate_hazards()

        # Base hazards
        lQ = agg["lambda_Q_sys"]
        lM = agg["lambda_M_sys"]
        lC = agg["lambda_C_sys"]

        # Failure progression rates
        mu_QF = self.hz.mu_quench_to_fail
        mu_MF = self.hz.mu_misalignment_to_fail
        mu_CF = self.hz.mu_cooling_to_fail

        # Repairs
        rQ = self.hz.repair_from_quench
        rM = self.hz.repair_from_misalignment
        rC = self.hz.repair_from_cooling

        # Multipliers for dependency
        a_MQ = self.hz.alpha_MQ
        a_CQ = self.hz.alpha_CQ
        a_ccQ = self.hz.alpha_common_cause_quench

        # Helper to add rate
        def add_rate(src: str, dst: str, rate: float):
            if rate <= 0:
                return
            i, j = idx[src], idx[dst]
            Q[i, j] += rate

        # Healthy transitions
        add_rate("H", "Q", lQ)
        add_rate("H", "M", lM)
        add_rate("H", "C", lC)

        if "F" in idx:
            # Allow direct rare common-cause to failure if desired (set via mu_* or add custom)
            pass

        # From M (misalignment):
        # Increased quench hazard
        add_rate("M", "Q", lQ * a_MQ)
        # Natural progression to failure from M
        add_rate("M", "F", mu_MF)
        # Repair back to H
        add_rate("M", "H", rM)
        # If combined states enabled, allow cooling to also occur while misaligned
        if "MC" in idx:
            add_rate("M", "MC", lC)

        # From C (cooling loss):
        add_rate("C", "Q", lQ * a_CQ)
        add_rate("C", "F", mu_CF)
        add_rate("C", "H", rC)
        if "MC" in idx:
            add_rate("C", "MC", lM)  # misalignment while cooling impaired

        # From Q (quench):
        # Progress to failure
        add_rate("Q", "F", mu_QF)
        # Repairs back to H (e.g., safe shutdown & recovery)
        add_rate("Q", "H", rQ)
        # Allow that while in quench, cooling/misalignment can also be detected (optional small rates)
        if self.mp.enable_combined_states:
            add_rate("Q", "QM", lM * 0.1)  # rare concurrent misalignment detection
            add_rate("Q", "QC", lC * 0.1)  # rare concurrent cooling fault detection

        if self.mp.enable_combined_states:
            # From MC (misalignment + cooling loss)
            # Quench hazard gets both multipliers + optional common-cause multiplier
            add_rate("MC", "Q", lQ * a_MQ * a_CQ * a_ccQ)
            add_rate("MC", "F", mu_MF + mu_CF)  # either may push to failure
            # Repairs: assume if either is repaired, we return to single-degraded states or H
            add_rate("MC", "M", rC)  # cooling repaired, still misaligned
            add_rate("MC", "C", rM)  # misalignment corrected, still cooling loss
            # From QM (quench + misalignment)
            add_rate("QM", "F", mu_QF * (1 + 0.5))  # faster failure under combined stress
            add_rate("QM", "M", rQ)  # quench resolved, misalignment remains
            # From QC (quench + cooling loss)
            add_rate("QC", "F", mu_QF * (1 + 0.7))
            add_rate("QC", "C", rQ)

        # Absorbing failure state: no exits
        # Finalize row sums to zero
        for i in range(S):
            Q[i, i] = -np.sum(Q[i, :])

        return Q

    def rk4_transient(self, t_end_hours: float, dt_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dp/dt = p * Q with RK4 over [0, t_end].
        Returns: times, probabilities matrix (len(times) x S)
        """
        S = len(self.states)
        Q = self.Q
        t = 0.0
        times = [t]
        p = np.zeros(S)
        p[self.state_index["H"]] = 1.0  # start healthy
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
                p = p / s
            t += h
            times.append(t)
            Ps.append(p.copy())

        return np.array(times), np.vstack(Ps)

    def reliability(self, pvec: np.ndarray) -> float:
        """
        Reliability R = 1 - P(Failed).
        """
        return 1.0 - pvec[self.state_index["F"]]

    def run_mission(self) -> Dict[str, float]:
        """
        Integrate over mission time and return end-state metrics.
        """
        T = self.mp.mission_hours
        dt = self.mp.dt_hours
        times, Ps = self.rk4_transient(T, dt)
        pT = Ps[-1, :]
        return {
            "mission_hours": T,
            "P_failed": float(pT[self.state_index["F"]]),
            "Reliability": float(self.reliability(pT)),
            "state_probs": {s: float(pT[self.state_index[s]]) for s in self.states},
        }

    # ----------------------- Monte Carlo sanity check -----------------------

    def _sample_exponential(self, rate: float, rng: np.random.Generator) -> float:
        if rate <= 0.0:
            return np.inf
        return rng.exponential(1.0 / rate)

    def monte_carlo_failure_prob(self) -> float:
        """
        Simple discrete-event Monte Carlo to estimate P(Failure by T).
        This is a coarse sanity check, not a full CTMC sampler of all path dynamics.
        """
        T = self.mp.mission_hours
        trials = self.mp.mc_trials
        rng = np.random.default_rng(self.mp.random_seed)

        agg = self._aggregate_hazards()
        lQ = agg["lambda_Q_sys"]
        lM = agg["lambda_M_sys"]
        lC = agg["lambda_C_sys"]

        # For simplicity, simulate earliest event times and apply dependency logic
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
                    if t_event == tQ:
                        state = "Q"
                    elif t_event == tM:
                        state = "M"
                    else:
                        state = "C"
                    t_now += t_event
                elif state == "M":
                    # Quench hazard increases
                    tQ = self._sample_exponential(lQ * self.hz.alpha_MQ, rng)
                    tF = self._sample_exponential(self.hz.mu_misalignment_to_fail, rng)
                    tR = self._sample_exponential(self.hz.repair_from_misalignment, rng)
                    tC = self._sample_exponential(lC, rng)  # may move to MC
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
                    # Strongly elevated quench + two ways to fail or repair towards single degraded
                    tQ = self._sample_exponential(lQ * self.hz.alpha_MQ * self.hz.alpha_CQ * self.hz.alpha_common_cause_quench, rng)
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
                    # Progress to failure, or repair. Rarely transition to combined states.
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
                    # F absorbing
                    break

            if state == "F":
                failed += 1

        return failed / trials


# --------------------------- Example usage ---------------------------

if __name__ == "__main__":
    # Configure system and hazards (edit these to your case)
    config = SystemConfig(
        n_tf_coils=12,
        n_joints_total=2880,
        joints_per_coil=None  # will be auto-distributed
    )

    hazards = HazardParams(
        lambda_quench_per_joint=1e-7,       # per joint per hour
        lambda_misalignment_per_joint=5e-8, # per joint per hour
        lambda_cooling_per_coil=1e-6,       # per coil per hour
        lambda_cooling_system_cc=2e-6,      # system-level per hour
        mu_quench_to_fail=1e-3,
        mu_misalignment_to_fail=1e-5,
        mu_cooling_to_fail=2e-4,
        repair_from_quench=5e-4,
        repair_from_misalignment=1e-3,
        repair_from_cooling=8e-4,
        alpha_MQ=5.0,
        alpha_CQ=8.0,
        alpha_common_cause_quench=2.0
    )

    mp = ModelParams(
        use_per_coil_cooling=True,
        use_system_cc_cooling=True,
        enable_combined_states=True,
        mission_hours=1000.0,
        dt_hours=0.5,
        mc_trials=5000,
        random_seed=42
    )

    model = CTMCModel(config, hazards, mp)

    # Solve CTMC transient and report mission reliability
    results = model.run_mission()
    print("=== CTMC transient results ===")
    print(f"Mission time (h): {results['mission_hours']:.1f}")
    print(f"P(Failure by T): {results['P_failed']:.6f}")
    print(f"Reliability R(T): {results['Reliability']:.6f}")
    print("State probabilities at T:")
    for s, p in results["state_probs"].items():
        print(f"  {s}: {p:.6f}")

    # Monte Carlo sanity check
    pc_mc = model.monte_carlo_failure_prob()
    print("\n=== Monte Carlo estimate ===")
    print(f"P(Failure by T) ~ {pc_mc:.6f}")

    # Quick consistency note
    print("\nNote: CTMC transient and MC results should be close for these settings.")
