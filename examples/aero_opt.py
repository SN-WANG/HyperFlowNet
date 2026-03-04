# Aero Optimization Solver Benchmark Platform
# Author: Shengning Wang

import os
import sys
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from scipy.optimize import OptimizeResult


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)


# Base Surrogate Model
from wsnet.models.classical.krg import KRG

# Ensemble Surrogate Models
from wsnet.models.ensemble.t_ahs import TAHS
from wsnet.models.ensemble.aes_msi import AESMSI

# Multi-fidelity Surrogate Models
from wsnet.models.multi_fidelity.mfs_mls import MFSMLS
from wsnet.models.multi_fidelity.mmfs import MMFS
from wsnet.models.multi_fidelity.cca_mfs import CCAMFS

# Sequential Sampling Methods
from wsnet.sampling.so_infill import SingleObjectiveInfill
from wsnet.sampling.mf_infill import MultiFidelityInfill
from wsnet.sampling.mo_infill import MultiObjectiveInfill

# Optimization Methods
from wsnet.models.optimization.dragonfly import dragonfly_optimize

# Tools for machine learning experiments
from wsnet.sampling.doe import lhs_design
from wsnet.utils.seeder import seed_everything
from wsnet.utils.hue_logger import hue, logger


# ----------------------------------------------------------------------
# Mock Utils (to match the requested style/dependencies)
# ----------------------------------------------------------------------

class AbaqusModel:
    """
    Mock AbaqusModel for debugging the testing platform.

    This class simulates the behavior of an external finite element simulation
    by mapping 3 input features to 4 output responses.
    """

    def __init__(self, fidelity: str = "high"):
        """
        initializes the mock simulation model.

        Args:
            fidelity (str): simulation fidelity, "high" or "low".
        """
        self.fidelity = fidelity
        # define input and output metadata to match get_simulation.py
        self.input_vars = ["thick1", "thick2", "thick3"]
        self.output_vars = ["weight", "displacement", "stress_skin", "stress_stiff"]

    def run(self, input_arr: np.ndarray) -> np.ndarray:
        """
        simulates the execution of an abaqus job.

        Args:
            input_arr (np.ndarray): input parameters. shape: (3,).

        Returns:
            np.ndarray: simulated results. shape: (4,).
        """
        # ensure input is a 1D array
        x = np.array(input_arr).flatten()
        if len(x) != 3:
            raise ValueError(f"expected 3 inputs, got {len(x)}")

        # define a baseline non-linear function (modified branin-like)
        def base_func(x1, x2):
            a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
            return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s

        # compute 4 different outputs based on the 3 inputs
        # output 1: weight (primary objective)
        y1 = base_func(x[0], x[1]) + 2.0 * x[2]

        # output 2: displacement
        y2 = 0.5 * x[0]**2 + 1.2 * x[1] + np.sin(x[2])

        # output 3: stress_skin
        y3 = base_func(x[1], x[2]) + 0.8 * x[0]

        # output 4: stress_stiff
        y4 = (x[0] - 5)**2 + (x[1] - 5)**2 + (x[2] - 5)**2

        # construct the result array
        res = np.array([y1, y2, y3, y4])

        # simulate low-fidelity error (bias and noise) if requested
        if self.fidelity == "low":
            res = 0.85 * res + 5.0 + np.random.normal(0, 0.1, size=res.shape)

        return res


# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def scale_to_bounds(x_norm: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Scales normalized LHS samples [0, 1] to physical bounds.

    Args:
        x_norm (np.ndarray): normalized samples. shape: (n, dim).
        bounds (np.ndarray): physical bounds. shape: (dim, 2).

    Returns:
        np.ndarray: scaled samples. shape: (n, dim).
    """
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    return lower + x_norm * (upper - lower)


def run_abaqus_batch(x: np.ndarray, fidelity: str = "high") -> np.ndarray:
    """
    Executes the AbaqusModel in a loop for a batch of samples.

    Args:
        x (np.ndarray): input samples. shape: (n_samples, num_features).
        fidelity (str): simulation fidelity, "high" or "low".

    Returns:
        np.ndarray: collected outputs. shape: (n_samples, num_outputs).
    """
    num_samples = x.shape[0]
    results = []

    # instantiate model with specific fidelity
    model = AbaqusModel(fidelity=fidelity)

    logger.info(f"running abaqus batch (fidelity={fidelity}, samples={num_samples})...")

    for i in range(num_samples):
        # run single simulation: input x[i] is (num_features,), output is (num_outputs,)
        y_out = model.run(x[i])
        results.append(y_out)

    return np.array(results)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str):
    """
    Calculates and logs performance metrics.

    Args:
        y_true (np.ndarray): ground truth. shape: (n, out).
        y_pred (np.ndarray): predictions. shape: (n, out).
        label (str): model identifier label.
    """
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    logger.info(f"--- {label} Performance ---")
    logger.info(f"R2  : {hue.m}{r2:.6f}{hue.q}")
    logger.info(f"MSE : {hue.m}{mse:.6f}{hue.q}")
    logger.info(f"RMSE: {hue.m}{rmse:.6f}{hue.q}")
    print("") # spacer


# ----------------------------------------------------------------------
# Main Platform Logic
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Configuration & Initialization
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}initializing test platform...{hue.q}")

    # problem definition based on abaqus model
    # inputs: thick1, thick2, thick3
    # outputs: weight, displacement, stress_skin, stress_stiff
    num_features = 3
    num_outputs = 4

    # define engineering bounds for thickness [min, max]
    # adjusting bounds to encompass the example input [5.0, 6.0, 4.0]
    bounds = np.array([
        [4.0, 10.0],  # thick1
        [4.0, 10.0],  # thick2
        [4.0, 10.0]   # thick3
    ])

    # optimization target: index 0 (weight)
    target_index = 0

    # experiment settings
    num_train = 20        # standard high-fidelity training samples
    num_test = 10         # testing samples (reduced count for speed in demo)
    num_lf = 30           # low-fidelity samples (for mmfs)
    num_hf = 10           # high-fidelity samples (for mmfs)
    num_infill = 5        # number of sequential sampling iterations

    seed_everything(42)

    # ------------------------------------------------------------------
    # 2. Design of Experiments (DoE)
    # ------------------------------------------------------------------
    logger.info("generating samples via LHS...")

    # generate normalized samples [0, 1]
    x_train_norm = lhs_design(num_train, num_features, iterations=50)
    x_test_norm = lhs_design(num_test, num_features, iterations=50)

    # mmfs specific sampling
    x_lf_norm = lhs_design(num_lf, num_features, iterations=50)
    x_hf_norm = lhs_design(num_hf, num_features, iterations=50)

    # scale to physical bounds
    x_train = scale_to_bounds(x_train_norm, bounds)
    x_test = scale_to_bounds(x_test_norm, bounds)
    x_lf = scale_to_bounds(x_lf_norm, bounds)
    x_hf = scale_to_bounds(x_hf_norm, bounds)

    # evaluate external function (abaqus) to get targets
    # standard datasets (high fidelity default)
    y_train = run_abaqus_batch(x_train, fidelity="high")
    y_test = run_abaqus_batch(x_test, fidelity="high")

    # multi-fidelity datasets
    y_lf = run_abaqus_batch(x_lf, fidelity="low")
    y_hf = run_abaqus_batch(x_hf, fidelity="high")

    logger.info(f"{hue.g}data generation complete.{hue.q} train shape: {hue.m}{x_train.shape}{hue.q}")

    # ------------------------------------------------------------------
    # 3. Model A: T-AHS (Two-Stage Adaptive Hybrid Surrogate)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model A: T-AHS{hue.q}")

    model_tahs = TAHS()
    model_tahs.fit(x_train, y_train)

    y_pred_tahs = model_tahs.predict(x_test)
    evaluate_metrics(y_test, y_pred_tahs, "T-AHS")

    # ------------------------------------------------------------------
    # 4. Model B: AES-MSI (Adaptive Ensemble of Surrogate Models by Minimum Screening Index)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model B: AES-MSI{hue.q}")

    model_aesmsi = AESMSI()
    model_aesmsi.fit(x_train, y_train)

    y_pred_aesmsi = model_aesmsi.predict(x_test)
    evaluate_metrics(y_test, y_pred_aesmsi, "AES-MSI")

    # ------------------------------------------------------------------
    # 5. Model C: MFS-MLS (Multi-Fidelity Surrogate Model based on Moving Least Squares)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model C: MFS-MLS{hue.q}")

    model_mfsmls = MFSMLS()
    model_mfsmls.fit(x_lf, y_lf, x_hf, y_hf)

    y_pred_mfsmls = model_mfsmls.predict(x_test)
    evaluate_metrics(y_test, y_pred_mfsmls, "MFS-MLS")

    # ------------------------------------------------------------------
    # 6. Model D: MMFS (Modified Multi-Fidelity Surrogate)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model D: MMFS{hue.q}")

    model_mmfs = MMFS()
    model_mmfs.fit(x_lf, y_lf, x_hf, y_hf)

    y_pred_mmfs = model_mmfs.predict(x_test)
    evaluate_metrics(y_test, y_pred_mmfs, "MMFS")

    # ------------------------------------------------------------------
    # 7. Model E: CCA-MFS (Multi-Fidelity Surrogate Model based on Canonical Correlation Analysis and Least Squares)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model E: CCA-MFS{hue.q}")

    model_ccamfs = CCAMFS()
    model_ccamfs.fit(x_lf, y_lf, x_hf, y_hf)

    y_pred_ccamfs = model_ccamfs.predict(x_test)
    evaluate_metrics(y_test, y_pred_ccamfs, "CCA-MFS")

    # ------------------------------------------------------------------
    # 8. Model F: KRG + Sequential Infill (Active Learning)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model F: KRG with Infill{hue.q}")

    # initial krg training
    # using a copy of training data to allow appending without affecting original sets
    x_current = np.copy(x_train)
    y_current = np.copy(y_train)

    model_krg = KRG()
    model_krg.fit(x_current, y_current)

    # infill loop
    for i in range(num_infill):
        # instantiate infill strategy (ei: expected improvement)
        strategy = SingleObjectiveInfill(
            model=model_krg, bounds=bounds, y_train=y_current, criterion="ei", target_index=target_index
        )

        # propose new candidate
        x_new = strategy.propose() # shape (1, num_features)

        # evaluate candidate using abaqus (high fidelity)
        # flatten x_new for the run method, then reshape result
        y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
        y_new = y_new_val.reshape(1, -1)

        # check for failure (nan) before appending
        if np.isnan(y_new).any():
            logger.info(f"{hue.r}simulation failed (returned NaN). skipping update.{hue.q}")
            continue

        # update dataset
        x_current = np.vstack([x_current, x_new])
        y_current = np.vstack([y_current, y_new])

        # refit model
        model_krg.fit(x_current, y_current)

    # final prediction after active learning
    y_pred_krg, _ = model_krg.predict(x_test)
    evaluate_metrics(y_test, y_pred_krg, "KRG + Infill")

    # ------------------------------------------------------------------
    # 9. Model G: KRG + MICOInfill (MICO Sequential Sampling)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model G: KRG with MICOInfill{hue.q}")

    # start from the 10 HF samples to demonstrate MICO's multi-fidelity advantage
    x_mico = np.copy(x_hf)
    y_mico = np.copy(y_hf)

    model_krg_mico = KRG()
    model_krg_mico.fit(x_mico, y_mico)

    for i in range(num_infill):
        mico_strategy = MultiFidelityInfill(
            model=model_krg_mico,
            x_hf=x_mico,
            y_hf=y_mico,
            x_lf=x_lf,
            y_lf=y_lf,
            target_index=target_index,
            ratio=0.5,
        )

        x_new = mico_strategy.propose()  # shape (1, num_features)

        y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
        y_new = y_new_val.reshape(1, -1)

        if np.isnan(y_new).any():
            logger.info(f"{hue.r}simulation failed (returned NaN). skipping update.{hue.q}")
            continue

        x_mico = np.vstack([x_mico, x_new])
        y_mico = np.vstack([y_mico, y_new])
        model_krg_mico.fit(x_mico, y_mico)

    y_pred_krg_mico, _ = model_krg_mico.predict(x_test)
    evaluate_metrics(y_test, y_pred_krg_mico, "KRG + MICOInfill")

    # ------------------------------------------------------------------
    # 10. Model H: KRG + MOInfill (Constrained Many-Objective IS-EHVI)
    # ------------------------------------------------------------------
    logger.info(f"{hue.b}>>> Model H: KRG with MOInfill (Constrained IS-EHVI){hue.q}")

    obj_indices        = [2, 3]   # stress_skin, stress_stiff → minimize
    constraint_indices = [0]      # weight → constrain
    constraint_ubs = np.array([
        np.percentile(y_train[:, 0], 50),   # weight upper bound (mock-scale adaptation of Isight's 0.31)
    ])

    x_mo = np.copy(x_train)
    y_mo = np.copy(y_train)
    model_krg_mo = KRG()
    model_krg_mo.fit(x_mo, y_mo)

    for i in range(num_infill):
        mo_strategy = MultiObjectiveInfill(
            model=model_krg_mo, bounds=bounds, y_train=y_mo,
            obj_indices=obj_indices,
            constraint_indices=constraint_indices,
            constraint_ubs=constraint_ubs,
        )
        x_new = mo_strategy.propose()        # (1, num_features)
        y_new_val = AbaqusModel(fidelity="high").run(x_new.flatten())
        y_new = y_new_val.reshape(1, -1)
        if np.isnan(y_new).any():
            logger.info(f"{hue.r}simulation failed (NaN). skipping.{hue.q}")
            continue
        x_mo = np.vstack([x_mo, x_new])
        y_mo = np.vstack([y_mo, y_new])
        model_krg_mo.fit(x_mo, y_mo)
        logger.info(f"  iteration {i+1}/{num_infill}: stress_skin={y_new[0,2]:.4f}, stress_stiff={y_new[0,3]:.4f}")

    # report Pareto front (objectives only)
    y_mo_obj = y_mo[:, obj_indices]
    pf_mask = mo_strategy._compute_pareto_mask(y_mo_obj)
    y_pf = y_mo_obj[pf_mask]
    logger.info(f"Pareto front ({pf_mask.sum()} points): stress_skin, stress_stiff")
    for pt in y_pf:
        logger.info(f"  stress_skin={pt[0]:.4f}, stress_stiff={pt[1]:.4f}")

    # surrogate accuracy (all outputs)
    y_pred_krg_mo, _ = model_krg_mo.predict(x_test)
    evaluate_metrics(y_test, y_pred_krg_mo, "KRG + MOInfill")

    # ------------------------------------------------------------------
    # 11. Model I & Model J: Constrained Optimization on Surrogate Surface
    # ------------------------------------------------------------------

    opt_single_idx = 2          # stress_skin (Von Mises) — Isight single-obj target
    mo_obj_indices  = [2, 3]   # stress_skin, stress_stiff — multi-obj targets
    scipy_bounds    = [(bounds[i, 0], bounds[i, 1]) for i in range(num_features)]
    weight_ub       = float(np.percentile(y_train[:, 0], 50))  # Isight: weight <= 0.31

    def _single_obj(x_vec: np.ndarray) -> float:
        """Predict stress_skin (mises) from KRG surrogate.

        Args:
            x_vec (np.ndarray): design point. shape: (num_features,).

        Returns:
            float: predicted Von Mises stress at x_vec.
        """
        pred, _ = model_krg.predict(x_vec.reshape(1, -1))
        return float(pred[0, opt_single_idx])

    def _multi_obj(x_vec: np.ndarray) -> np.ndarray:
        """Predict [stress_skin, stress_stiff] from KRG surrogate.

        Args:
            x_vec (np.ndarray): design point. shape: (num_features,).

        Returns:
            np.ndarray: predicted [stress_skin, stress_stiff]. shape: (2,).
        """
        pred, _ = model_krg.predict(x_vec.reshape(1, -1))
        return pred[0, mo_obj_indices]

    weight_con = NonlinearConstraint(
        fun=lambda x: float(model_krg.predict(x.reshape(1, -1))[0][0, 0]),
        lb=-np.inf, ub=weight_ub,
    )

    # ---- Model I: DE ----
    logger.info(f"{hue.b}>>> Model I: DE{hue.q}")

    ri_s: OptimizeResult = differential_evolution(
        func=_single_obj, bounds=scipy_bounds, constraints=weight_con,
        strategy="best1bin", maxiter=50, popsize=10, tol=1e-6, seed=42,
    )
    true_i_s = AbaqusModel(fidelity="high").run(ri_s.x)
    logger.info(f"  [Single-obj] best x : {ri_s.x}")
    logger.info(f"  predicted mises     : {hue.c}{ri_s.fun:.6f}{hue.q}  |  verified: {hue.g}{true_i_s[opt_single_idx]:.6f}{hue.q}")
    logger.info(f"  verified weight     : {true_i_s[0]:.6f}  (ub={weight_ub:.4f})")

    def _de_mo_scalarized(x_vec: np.ndarray) -> float:
        return float(np.sum(_multi_obj(x_vec)))  # equal-weight scalarization

    ri_m: OptimizeResult = differential_evolution(
        func=_de_mo_scalarized, bounds=scipy_bounds, constraints=weight_con,
        strategy="best1bin", maxiter=50, popsize=10, tol=1e-6, seed=42,
    )
    true_i_m = AbaqusModel(fidelity="high").run(ri_m.x)
    logger.info(f"  [Multi-obj]  best x : {ri_m.x}")
    logger.info(f"  predicted [sk, ss]  : {_multi_obj(ri_m.x)}")
    logger.info(f"  verified  [sk, ss]  : {hue.g}{true_i_m[mo_obj_indices]}{hue.q}")
    print("")

    # ---- Model J: CFSSDA ----
    logger.info(f"{hue.b}>>> Model J: CFSSDA{hue.q}")

    rj_s = dragonfly_optimize(
        func=_single_obj, bounds=scipy_bounds, constraints=weight_con,
        maxiter=120, popsize=20, tol=1e-6, seed=42, multi_objective=False,
    )
    true_j_s = AbaqusModel(fidelity="high").run(rj_s.x)
    logger.info(f"  [Single-obj] best x : {rj_s.x}")
    logger.info(f"  predicted mises     : {hue.c}{rj_s.fun:.6f}{hue.q}  |  verified: {hue.g}{true_j_s[opt_single_idx]:.6f}{hue.q}")
    logger.info(f"  verified weight     : {true_j_s[0]:.6f}  (ub={weight_ub:.4f})")

    rj_m = dragonfly_optimize(
        func=_multi_obj, bounds=scipy_bounds, constraints=weight_con,
        maxiter=120, popsize=20, tol=1e-6, seed=42,
        multi_objective=True, scalarization="weighted_sum", return_pareto=True,
    )
    true_j_m = AbaqusModel(fidelity="high").run(rj_m.x)
    logger.info(f"  [Multi-obj]  best x : {rj_m.x}")
    logger.info(f"  predicted [sk, ss]  : {_multi_obj(rj_m.x)}")
    logger.info(f"  verified  [sk, ss]  : {hue.g}{true_j_m[mo_obj_indices]}{hue.q}")
    if hasattr(rj_m, "pareto_f") and rj_m.pareto_f is not None:
        logger.info(f"  Pareto front ({len(rj_m.pareto_f)} pts):")
        for pt in rj_m.pareto_f:
            logger.info(f"    stress_skin={pt[0]:.4f}, stress_stiff={pt[1]:.4f}")

    logger.info(f"{hue.b}process completed successfully.{hue.q}")
