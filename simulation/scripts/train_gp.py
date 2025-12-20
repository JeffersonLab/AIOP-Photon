#!/usr/bin/env python3
"""
train_gp.py
Train a Gaussian Process surrogate on full-physics data.

Run:
python -m gonio_sim.scripts.train_gp --data gp_train.jsonl --model gp_model.joblib
"""

import argparse, json
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
    

# Keep the feature order STABLE (must match gen_gp_data)
BASE_KEYS = [
    "yaw_rb","pitch_rb","a_h","a_v","dir_h","dir_v","run_h","run_v",
    "E_ma","E_std","sin_y","cos_y","sin_p","cos_p"
]

def flatten_feat(d, k_hist):
    x = [d[k] for k in BASE_KEYS]
    x.extend(d["ah_hist"][:k_hist])
    x.extend(d["av_hist"][:k_hist])
    return np.array(x, dtype=np.float64)

def load_jsonl(path, k_hist):
    X, y = [], []
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            X.append(flatten_feat(obj["x"], k_hist))
            y.append(obj["y"])
    return np.vstack(X), np.array(y, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL from gen_gp_data.py")
    ap.add_argument("--model", default="gp_model.joblib")
    ap.add_argument("--scaler", default="gp_scaler.joblib")
    ap.add_argument("--k-hist", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=15000,
                    help="cap training size to keep GP tractable (O(n^3))")
    args = ap.parse_args()

    X, y = load_jsonl(args.data, args.k_hist)
    if len(X) > args.max_train:
        X, y = X[:args.max_train], y[:args.max_train]
        print(f"Truncated to {len(X)} samples for exact GP tractability.")

    # Split time-wise (don’t shuffle) to avoid leaking sequences
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Standardize features (very important for GP lengthscales)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Composite kernel: amplitude * (RBF + RationalQuadratic) + white noise
    kernel = C(1.0, (1e-2, 1e3)) * (RBF(length_scale=np.ones(X.shape[1])) +
                                     RationalQuadratic(alpha=1.0, length_scale=1.0)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,            # jitter for numerical stability
        normalize_y=True,      # center y
        n_restarts_optimizer=4,
        random_state=0
    )

    print("Fitting GP...")
    gp.fit(Xtr_s, ytr)

    # --------------------------------------------------------
    # Evaluate GP and create diagnostic plots
    # --------------------------------------------------------
    print("Evaluating model...")
    mu, std = gp.predict(Xte_s, return_std=True)
    
    # --- Metrics ---
    mae = np.mean(np.abs(mu - yte))
    rmse = np.sqrt(np.mean((mu - yte)**2))
    r2 = r2_score(yte, mu)
    residuals = mu - yte
    res_mean, res_std = np.mean(residuals), np.std(residuals)
    
    print(f"Test MAE={mae:.3f} GeV, RMSE={rmse:.3f} GeV, R²={r2:.4f}")
    print(f"Residual mean={res_mean:.4f} GeV, std={res_std:.4f} GeV")
    print(f"Final kernel: {gp.kernel_}")
    
    # --------------------------------------------------------
    # Plot 1: Predicted vs Observed
    # --------------------------------------------------------
    plt.figure(figsize=(6,6))
    plt.scatter(yte, mu, s=10, alpha=0.5, color="tab:blue", label="Predictions")
    plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()],
             'r--', lw=1.5, label="Ideal (y=x)")
    
    plt.fill_between(
        [yte.min(), yte.max()],
        [yte.min() - np.mean(std), yte.max() - np.mean(std)],
        [yte.min() + np.mean(std), yte.max() + np.mean(std)],
        color="gray", alpha=0.2, label="±1σ mean"
    )
    plt.xlabel("Observed coherent edge (GeV)")
    plt.ylabel("Predicted coherent edge (GeV)")
    plt.title(f"GP surrogate\nMAE={mae:.3f} GeV  RMSE={rmse:.3f} GeV  R²={r2:.3f}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gp_pred_vs_obs.png", dpi=200)
    print("Saved plot to gp_pred_vs_obs.png")
    
    # --------------------------------------------------------
    # Plot 2: Residuals vs Observed
    # --------------------------------------------------------
    plt.figure(figsize=(6,4))
    plt.scatter(yte, residuals, s=10, alpha=0.5, color="tab:orange")
    plt.axhline(0, color="k", lw=1)
    plt.axhline(res_mean, color="red", ls="--", lw=1, label=f"Mean={res_mean:.3f}")
    plt.axhline(res_mean + res_std, color="gray", ls=":", lw=1)
    plt.axhline(res_mean - res_std, color="gray", ls=":", lw=1)
    plt.xlabel("Observed coherent edge (GeV)")
    plt.ylabel("Residual (Pred − Obs) [GeV]")
    plt.title("Residuals vs Observed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gp_residuals.png", dpi=200)
    print("Saved plot to gp_residuals.png")

    # Optional: show interactively
    plt.show()

    # --------------------------------------------------------
    # Save model + scaler
    # --------------------------------------------------------
    joblib.dump(gp, args.model)
    joblib.dump(scaler, args.scaler)
    print(f"Saved model to {args.model} and scaler to {args.scaler}")
    

if __name__ == "__main__":
    main()
