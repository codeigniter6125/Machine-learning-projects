"""
House Price Prediction 2026
===========================
Tools  : Python, pandas, NumPy, scikit-learn, matplotlib
Models : Linear Regression  |  Random Forest Regressor
Metrics: R²  |  RMSE  |  RMSLE
Target  R²   : 0.86 – 0.90
Target RMSE  : ~0.13 – 0.16 (log scale)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  HOUSE PRICE PREDICTION 2026")
print("=" * 60)

print("\n[1] Loading California Housing dataset …")
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
print(f"    Shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis …")
print(df.describe().round(3))

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Housing Dataset – Feature Distributions", fontsize=16, fontweight="bold")
for ax, col in zip(axes.flat, df.columns):
    ax.hist(df[col], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(col, fontsize=9)
    ax.set_ylabel("Count", fontsize=8)
plt.tight_layout()
plt.savefig("house_eda_distributions.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: house_eda_distributions.png")

# Correlation heatmap
import matplotlib.cm as cm

fig, ax = plt.subplots(figsize=(9, 7))
corr = df.corr()
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("house_correlation_heatmap.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: house_correlation_heatmap.png")

# ─────────────────────────────────────────────
# 3. DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] Cleaning & Feature Engineering …")

# Check missing values
missing = df.isnull().sum()
print(f"    Missing values:\n{missing[missing > 0] if missing.any() else '    None found [OK]'}")

# Cap extreme outliers (IQR method) for key features
for col in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

# Feature engineering
df["RoomsPerPerson"]   = df["AveRooms"]   / df["AveOccup"].replace(0, np.nan)
df["BedrmsPerRoom"]    = df["AveBedrms"]  / df["AveRooms"].replace(0, np.nan)
df["PopDensity"]       = df["Population"] / df["AveOccup"].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

print(f"    Engineered features added. New shape: {df.shape}")

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT & SCALING
# ─────────────────────────────────────────────
print("\n[4] Train/Test Split & Scaling …")
TARGET = "MedHouseVal"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"    Train size : {X_train.shape[0]:,} samples")
print(f"    Test  size : {X_test.shape[0]:,} samples")

# ─────────────────────────────────────────────
# 5. TRAIN MODELS
# ─────────────────────────────────────────────
print("\n[5] Training Models …")

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)
print("    [OK]  Linear Regression trained.")

# --- Random Forest ---
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("    [OK]  Random Forest trained.")

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
def rmsle(y_true, y_pred):
    """Root Mean Squared Log Error (clips negatives to 0)."""
    y_pred_clipped = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred_clipped)))

print("\n[6] Evaluation Metrics")
print("-" * 50)
models = {
    "Linear Regression": (lr_pred, y_test),
    "Random Forest"    : (rf_pred, y_test),
}

results = {}
for name, (pred, true) in models.items():
    r2   = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    rmsle_val = rmsle(true, pred)
    results[name] = {"R²": r2, "RMSE": rmse, "RMSLE": rmsle_val}
    print(f"  {name}")
    print(f"    R²    : {r2:.4f}")
    print(f"    RMSE  : {rmse:.4f}")
    print(f"    RMSLE : {rmsle_val:.4f}")
    print()

# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
print("[7] Plotting Results …")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("House Price Prediction 2026 — Results", fontsize=18, fontweight="bold", y=1.01)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

COLORS = {"Linear Regression": "#E07B54", "Random Forest": "#4C72B0"}

# ── 7a. Actual vs Predicted (LR) ──────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, lr_pred, alpha=0.3, s=12, color=COLORS["Linear Regression"])
lims = [min(y_test.min(), lr_pred.min()), max(y_test.max(), lr_pred.max())]
ax1.plot(lims, lims, "k--", lw=1.5, label="Perfect fit")
ax1.set_xlabel("Actual Price"); ax1.set_ylabel("Predicted Price")
ax1.set_title(f"Linear Regression\nActual vs Predicted  (R²={results['Linear Regression']['R²']:.3f})")
ax1.legend(fontsize=8)

# ── 7b. Actual vs Predicted (RF) ──────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, rf_pred, alpha=0.3, s=12, color=COLORS["Random Forest"])
ax2.plot(lims, lims, "k--", lw=1.5, label="Perfect fit")
ax2.set_xlabel("Actual Price"); ax2.set_ylabel("Predicted Price")
ax2.set_title(f"Random Forest\nActual vs Predicted  (R²={results['Random Forest']['R²']:.3f})")
ax2.legend(fontsize=8)

# ── 7c. Residuals (LR) ────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
lr_res = y_test - lr_pred
ax3.scatter(lr_pred, lr_res, alpha=0.3, s=12, color=COLORS["Linear Regression"])
ax3.axhline(0, color="k", lw=1.5, linestyle="--")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Residuals")
ax3.set_title("Linear Regression — Residuals")

# ── 7d. Residuals (RF) ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rf_res = y_test - rf_pred
ax4.scatter(rf_pred, rf_res, alpha=0.3, s=12, color=COLORS["Random Forest"])
ax4.axhline(0, color="k", lw=1.5, linestyle="--")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Residuals")
ax4.set_title("Random Forest — Residuals")

# ── 7e. Metric comparison bar chart ───────────
ax5 = fig.add_subplot(gs[2, 0])
metrics = ["R²", "RMSE", "RMSLE"]
x = np.arange(len(metrics))
w = 0.3
bars_lr = [results["Linear Regression"][m] for m in metrics]
bars_rf = [results["Random Forest"][m]     for m in metrics]
ax5.bar(x - w/2, bars_lr, w, label="Linear Regression", color=COLORS["Linear Regression"])
ax5.bar(x + w/2, bars_rf, w, label="Random Forest",     color=COLORS["Random Forest"])
ax5.set_xticks(x); ax5.set_xticklabels(metrics)
ax5.set_title("Model Metric Comparison"); ax5.legend(fontsize=8)
for i, (v1, v2) in enumerate(zip(bars_lr, bars_rf)):
    ax5.text(i - w/2, v1 + 0.005, f"{v1:.3f}", ha="center", va="bottom", fontsize=7)
    ax5.text(i + w/2, v2 + 0.005, f"{v2:.3f}", ha="center", va="bottom", fontsize=7)

# ── 7f. Feature Importance (RF) ───────────────
ax6 = fig.add_subplot(gs[2, 1])
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_bar = ["#4C72B0" if v > feat_imp.median() else "#A8BFD8" for v in feat_imp]
feat_imp.plot(kind="barh", ax=ax6, color=colors_bar)
ax6.set_title("Random Forest — Feature Importance")
ax6.set_xlabel("Importance")

plt.savefig("house_price_results.png", dpi=150, bbox_inches="tight")
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: house_price_results.png")

# ─────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
for name, m in results.items():
    print(f"  {name:<22}  R²={m['R²']:.4f}  RMSE={m['RMSE']:.4f}  RMSLE={m['RMSLE']:.4f}")
winner = max(results, key=lambda k: results[k]["R²"])
print(f"\n  Best Model: {winner}  [WINNER]")
print("=" * 60)
