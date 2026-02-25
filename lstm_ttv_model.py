"""
LSTM Ensemble Model for Transit Timing Variation (TTV) Analysis
===============================================================
Predicts exoplanet orbital parameters (mass, period, argument of periastron,
eccentricity) from TTV signals using an LSTM-based neural network with
MC Dropout for uncertainty quantification.

Sections:
  1. Imports & Constants
  2. Helper Functions (TTV simulation, noise, preprocessing)
  3. Data Loading & Preprocessing
  4. Model Architecture & Training
  5. Prediction with MC Dropout
  6. Single-System Validation (Simulated)
  7. Kepler Data Validation
  8. Multi-System Validation & Metrics
  9. Missing Data Analysis
"""

# =============================================================================
# 1. IMPORTS & CONSTANTS
# =============================================================================

import os
import time
import math
import random
import pickle
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import corner
import ttvfast
from ttvfast import models
from scipy.stats import beta
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization, concatenate
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Matplotlib style
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 14})

# Physical constants
rsun   = 6.955e8              # m
msun   = 1.989e30             # kg
mjup   = 1.898e27             # kg
rjup   = 7.1492e7             # m
mearth = 5.972e24             # kg
rearth = 6.3781e6             # m
au     = 1.496e11             # m
G      = 0.00029591220828559104  # day, AU, Msun


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def ttv_generate(ms, p1, m1, p2, m2, o2, e2):
    """Run a 2-planet TTVFast simulation and return the raw results dict."""
    stellar_mass = ms
    planet1 = models.Planet(
        mass=m1, period=p1, eccentricity=0,
        inclination=90, longnode=0, argument=0, mean_anomaly=0,
    )
    planet2 = models.Planet(
        mass=m2, period=p2, eccentricity=e2,
        inclination=90, longnode=0, argument=o2, mean_anomaly=0,
    )
    Time  = -1045   # days
    dt    = 1 / 24  # days
    Total = 1700    # days
    return ttvfast.ttvfast([planet1, planet2], stellar_mass, Time, dt, Total)


def TTV(epochs, tt):
    """Subtract the best-fit linear ephemeris; return (O-C, slope, intercept)."""
    N = len(epochs)
    A = np.vstack([np.ones(N), epochs]).T
    b, m = np.linalg.lstsq(A, tt, rcond=None)[0]
    ttv = tt - m * np.array(epochs) - b
    return ttv, m, b


def generate_random_resonance(min_val=2, max_val=10):
    """Return a simplified (p:q) resonance pair with p > q."""
    while True:
        num = random.randint(min_val, max_val)
        den = random.randint(min_val, max_val)
        if num != den:
            break
    g = math.gcd(num, den)
    num //= g
    den //= g
    if num < den:
        num, den = den, num
    return num, den


def generate_random_exclude_1(n, low=0.3, high=2.5, exclude_range=(0.9, 1.1)):
    """Sample n uniform values in [low, high] excluding the near-unity range."""
    out = []
    while len(out) < n:
        val = np.random.uniform(low, high)
        if not (exclude_range[0] <= val <= exclude_range[1]):
            out.append(val)
    return np.array(out)


def add_noise(data, noise_factor=0.2):
    """Add Gaussian noise proportional to the absolute signal amplitude."""
    noise = np.random.normal(0, noise_factor * np.abs(data), data.shape)
    return data + noise


def remove_points_and_interpolate(ttv, missing_fraction):
    """Randomly drop `missing_fraction` of TTV points and linearly interpolate."""
    n = len(ttv)
    t = np.arange(n)
    mask = np.random.rand(n) > missing_fraction
    if mask.sum() < 3:
        mask[np.random.choice(n, 3, replace=False)] = True
    f = interp1d(t[mask], ttv[mask], kind="linear", fill_value="extrapolate")
    return f(t)


def predict_with_uncertainty(model_fn, x, n_iter=100):
    """Run n_iter stochastic forward passes (MC Dropout) and return mean & std."""
    results = np.array([model_fn(x, training=True) for _ in range(n_iter)])
    return results.mean(axis=0), results.std(axis=0)


# =============================================================================
# 3. DATA LOADING & PREPROCESSING
# =============================================================================

# --- Paths (update before running) ---
filepath      = "/content/gdrive/MyDrive/S3/paper22/"
datatrain_name = "datatrain_paper2_10k_250_filtered"
filename      = filepath + datatrain_name + ".pkl"

log_dir = "/content/gdrive/MyDrive/S3/revisi_paper/revisi_paper/model_logs_LSTM/experiment_2025-11-22_21-28-54"


def load_and_preprocess(filename, limit=100000, noise_factor=0.2):
    """
    Load a TTV pickle dataset, apply amplitude filter, add noise,
    normalize all features, and split into train/test sets.

    Returns:
        X_train_inputs, X_test_inputs: lists of input arrays
        y_train_outputs, y_test_outputs: lists of output arrays
        scalers: dict of fitted MinMaxScaler objects
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)

    ttv_data             = np.array(data["ttv_data"])[0:limit]
    star_mass            = np.array(data["star_mass"])[0:limit]
    transiting_period    = np.array(data["transiting_period"])[0:limit]
    transiting_mass      = np.array(data["transiting_mass"])[0:limit]
    mass_out             = np.array(data["perturbing_mass"])[0:limit]
    period_out           = np.array(data["perturbing_period"])[0:limit]
    eccentricity_out     = np.zeros(len(data["perturbing_eccentricity"]))[0:limit]
    argument_perigee_out = np.array(data["perturbing_argument_perigee"])[0:limit]

    # Keep only systems where transiting planet is inner (ratio < 1)
    ratio = transiting_period / period_out
    mask  = np.where(ratio < 1)[0]
    ttv_data, star_mass      = ttv_data[mask], star_mass[mask]
    transiting_period        = transiting_period[mask]
    transiting_mass          = transiting_mass[mask]
    mass_out, period_out     = mass_out[mask], period_out[mask]
    eccentricity_out         = eccentricity_out[mask]
    argument_perigee_out     = argument_perigee_out[mask]

    # Add noise and reshape TTV to (N, timesteps, 1)
    ttv_data  = add_noise(ttv_data, noise_factor=noise_factor)
    n_timesteps = ttv_data.shape[1]
    ttv_data  = ttv_data.reshape((-1, n_timesteps, 1))

    # Fit scalers
    scalers = {}
    def fit_scale(arr, name, is_sequence=False):
        sc = MinMaxScaler()
        if is_sequence:
            transformed = sc.fit_transform(arr.reshape(-1, 1)).reshape(arr.shape)
        else:
            transformed = sc.fit_transform(arr.reshape(-1, 1)).reshape(arr.shape)
        scalers[name] = sc
        return transformed

    ttv_data          = fit_scale(ttv_data,          "ttv",          is_sequence=True)
    star_mass         = fit_scale(star_mass,         "star_mass")
    transiting_period = fit_scale(transiting_period, "transiting_period")
    transiting_mass   = fit_scale(transiting_mass,   "transiting_mass")
    mass_out          = fit_scale(mass_out,          "mass_output")
    period_out        = fit_scale(period_out,        "period_output")
    eccentricity_out  = fit_scale(eccentricity_out,  "eccentricity_output")
    argument_perigee_out = fit_scale(argument_perigee_out, "argument_perigee_output")

    # Train/test split (80/20)
    split_kw = dict(test_size=0.2, random_state=42)
    Xttv_tr,  Xttv_te  = train_test_split(ttv_data,          **split_kw)
    Xms_tr,   Xms_te   = train_test_split(star_mass,         **split_kw)
    Xp1_tr,   Xp1_te   = train_test_split(transiting_period, **split_kw)
    Xm1_tr,   Xm1_te   = train_test_split(transiting_mass,   **split_kw)
    ym_tr,    ym_te     = train_test_split(mass_out,          **split_kw)
    yp_tr,    yp_te     = train_test_split(period_out,        **split_kw)
    ye_tr,    ye_te     = train_test_split(eccentricity_out,  **split_kw)
    yo_tr,    yo_te     = train_test_split(argument_perigee_out, **split_kw)

    X_train = [Xttv_tr, Xms_tr, Xp1_tr, Xm1_tr]
    X_test  = [Xttv_te, Xms_te, Xp1_te, Xm1_te]
    y_train = [ym_tr, yp_tr, ye_tr, yo_tr]
    y_test  = [ym_te, yp_te, ye_te, yo_te]

    print("X_train shapes:", [x.shape for x in X_train])
    print("y_train shapes:", [y.shape for y in y_train])
    return X_train, X_test, y_train, y_test, scalers, n_timesteps


# =============================================================================
# 4. MODEL ARCHITECTURE & TRAINING
# =============================================================================

def build_model(n_timesteps):
    """Build and compile the LSTM ensemble model."""
    ttv_input              = Input(shape=(n_timesteps, 1), name="ttv_input")
    star_mass_input        = Input(shape=(1,),             name="star_mass_input")
    transiting_period_input = Input(shape=(1,),            name="transiting_period_input")
    transiting_mass_input  = Input(shape=(1,),             name="transiting_mass_input")

    # LSTM branch
    x = LSTM(128, activation="tanh", return_sequences=True)(ttv_input)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = LSTM(128, return_sequences=False)(x)

    # Scalar input branches
    ms_dense = Dense(8, activation="relu")(star_mass_input)
    p1_dense = Dense(8, activation="relu")(transiting_period_input)
    m1_dense = Dense(8, activation="relu")(transiting_mass_input)

    merged = concatenate([x, ms_dense, p1_dense, m1_dense])

    # Shared fully-connected head
    h = Dense(128, activation="relu")(merged)
    h = Dropout(0.1)(h)
    h = Dense(64,  activation="relu")(h)
    h = Dropout(0.2)(h)
    h = Dense(32,  activation="relu")(h)
    h = Dropout(0.1)(h)
    h = Dense(16,  activation="relu")(h)
    h = Dense(8,   activation="relu")(h)

    # Output heads
    mass_out     = Dense(1, activation="linear",  name="mass_output")(h)
    period_out   = Dense(1, activation="linear",  name="period_output")(h)
    ecc_out      = Dense(1, activation="sigmoid", name="eccentricity_output")(h)
    omega_out    = Dense(1, activation="linear",  name="argument_perigee_output")(h)

    model = Model(
        inputs=[ttv_input, star_mass_input, transiting_period_input, transiting_mass_input],
        outputs=[mass_out, period_out, ecc_out, omega_out],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "mass_output":             "mse",
            "period_output":           "mse",
            "eccentricity_output":     "mae",
            "argument_perigee_output": "mse",
        },
        metrics={
            "mass_output":             "mse",
            "period_output":           "mse",
            "eccentricity_output":     "mae",
            "argument_perigee_output": "mse",
        },
        loss_weights={
            "mass_output": 1.0, "period_output": 1.0,
            "eccentricity_output": 1.0, "argument_perigee_output": 1.0,
        },
    )
    return model


def save_model_summary(model, log_dir, datatrain_name, model_name):
    """Write model.summary() to a text file."""
    path = os.path.join(log_dir, f"model_summary_{datatrain_name}{model_name}.txt")
    with open(path, "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))


def create_log_directory(base_dir="model_logs_LSTM"):
    """Create a timestamped experiment directory and return its path."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base_dir, f"experiment_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def train_model(model, X_train, y_train, log_dir, datatrain_name, model_name,
                epochs=100, batch_size=64):
    """Train the model with early stopping and learning-rate scheduling."""
    early_stop = EarlyStopping(monitor="val_loss", patience=20,
                               restore_best_weights=True, verbose=1)
    reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=5, min_lr=1e-6, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(log_dir, f"model_tanh_{datatrain_name}{model_name}.weights.h5"),
        monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1,
    )

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs, batch_size=batch_size, validation_split=0.3,
        callbacks=[early_stop, reduce_lr, checkpoint], verbose=1,
    )
    print(f"Training time: {(time.time()-start)/60:.2f} min")

    model.save(os.path.join(log_dir, f"exoplanet_ens_model_tanh_{datatrain_name}{model_name}.keras"))
    with open(os.path.join(log_dir, f"history_tanh_{datatrain_name}{model_name}.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    return history


def plot_training_history(history, model_path, log_dir):
    """Plot and save the training vs. validation loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"],     label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("LSTM Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(model_path[:-6] + "_history_loss.pdf", dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# 5. PREDICTION WITH MC DROPOUT
# =============================================================================

def mc_dropout_predict(saved_model, new_X_inputs, scalers, n_iter=100):
    """
    Run MC Dropout inference and return de-normalized predictions + uncertainties.

    Parameters
    ----------
    saved_model : tf.keras.Model  (loaded with compile=False)
    new_X_inputs : list of normalized input arrays
    scalers : dict returned by load_and_preprocess
    n_iter : number of stochastic forward passes

    Returns
    -------
    dict with keys: predicted_mass, predicted_period, predicted_eccentricity,
                    predicted_argument_perigee, and *_uncertainty counterparts,
                    plus raw sample arrays real_mass_samples, etc.
    """
    predict_fn = tf.function(lambda x, training: saved_model(x, training=training))

    raw_results = [predict_fn(new_X_inputs, training=True) for _ in range(n_iter)]

    raw_mass   = np.array([r[0].numpy() for r in raw_results]).reshape(n_iter, 1)
    raw_period = np.array([r[1].numpy() for r in raw_results]).reshape(n_iter, 1)
    raw_ecc    = np.array([r[2].numpy() for r in raw_results]).reshape(n_iter, 1)
    raw_omega  = np.array([r[3].numpy() for r in raw_results]).reshape(n_iter, 1)

    real_mass   = scalers["mass_output"].inverse_transform(raw_mass)
    real_period = scalers["period_output"].inverse_transform(raw_period)
    real_ecc    = scalers["eccentricity_output"].inverse_transform(raw_ecc)
    real_omega  = scalers["argument_perigee_output"].inverse_transform(raw_omega)

    return {
        "predicted_mass":              np.array([np.mean(real_mass)]),
        "mass_uncertainty":            np.array([np.std(real_mass)]),
        "predicted_period":            np.array([np.mean(real_period)]),
        "period_uncertainty":          np.array([np.std(real_period)]),
        "predicted_eccentricity":      np.array([np.mean(real_ecc)]),
        "eccentricity_uncertainty":    np.array([np.std(real_ecc)]),
        "predicted_argument_perigee":  np.array([np.mean(real_omega)]),
        "argument_perigee_uncertainty": np.array([np.std(real_omega)]),
        "real_mass_samples":   real_mass,
        "real_period_samples": real_period,
        "real_ecc_samples":    real_ecc,
        "real_omega_samples":  real_omega,
    }


def prepare_single_input(ttv_curve, star_mass_val, transiting_period_val,
                          transiting_mass_val, scalers):
    """Normalize a single system's inputs ready for the model."""
    n_timesteps = len(ttv_curve)
    ttv = scalers["ttv"].transform(ttv_curve.reshape(-1, 1)).reshape(1, n_timesteps, 1)
    ms  = scalers["star_mass"].transform(np.array([[star_mass_val]]))
    p1  = scalers["transiting_period"].transform(np.array([[transiting_period_val]]))
    m1  = scalers["transiting_mass"].transform(np.array([[transiting_mass_val]]))
    return [ttv, ms, p1, m1]


# =============================================================================
# 6. SINGLE-SYSTEM VALIDATION (SIMULATED TTV)
# =============================================================================

def simulate_random_system(max_ttv=150, min_ttv=3, n_data=50, noise_factor=0.2):
    """
    Draw random orbital parameters, run TTVFast, and return a noisy TTV curve
    along with the true parameter values.
    """
    while True:
        resonance = generate_random_resonance()
        ms  = np.random.uniform(0.5, 2)
        m1  = np.random.uniform(1 * mearth / msun, 100 * mearth / msun)
        p1  = np.random.uniform(5, 20)
        m2  = m1 * np.random.uniform(0.5, 2.0)
        p2  = p1 * generate_random_exclude_1(n=1, low=1, high=3.0)[0]
        o2  = np.random.uniform(0, 360)
        e2  = 0.001 * beta.rvs(a=1.5, b=2)

        ress = ttv_generate(ms=ms, p1=p1, m1=m1, p2=p2, m2=m2, o2=o2, e2=e2)
        transit, epoch = [], []
        for i in range(len(ress["positions"][0])):
            if ress["positions"][0][i] == 0 and ress["positions"][2][i] != -2:
                transit.append(ress["positions"][2][i])
                epoch.append(ress["positions"][1][i])

        T1 = TTV(epoch, transit)
        ttv_curve = T1[0][:n_data] * 24 * 60  # convert to minutes
        if (len(T1[0]) > n_data
                and abs(ttv_curve).max() <= max_ttv
                and ttv_curve.max() >= min_ttv
                and ttv_curve.min() <= -min_ttv):
            ttv_noisy = add_noise(ttv_curve, noise_factor=noise_factor)
            return {
                "ms": ms, "m1": m1, "p1": p1, "m2": m2,
                "p2": p2, "o2": o2, "e2": e2,
                "ttv_curve": ttv_noisy, "epoch": epoch[:n_data],
            }


def chi_square_ttv(T_obs, T_gen, T_obs_uncertainty):
    """Return (chi2, reduced_chi2) for observed vs. generated TTV."""
    residuals    = T_obs - T_gen
    chi2         = np.sum((residuals / T_obs_uncertainty) ** 2)
    reduced_chi2 = chi2 / (len(T_obs) - 1)
    return chi2, reduced_chi2


def generate_ttv_uncertainty_envelope(ms, p1, m1, predicted, uncertainties,
                                       n_unc=500, n_data=50, max_scale=1.5, Terr=None):
    """
    Sample from the prediction uncertainty distributions, simulate TTVs,
    and return an array of valid TTV curves (the uncertainty envelope).
    """
    predicted_mass, predicted_period = predicted["mass"], predicted["period"]
    predicted_omega, predicted_ecc   = predicted["omega"], predicted["ecc"]
    unc_m, unc_p, unc_o, unc_e       = (uncertainties["mass"], uncertainties["period"],
                                         uncertainties["omega"], uncertainties["ecc"])
    max_ttv_val = 1.5 * (max(Terr) if Terr is not None else 150)

    all_TTV, mpred, ppred, opred = [], [], [], []
    for _ in range(n_unc):
        p2_s = abs(np.random.uniform(predicted_period - unc_p, predicted_period + unc_p))
        m2_s = abs(np.random.uniform(predicted_mass - unc_m, predicted_mass + unc_m))
        o2_s = np.random.uniform(predicted_omega - unc_o, predicted_omega + unc_o)
        e2_s = abs(np.random.uniform(0, predicted_ecc + unc_e))

        ress = ttv_generate(ms=ms, p1=p1, m1=m1,
                            p2=p2_s, m2=m2_s * mearth / msun, o2=o2_s, e2=e2_s)
        transit2, epoch2 = [], []
        for i in range(len(ress["positions"][0])):
            if ress["positions"][0][i] == 0 and ress["positions"][2][i] != -2:
                transit2.append(ress["positions"][2][i])
                epoch2.append(ress["positions"][1][i])

        if len(transit2) <= n_data:
            continue
        T2  = TTV(epoch2, transit2)
        ttv = T2[0][:n_data] * 24 * 60
        if ttv.max() > max_ttv_val or ttv.min() < -max_ttv_val:
            continue
        all_TTV.append(ttv)
        mpred.append(m2_s)
        ppred.append(p2_s)
        opred.append(o2_s)

    return np.array(all_TTV), mpred, ppred, opred, epoch2[:n_data]


# =============================================================================
# 7. KEPLER DATA VALIDATION
# =============================================================================

def load_kepler_data(kepler_dir):
    """
    Load the Kepler TTV dataset and the known planet parameter table.

    Returns:
        df        : DataFrame of observed TTV data (all KOIs)
        df_params : DataFrame with known orbital parameters per system
    """
    from astropy.table import Table

    datafile  = os.path.join(kepler_dir, "kepler-419.fit")
    ttv_excel = os.path.join(kepler_dir, "TTV_kepler.xlsx")

    ttv_table = Table.read(datafile)
    df        = ttv_table.to_pandas()

    df_params = pd.read_excel(ttv_excel, header=0)
    return df, df_params


def validate_kepler_system(i, df, df_params, saved_model, scalers, n_iter=100):
    """
    Run the full validation pipeline for the i-th Kepler system:
      load observed data → predict → build envelope → find best chi2 → plot.
    """
    koi    = df_params["KOI"][i]
    kepler = df_params["Kepler Name"][i]
    ms     = df_params["ms"][i]
    m1     = df_params["m1"][i] * mearth / msun
    p1     = df_params["p1"][i]
    m2     = df_params["m2"][i] * mearth / msun
    p2     = df_params["p2"][i]
    o2     = df_params["o2"][i]
    e2     = df_params["e2"][i]

    df_      = df[df["KOI"] == koi]
    new_df   = df_[abs(df_["O-C"]) < 100].reset_index(drop=True)
    n_data   = len(new_df["N"])

    # Prepare inputs
    ttv_obs = np.array(new_df["O-C"])
    inputs  = prepare_single_input(ttv_obs, ms, p1, m1 * msun / mearth, scalers)

    # MC Dropout prediction
    preds = mc_dropout_predict(saved_model, inputs, scalers, n_iter=n_iter)

    # Simulate best-fit TTV
    ress_pred = ttv_generate(
        ms=ms, p1=p1, m1=m1,
        p2=preds["predicted_period"],
        m2=preds["predicted_mass"] * mearth / msun,
        o2=preds["predicted_argument_perigee"],
        e2=preds["predicted_eccentricity"],
    )
    transit2, epoch2 = [], []
    for idx in range(len(ress_pred["positions"][0])):
        if ress_pred["positions"][0][idx] == 0 and ress_pred["positions"][2][idx] != -2:
            transit2.append(ress_pred["positions"][2][idx])
            epoch2.append(ress_pred["positions"][1][idx])
    T2 = TTV(epoch2, transit2)

    T_obs = np.array(new_df["O-C"][:n_data])
    T_unc = np.array(new_df["e_O-C"][:n_data])
    _, best_chi2 = chi_square_ttv(T_obs, T2[0][:n_data] * 24 * 60, T_unc)

    print(f"{kepler}: χ²_red = {best_chi2:.3f}")
    print(f"  m2 true={m2*msun/mearth:.2f} Mearth  pred={preds['predicted_mass'][0]:.2f}")
    print(f"  p2 true={p2:.2f} d         pred={preds['predicted_period'][0]:.2f}")
    return preds, T2, epoch2, new_df, n_data, best_chi2, kepler, koi


# =============================================================================
# 8. MULTI-SYSTEM VALIDATION & METRICS
# =============================================================================

def ensure_1d(x):
    """Flatten any array to 1-D."""
    return np.array(x).flatten()


def save_predictions_to_csv(m2_pred, p2_pred, o2_pred, e2_pred,
                             m2_true, p2_true, o2_true, e2_true,
                             m2_unc, p2_unc, o2_unc, e2_unc,
                             log_dir=".", filename="predictions.csv", append=False):
    """Save prediction results + uncertainties to a CSV file."""
    df = pd.DataFrame({
        "Mass_True":              ensure_1d(m2_true),
        "Mass_Pred":              ensure_1d(m2_pred),
        "Mass_Unc":               ensure_1d(m2_unc),
        "Period_True":            ensure_1d(p2_true),
        "Period_Pred":            ensure_1d(p2_pred),
        "Period_Unc":             ensure_1d(p2_unc),
        "ArgumentPerigee_True":   ensure_1d(o2_true),
        "ArgumentPerigee_Pred":   ensure_1d(o2_pred),
        "ArgumentPerigee_Unc":    ensure_1d(o2_unc),
        "Eccentricity_True":      ensure_1d(e2_true),
        "Eccentricity_Pred":      ensure_1d(e2_pred),
        "Eccentricity_Unc":       ensure_1d(e2_unc),
    })
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, filename)
    if append and os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    print(f"Saved predictions to: {csv_path}")
    return csv_path


def circular_diff(y_true, y_pred):
    """Shortest angular difference in degrees (handles wraparound)."""
    diff = y_pred - y_true
    return np.arctan2(np.sin(np.radians(diff)), np.cos(np.radians(diff))) * (180 / np.pi)


def willmott_d(y_true, y_pred):
    """Willmott index of agreement (0–1, higher is better)."""
    num = np.sum((y_pred - y_true) ** 2)
    den = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    return 1 - num / den


def regression_metrics(y_true, y_pred, label="", is_angle=False):
    """Compute a comprehensive set of regression metrics."""
    y_true = np.ravel(pd.to_numeric(pd.Series(y_true), errors="coerce"))
    y_pred = np.ravel(pd.to_numeric(pd.Series(y_pred), errors="coerce"))
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if is_angle:
        errors = circular_diff(y_true, y_pred)
        mae    = np.mean(np.abs(errors))
        rmse   = np.sqrt(np.mean(errors ** 2))
        mape   = np.nan
        bias   = np.mean(errors)
    else:
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        bias = np.mean(y_pred - y_true)

    pearson_r, _ = pearsonr(y_true, y_pred)
    return {
        "Parameter":   label,
        "R²":          r2_score(y_true, y_pred),
        "MAE":         mae,
        "RMSE":        rmse,
        "MAPE (%)":    mape,
        "Bias":        bias,
        "Pearson r":   pearson_r,
        "Willmott d":  willmott_d(y_true, y_pred),
    }


def compute_accuracy_within_sigma(predicted, true, uncertainty, n_sigma=1):
    """Return fraction of predictions where |residual| < n_sigma * uncertainty."""
    return np.mean(np.abs(predicted - true) < n_sigma * uncertainty)


def linreg(x, y):
    """Fit a linear regression and return (slope, intercept, y_pred)."""
    x = np.array(x).reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    return lr.coef_[0], lr.intercept_, lr.predict(x)


# =============================================================================
# 9. MISSING DATA ANALYSIS
# =============================================================================

def run_prediction_with_missing_for_one_system(
        Terr, ms, p1, m1,
        saved_model, predict_fn, scalers,
        missing_fractions):
    """
    Test prediction robustness by repeatedly dropping points from `Terr`
    at each fraction in `missing_fractions`, then running MC Dropout.

    Returns a DataFrame with results per missing fraction.
    """
    results = []
    for frac in missing_fractions:
        corrupted = remove_points_and_interpolate(Terr, frac)
        inputs    = prepare_single_input(corrupted, ms, p1, m1, scalers)

        n_iter      = 100
        raw_results = [predict_fn(inputs, training=True) for _ in range(n_iter)]

        raw_mass   = np.array([r[0].numpy() for r in raw_results]).reshape(n_iter, 1)
        raw_period = np.array([r[1].numpy() for r in raw_results]).reshape(n_iter, 1)
        raw_ecc    = np.array([r[2].numpy() for r in raw_results]).reshape(n_iter, 1)
        raw_omega  = np.array([r[3].numpy() for r in raw_results]).reshape(n_iter, 1)

        real_mass   = scalers["mass_output"].inverse_transform(raw_mass)
        real_period = scalers["period_output"].inverse_transform(raw_period)
        real_ecc    = scalers["eccentricity_output"].inverse_transform(raw_ecc)
        real_omega  = scalers["argument_perigee_output"].inverse_transform(raw_omega)

        results.append({
            "missing_fraction": frac,
            "Mass_Pred":   np.mean(real_mass),   "Mass_Unc":   np.std(real_mass),
            "Period_Pred": np.mean(real_period),  "Period_Unc": np.std(real_period),
            "Ecc_Pred":    np.mean(real_ecc),     "Ecc_Unc":    np.std(real_ecc),
            "Omega_Pred":  np.mean(real_omega),   "Omega_Unc":  np.std(real_omega),
        })
    return pd.DataFrame(results)


def run_missing_data_study(ttv_, ms_, p1_, m1_, m2_, p2_, e2_, o2_,
                            saved_model, scalers, Nsys=100,
                            missing_fractions=None):
    """
    Run the missing-data robustness study across Nsys systems.
    Returns a concatenated DataFrame with residuals.
    """
    if missing_fractions is None:
        missing_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    predict_fn = tf.function(lambda x, training: saved_model(x, training=training))
    all_results = []

    for i in range(min(Nsys, len(ttv_))):
        print(f"System {i+1}/{Nsys}")
        df_sys = run_prediction_with_missing_for_one_system(
            Terr=ttv_[i], ms=ms_[i], p1=p1_[i], m1=m1_[i],
            saved_model=saved_model, predict_fn=predict_fn,
            scalers=scalers, missing_fractions=missing_fractions,
        )
        df_sys["system_id"] = i
        df_sys["Mass_True"]   = m2_[i]
        df_sys["Period_True"] = p2_[i]
        df_sys["Ecc_True"]    = e2_[i]
        df_sys["Omega_True"]  = o2_[i]
        all_results.append(df_sys)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all["Mass_resid"]   = df_all["Mass_Pred"]   - df_all["Mass_True"]
    df_all["Period_resid"] = df_all["Period_Pred"]  - df_all["Period_True"]
    df_all["Ecc_resid"]    = df_all["Ecc_Pred"]    - df_all["Ecc_True"]
    df_all["Omega_resid"]  = df_all["Omega_Pred"]  - df_all["Omega_True"]
    return df_all


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def plot_parameter_distributions(mass_samples, period_samples,
                                  ecc_samples, omega_samples, save_path=None):
    """Plot MC Dropout prediction distributions for all 4 parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def _plot(ax, data, xlabel, color):
        sns.histplot(data, kde=True, ax=ax, color=color, bins=15, alpha=0.6)
        mean_, std_ = np.mean(data), np.std(data)
        ax.axvline(mean_, color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {mean_:.4f}")
        ax.axvline(mean_ - std_, color="black", linestyle=":", alpha=0.5,
                   label=fr"±1σ: {std_:.4f}")
        ax.axvline(mean_ + std_, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], mass_samples,         r"Mass ($M_{\oplus}$)",        "skyblue")
    _plot(axes[0, 1], period_samples,        "Period (days)",               "orange")
    _plot(axes[1, 0], ecc_samples * 1e7,     r"Eccentricity ($10^{-7}$)",   "green")
    _plot(axes[1, 1], omega_samples,         r"Arg. of Periastron ($°$)",   "purple")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_actual_vs_predicted(m2_, p2_, o2_,
                              predicted_mass, predicted_period,
                              predicted_argument_perigee,
                              log_dir, datatrain_name, model_name, file_path=""):
    """Hexbin plots of actual vs. predicted for mass, period, and argument of periastron."""
    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    for ax, x_true, y_pred, xlabel, ylabel in [
        (axs[0], m2_, predicted_mass.reshape(len(m2_)),
         r"Actual Mass $(M_⊕)$", r"Predicted Mass $(M_⊕)$"),
        (axs[1], p2_, predicted_period.reshape(len(m2_)),
         "Actual Period (days)", "Predicted Period (days)"),
        (axs[2], o2_, predicted_argument_perigee.reshape(len(m2_)),
         "Actual Arg. of Periastron (°)", "Predicted Arg. of Periastron (°)"),
    ]:
        slope, intercept, y_fit = linreg(x_true, y_pred)
        r2 = r2_score(x_true, y_pred)
        hb = ax.hexbin(x_true, y_pred, gridsize=40, cmap="Greens", mincnt=1, bins="log")
        ax.plot(x_true, y_fit, color="red",
                label=f"y = {slope:.2f}x + {intercept:.2f}")
        ax.plot([x_true.min(), x_true.max()],
                [x_true.min(), x_true.max()], "k--", label="Perfect Prediction")
        ax.text(0.05, 0.55, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=14,
                verticalalignment="top")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        fig.colorbar(hb, ax=ax)

    axs[1].set_title("LSTM Model")
    plt.savefig(
        os.path.join(log_dir, f"actual_vs_predicted_{datatrain_name}{model_name}.pdf"),
        dpi=300, bbox_inches="tight",
    )
    plt.show()


def plot_normalized_residuals(m2_, p2_, o2_,
                               predicted_mass, predicted_period,
                               predicted_argument_perigee,
                               unc_m, unc_p, unc_o,
                               log_dir):
    """Combined hexbin heatmap of normalized residuals for mass, period, and omega."""
    N  = len(m2_)
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
    fig.subplots_adjust(wspace=0.3)

    datasets = [
        (p2_.reshape(N), predicted_period.reshape(N), np.array(unc_p).reshape(N),
         r"Planet Period (day)", axes[0]),
        (o2_.reshape(N), predicted_argument_perigee.reshape(N), np.array(unc_o).reshape(N),
         r"Planet Arg. of Periastron $(°)$", axes[1]),
        (m2_.reshape(N), predicted_mass.reshape(N), np.array(unc_m).reshape(N),
         r"Planet Mass $(M_\oplus)$", axes[2]),
    ]

    for true, pred, unc, xlabel, ax in datasets:
        hb = ax.hexbin(true, (true - pred) / unc, gridsize=50, cmap="viridis", bins="log")
        for level, color, ls in [(0, "black", "dashed"), (1, "red", "dashed"),
                                  (-1, "red", "dashed"), (2, "orange", "dashed"),
                                  (-2, "orange", "dashed")]:
            ax.axhline(level, color=color, linestyle=ls)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Residual")
        ax.legend([r"±1σ", r"±2σ"], loc="upper left")
        ax.grid(True)
        fig.colorbar(hb, ax=ax, orientation="vertical", label="Log Density", pad=0.01)

    plt.savefig(os.path.join(log_dir, "combined_heatmap_residuals_LSTM.pdf"),
                dpi=600, bbox_inches="tight")
    plt.show()


def plot_missing_data_residuals(df_all, df_avg, df_std, log_dir):
    """Plot mean ± std residuals vs. missing fraction for period, omega, and mass."""
    params = ["Period_resid", "Omega_resid", "Mass_resid"]
    labels = ["Period", "Omega", "Mass"]
    colors = ["tab:blue", "tab:green", "tab:red"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    for ax, param, label, color in zip(axes, params, labels, colors):
        ax.errorbar(
            df_avg["missing_fraction"], df_avg[param], yerr=df_std[param],
            fmt="o-", color=color, elinewidth=1.5, capsize=4,
            label=f"{label} mean ± std",
        )
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Missing Fraction")
        ax.set_ylabel(f"{label} Residual")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "residual_plot_errorbar.pdf"),
                dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    model_name = "_LSTM_16_Nov"

    # ── 1. Load & preprocess training data ───────────────────────────────────
    X_train, X_test, y_train, y_test, scalers, n_timesteps = \
        load_and_preprocess(filename, limit=100000)

    # ── 2. Build model ────────────────────────────────────────────────────────
    log_dir = create_log_directory()
    model   = build_model(n_timesteps)
    model.summary()
    save_model_summary(model, log_dir, datatrain_name, model_name)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    history = train_model(model, X_train, y_train, log_dir, datatrain_name, model_name)
    model_path = os.path.join(
        log_dir, f"exoplanet_ens_model_tanh_{datatrain_name}{model_name}.keras"
    )
    with open(os.path.join(
        log_dir, f"history_tanh_{datatrain_name}{model_name}.pkl"), "rb") as f:
        hist = pickle.load(f)
    plot_training_history(hist, model_path, log_dir)

    # ── 4. Simulate a random system and predict ───────────────────────────────
    system  = simulate_random_system()
    inputs  = prepare_single_input(
        system["ttv_curve"], system["ms"], system["p1"],
        system["m1"] * msun / mearth, scalers
    )
    saved_model = load_model(model_path, compile=False)
    preds = mc_dropout_predict(saved_model, inputs, scalers)

    print("\n=== Single-system prediction ===")
    print(f"Mass:   {preds['predicted_mass'][0]:.3f} ± {preds['mass_uncertainty'][0]:.3f} Mearth")
    print(f"Period: {preds['predicted_period'][0]:.3f} ± {preds['period_uncertainty'][0]:.3f} d")
    print(f"Omega:  {preds['predicted_argument_perigee'][0]:.2f} ± "
          f"{preds['argument_perigee_uncertainty'][0]:.2f} deg")

    plot_parameter_distributions(
        preds["real_mass_samples"], preds["real_period_samples"],
        preds["real_ecc_samples"],  preds["real_omega_samples"],
        save_path=model_path[:-6] + "_uncertainty.pdf",
    )
