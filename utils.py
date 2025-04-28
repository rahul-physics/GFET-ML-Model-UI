import pandas as pd
import numpy as np

def build_training_features(Vgs_series, L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac):
    df = pd.DataFrame({
        "Vgs (V)": Vgs_series,
        "L_um": [L_um] * len(Vgs_series),
        "W_um": [W_um] * len(Vgs_series),
        "tox_nm": [tox_nm] * len(Vgs_series),
        "mobility_cm2Vs": [mobility_cm2Vs] * len(Vgs_series),
        "n0": [n0] * len(Vgs_series),
        "v_dirac": [v_dirac] * len(Vgs_series)
    })
    return df

def build_simulation_features(Vgs_array, L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac):
    Vgs_array = np.array(Vgs_array) 
    df = pd.DataFrame({
        "Vgs (V)": Vgs_array,
        "L_um": [L_um] * len(Vgs_array),
        "W_um": [W_um] * len(Vgs_array),
        "tox_nm": [tox_nm] * len(Vgs_array),
        "mobility_cm2Vs": [mobility_cm2Vs] * len(Vgs_array),
        "n0": [n0] * len(Vgs_array),
        "v_dirac": [v_dirac] * len(Vgs_array)
    })
    return df
