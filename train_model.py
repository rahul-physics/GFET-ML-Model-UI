import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
from utils import build_training_features

def train_model(df, L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac):
    # Build features
    X = build_training_features(df['Vgs (V)'], L_um, W_um, tox_nm, mobility_cm2Vs, n0, v_dirac)
    y = df['Resistance (Ohm)']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

   
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train_scaled, y_train)

    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Training Done! RMSE = {mse**0.5:.4f}")

    
    joblib.dump(model, 'model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    return model, scaler
