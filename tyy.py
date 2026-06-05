import joblib
import numpy as np

# Charger les 3 fichiers
xgb = joblib.load('xgboost_final_lopo.pkl')
rf  = joblib.load('random_forest_final_lopo.pkl')
sc  = joblib.load('scaler_final_lopo.pkl')

# Test rapide avec des features fictives
test = [[75, 120, 95, 60, 3.9]]  # [Mean_BPM, SDNN, RMSSD, pNN50, Entropy]
test_scaled = sc.transform(test)

p_xgb = xgb.predict_proba(test_scaled)[0, 1]
p_rf  = rf.predict_proba(test_scaled)[0, 1]
p_ens = (p_xgb + p_rf) / 2

print(f"XGBoost   : {p_xgb:.3f}")
print(f"RF        : {p_rf:.3f}")
print(f"Ensemble  : {p_ens:.3f}")
print("✅ Chargement OK")