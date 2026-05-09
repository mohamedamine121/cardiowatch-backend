# ============================================================
# app/routes/session.py - PIPELINE HRV COMPLET + IA
# Modifié pour accepter 1500+ échantillons (60s @ 25Hz)
# Validation scientifique : PMC6953345, PMC4309304
# ✅ INTÉGRATION XGBoost pour détection FA
# ============================================================
 
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import logging
import joblib
from pathlib import Path

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
router = APIRouter()

# ============================================================
# CHARGEMENT MODÈLES IA
# ============================================================

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_cardiowatch.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

try:
    model_xgboost = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("=" * 60)
    logger.info("✅ MODÈLES IA CHARGÉS AVEC SUCCÈS")
    logger.info("=" * 60)
    logger.info(f"📁 XGBoost : {MODEL_PATH}")
    logger.info(f"📁 Scaler  : {SCALER_PATH}")
    logger.info("=" * 60)
except FileNotFoundError as e:
    logger.error("=" * 60)
    logger.error("❌ FICHIERS MODÈLES INTROUVABLES")
    logger.error("=" * 60)
    logger.error(f"Erreur: {e}")
    logger.error(f"Chemin recherché : {MODEL_PATH}")
    logger.error("Vérifier que /models/ contient xgboost_cardiowatch.pkl et scaler.pkl")
    logger.error("⚠️ MODE DÉGRADÉ : Calcul HRV uniquement (pas de prédiction FA)")
    logger.error("=" * 60)
    model_xgboost = None
    scaler = None
except Exception as e:
    logger.error("=" * 60)
    logger.error("❌ ERREUR CHARGEMENT MODÈLES IA")
    logger.error("=" * 60)
    logger.error(f"Erreur: {e}")
    logger.error("⚠️ MODE DÉGRADÉ : Calcul HRV uniquement (pas de prédiction FA)")
    logger.error("=" * 60)
    model_xgboost = None
    scaler = None

# ============================================================
# MODÈLES DE DONNÉES
# ============================================================
 
class SessionData(BaseModel):
    patient_id: str
    ppg_values: List[float]  # Signal PPG IR (1500+ échantillons)
    timestamps_us: List[int] = []  # Timestamps microsecondes (optionnel)
    spo2: int
    timestamp: str = ""
 
class HRVResponse(BaseModel):
    status: str
    spo2: int
    mean_bpm: float
    sdnn: float
    rmssd: float
    pnn50: float
    entropy: float
    fs_real: float
    n_samples: int
    # ✅ NOUVEAUX CHAMPS IA
    af_detected: int                # 0=Normal, 1=FA, -1=Erreur
    af_probability: Optional[float] # 0.0-1.0 ou null
    af_risk: str                    # "Faible"/"Élevé"/"Erreur IA"
    af_confidence: Optional[float]  # 0-100% ou null

# ============================================================
# FONCTION PRÉDICTION FA
# ============================================================

def predict_af(features: dict) -> dict:
    """
    Prédiction Fibrillation Auriculaire avec XGBoost
    
    Args:
        features: dict avec Mean_BPM, SDNN, RMSSD, pNN50, Entropy
    
    Returns:
        dict avec:
        - label: 0 (Normal) | 1 (FA) | -1 (Erreur)
        - probability: 0.0-1.0 (proba FA) ou None
        - risk: "Faible" | "Élevé" | "Erreur IA"
        - confidence: 0-100% ou None
    """
    
    # Vérifier que modèles sont chargés
    if model_xgboost is None or scaler is None:
        logger.warning("⚠️ Modèles IA non disponibles - Prédiction impossible")
        logger.warning("   Mode dégradé : Features HRV calculées sans prédiction FA")
        return {
            'label': -1,
            'probability': None,
            'risk': 'Erreur IA',
            'confidence': None
        }
    
    try:
        logger.info("=" * 60)
        logger.info("🔬 PRÉDICTION IA - DÉTECTION FA")
        logger.info("=" * 60)
        
        # ── Préparer features dans BON ORDRE (IDENTIQUE training) ──
        # ORDRE CRITIQUE : Mean_BPM, SDNN, RMSSD, pNN50, Entropy
        X = np.array([[
            features['Mean_BPM'],
            features['SDNN'],
            features['RMSSD'],
            features['pNN50'],
            features['Entropy']
        ]])
        
        logger.info("📊 Features d'entrée :")
        logger.info(f"   - Mean_BPM : {features['Mean_BPM']:.1f} BPM")
        logger.info(f"   - SDNN     : {features['SDNN']:.1f} ms")
        logger.info(f"   - RMSSD    : {features['RMSSD']:.1f} ms")
        logger.info(f"   - pNN50    : {features['pNN50']:.1f} %")
        logger.info(f"   - Entropy  : {features['Entropy']:.4f}")
        
        # ── Normaliser avec scaler (IDENTIQUE training) ──
        X_scaled = scaler.transform(X)
        logger.info("✅ Normalisation appliquée (StandardScaler)")
        
        # ── Prédire avec XGBoost ──
        label = int(model_xgboost.predict(X_scaled)[0])
        proba_array = model_xgboost.predict_proba(X_scaled)[0]
        proba_fa = float(proba_array[1])  # Probabilité classe 1 (FA)
        
        logger.info("🎯 Prédiction XGBoost :")
        logger.info(f"   - Proba Normal : {proba_array[0]:.4f}")
        logger.info(f"   - Proba FA     : {proba_array[1]:.4f}")
        
        # ── Calculer risque et confiance ──
        if label == 1:
            # FA détectée
            risk = 'Élevé'
            confidence = proba_fa * 100  # Confiance = probabilité FA
            logger.info(f"🚨 RÉSULTAT : FIBRILLATION AURICULAIRE DÉTECTÉE")
        else:
            # Normal
            risk = 'Faible'
            confidence = (1 - proba_fa) * 100  # Confiance = probabilité Normal
            logger.info(f"✅ RÉSULTAT : RYTHME NORMAL")
        
        logger.info(f"   - Label    : {label} ({'FA' if label==1 else 'Normal'})")
        logger.info(f"   - Risque   : {risk}")
        logger.info(f"   - Confiance: {confidence:.1f}%")
        logger.info("=" * 60)
        
        return {
            'label': label,
            'probability': round(proba_fa, 4),  # 0.0-1.0
            'risk': risk,
            'confidence': round(confidence, 1)
        }
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ PRÉDICTION IA ÉCHOUÉE")
        logger.error("=" * 60)
        logger.error(f"Erreur : {str(e)}")
        logger.error(f"Features reçues : {features}")
        logger.error("=" * 60)
        return {
            'label': -1,
            'probability': None,
            'risk': 'Erreur IA',
            'confidence': None
        }

# ============================================================
# ÉTAPE 1 : VALIDATION SIGNAL
# ============================================================
 
def validate_signal(signal: np.ndarray) -> None:
    """
    Valide le signal PPG avant traitement
    """
    # ✅ Longueur minimale : 1500 échantillons (60s à 25Hz)
    # Validation scientifique :
    # - PMC6953345 : 60s validé pour HRV court terme et détection FA
    # - PMC4309304 : PPG 25Hz corrélation acceptable avec ECG (r=0.7-0.78)
    # - pNN50 nécessite minimum 60s (standard HRV)
    if len(signal) < 1500:
        raise HTTPException(
            status_code=400,
            detail=f"Signal trop court : {len(signal)} échantillons (min 1500 pour 60s @ 25Hz)"
        )
    
    # Pas de NaN
    if np.any(np.isnan(signal)):
        raise HTTPException(
            status_code=400,
            detail="Signal contient des valeurs NaN"
        )
    
    # Range physiologique
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_range = signal_max - signal_min
    
    if signal_range < 100:
        raise HTTPException(
            status_code=422,
            detail=f"Signal plat (range={signal_range:.0f}) - Pas de pulsation détectée"
        )
    
    logger.info(f"✅ Validation signal OK")
    logger.info(f"   - Échantillons : {len(signal)}")
    logger.info(f"   - Range        : {signal_range:.0f}")
    logger.info(f"   - Min          : {signal_min:.0f}")
    logger.info(f"   - Max          : {signal_max:.0f}")
 
# ============================================================
# ÉTAPE 2 : CALCUL FRÉQUENCE RÉELLE
# ============================================================
 
def calculate_real_fs(signal: np.ndarray, timestamps_us: List[int] = None) -> float:
    """
    Calcule la fréquence d'échantillonnage réelle
    Support : 25Hz (avg=4) ou 100Hz (avg=1)
    """
    if timestamps_us and len(timestamps_us) >= 2:
        # Méthode 1 : Depuis timestamps microsecondes
        ts_array = np.array(timestamps_us) / 1e6  # Convertir en secondes
        periods = np.diff(ts_array)
        median_period = np.median(periods)
        fs_real = 1.0 / median_period
        
        logger.info(f"📊 FS calculée depuis timestamps : {fs_real:.2f} Hz")
    else:
        # Méthode 2 : Estimation depuis longueur signal (60s attendu)
        duration_s = 60.0
        fs_real = len(signal) / duration_s
        
        logger.info(f"📊 FS estimée depuis longueur : {fs_real:.2f} Hz (assumant 60s)")
    
    # Validation range élargie pour supporter 25Hz et 100Hz
    if not (20 <= fs_real <= 150):
        raise HTTPException(
            status_code=400,
            detail=f"Fréquence anormale : {fs_real:.2f} Hz (attendu 20-150 Hz)"
        )
    
    return fs_real
 
# ============================================================
# ÉTAPE 3 : RE-ÉCHANTILLONNAGE À 125 Hz
# ============================================================
 
def resample_to_125hz(signal: np.ndarray, fs_original: float) -> np.ndarray:
    """
    Re-échantillonne le signal à 125 Hz (compatibilité MIMIC)
    Adaptatif : fonctionne avec 25Hz (1500 pts) ou 100Hz (6000 pts)
    
    Exemples :
    - 1500 pts @ 25Hz  → 7500 pts @ 125Hz (×5)
    - 6000 pts @ 100Hz → 7500 pts @ 125Hz (×1.25)
    """
    from scipy.signal import resample
    
    n_original = len(signal)
    n_target = int(n_original * (125.0 / fs_original))
    
    signal_125hz = resample(signal, n_target)
    
    logger.info(f"📊 Re-sampling : {n_original} pts @ {fs_original:.1f}Hz → {n_target} pts @ 125Hz")
    
    # Validation longueur cible (~7500 pour 60s)
    expected = 125 * 60
    if abs(len(signal_125hz) - expected) > 200:
        logger.warning(f"⚠️ Longueur après resample : {len(signal_125hz)} (attendu ~{expected})")
    
    return signal_125hz
 
# ============================================================
# ÉTAPE 4 : FILTRAGE BUTTERWORTH
# ============================================================
 
def filter_butterworth(signal: np.ndarray, fs: float = 125) -> np.ndarray:
    """
    Filtre passe-bande Butterworth 0.5-8 Hz, ordre 3
    IDENTIQUE au code training MIMIC
    """
    from scipy.signal import butter, filtfilt
    
    nyq = fs / 2.0
    low = 0.5 / nyq
    high = 8.0 / nyq
    
    # Vérifier limites Nyquist
    if high >= 1.0:
        high = 0.99
    
    b, a = butter(3, [low, high], btype='bandpass')
    signal_filtered = filtfilt(b, a, signal)
    
    # Validation : pas de NaN, pas d'explosion
    if np.any(np.isnan(signal_filtered)) or np.any(np.abs(signal_filtered) > 1e6):
        raise HTTPException(
            status_code=500,
            detail="Filtrage Butterworth instable"
        )
    
    logger.info(f"✅ Filtrage Butterworth 0.5-8 Hz appliqué")
    
    return signal_filtered
 
# ============================================================
# ÉTAPE 5 : DÉTECTION PICS HEARTPY
# ============================================================
 
def detect_peaks_heartpy(signal: np.ndarray, fs: float = 125):
    """
    Détection pics avec HeartPy - ADAPTÉ SIGNAL PPG
    Contraintes relâchées pour signal bruité ESP32
    """
    import heartpy as hp
    
    try:
        # ✅ Paramètres adaptés PPG bruité
        working_data, measures = hp.process(
            signal,
            sample_rate=fs,
            bpmmin=30,           # ✅ Élargi : 30 au lieu de 40
            bpmmax=180,          # ✅ Élargi : 180 au lieu de 150
            high_precision=False, # ✅ Désactivé pour signal bruité
            clean_rr=True,
            clean_rr_method='iqr',
            reject_segmentwise=False  # ✅ Ne pas rejeter segments
        )
        
        bpm_detected = float(measures['bpm'])
        n_peaks = len(working_data['peaklist'])
        
        logger.info(f"✅ HeartPy - Détection pics réussie")
        logger.info(f"   - Pics détectés : {n_peaks}")
        logger.info(f"   - BPM calculé   : {bpm_detected:.1f}")
        
    except Exception as e:
        # ✅ Log détaillé pour debug
        logger.error("=" * 60)
        logger.error("❌ HEARTPY - DÉTECTION PICS ÉCHOUÉE")
        logger.error("=" * 60)
        logger.error(f"Erreur : {str(e)}")
        logger.error(f"Signal stats :")
        logger.error(f"   - Min  : {np.min(signal):.1f}")
        logger.error(f"   - Max  : {np.max(signal):.1f}")
        logger.error(f"   - Mean : {np.mean(signal):.1f}")
        logger.error(f"   - Std  : {np.std(signal):.1f}")
        logger.error("=" * 60)
        
        raise HTTPException(
            status_code=422,
            detail=f"HeartPy détection échouée : {str(e)}"
        )
    
    # Validation BPM ÉLARGIE
    if not (30 <= bpm_detected <= 180):  # ✅ Range élargi
        raise HTTPException(
            status_code=422,
            detail=f"BPM hors range : {bpm_detected:.1f} (attendu 30-180)"
        )
    
    return working_data, measures
 
# ============================================================
# ÉTAPE 6 : CALCUL IBI (Inter-Beat Intervals)
# ============================================================
 
def calculate_ibi(working_data, fs: float = 125) -> np.ndarray:
    """
    Calcule les intervalles RR en millisecondes
    """
    rr_list = working_data['RR_list']
    
    if len(rr_list) < 30:
        raise HTTPException(
            status_code=422,
            detail=f"Trop peu de battements : {len(rr_list)} (min 30)"
        )
    
    # Convertir en ms (RR_list déjà en ms normalement)
    ibi_ms = np.array(rr_list, dtype=np.float64)
    
    logger.info(f"📊 IBI calculés : {len(ibi_ms)} intervalles")
    logger.info(f"   - Mean IBI : {np.mean(ibi_ms):.1f} ms")
    logger.info(f"   - Std IBI  : {np.std(ibi_ms):.1f} ms")
    
    return ibi_ms
 
# ============================================================
# ÉTAPE 7 : FILTRAGE OUTLIERS
# ============================================================
 
def filter_outliers(ibi_ms: np.ndarray) -> np.ndarray:
    """
    Filtrage adapté signal PPG ESP32 :
    1. Physiologique : 400-1500 ms
    2. Malik TRÈS RELÂCHÉ : ±50% médiane (vs 20% ECG standard)
    
    Justification scientifique :
    - PPG variabilité naturelle > ECG (PMC4309304: r=0.7-0.8)
    - Filtre trop strict rejette battements valides
    - 50% = compromis entre robustesse et précision
    
    MODIFIÉ pour production ESP32
    """
    from hrvanalysis import remove_outliers, remove_ectopic_beats
    
    n_initial = len(ibi_ms)
    
    # Filtre physiologique
    ibi_clean = remove_outliers(
        rr_intervals=ibi_ms.tolist(),
        low_rri=400,   # Production : 400 ms (150 BPM)
        high_rri=1500  # Production : 1500 ms (40 BPM)
    )
    
    n_after_physio = len(ibi_clean)
    logger.info(f"📊 Filtre physiologique : {n_initial} → {n_after_physio} IBI")
    
    # Filtre Malik TRÈS RELÂCHÉ pour PPG
    try:
        ibi_clean = remove_ectopic_beats(
            rr_intervals=ibi_clean,
            method="malik",
            custom_removing_rule=0.50  # ✅ 50% tolérance PPG
        )
        n_after_malik = len(ibi_clean)
        logger.info(f"📊 Filtre Malik 50%     : {n_after_physio} → {n_after_malik} IBI")
    except Exception as e:
        # Si Malik échoue, garder filtre physio seulement
        logger.warning(f"⚠️ Malik échoué, filtre physio seul utilisé : {e}")
    
    # Convertir en array numpy
    ibi_clean = np.array(ibi_clean, dtype=np.float64)
    
    # Vérifier pas de NaN
    if np.any(np.isnan(ibi_clean)):
        raise HTTPException(
            status_code=422,
            detail="IBI contient NaN après filtrage outliers"
        )
    
    # Minimum 20 IBI valides (au lieu de 30 pour signal PPG réel)
    if len(ibi_clean) < 20:
        raise HTTPException(
            status_code=422,
            detail=f"Trop peu d'IBI valides : {len(ibi_clean)} (min 20)"
        )
    
    logger.info(f"✅ Filtrage outliers terminé : {n_initial} → {len(ibi_clean)} IBI conservés")
    
    return ibi_clean
 
# ============================================================
# ÉTAPE 8 : CALCUL FEATURES HRV
# ============================================================
 
def calculate_hrv_features(ibi_clean: np.ndarray) -> dict:
    """
    Calcule features HRV TIME-DOMAIN
    IDENTIQUE au code training MIMIC
    """
    from hrvanalysis import get_time_domain_features
    from scipy.stats import entropy as scipy_entropy
    
    # Mean BPM depuis IBI
    mean_bpm = 60000.0 / np.mean(ibi_clean)
    
    # Features HRV
    features = get_time_domain_features(ibi_clean.tolist())
    
    # Entropie Shannon (IDENTIQUE training)
    hist, _ = np.histogram(ibi_clean, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_entropy = scipy_entropy(hist, base=2)
    
    result = {
        'Mean_BPM': round(mean_bpm, 1),
        'SDNN': round(features['sdnn'], 1),
        'RMSSD': round(features['rmssd'], 1),
        'pNN50': round(features['pnni_50'], 1),  # ⚠️ Attention : pnni_50
        'Entropy': round(shannon_entropy, 4)
    }
    
    logger.info("=" * 60)
    logger.info("📊 FEATURES HRV CALCULÉES")
    logger.info("=" * 60)
    logger.info(f"Mean_BPM : {result['Mean_BPM']:.1f} BPM")
    logger.info(f"SDNN     : {result['SDNN']:.1f} ms")
    logger.info(f"RMSSD    : {result['RMSSD']:.1f} ms")
    logger.info(f"pNN50    : {result['pNN50']:.1f} %")
    logger.info(f"Entropy  : {result['Entropy']:.4f}")
    logger.info("=" * 60)
    
    return result
 
# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================
 
@router.post("/analyze", response_model=HRVResponse)
async def analyze_session(data: SessionData):
    """
    Pipeline HRV complet avec détection FA
    Support : 1500+ échantillons (25Hz ou 100Hz)
    
    ✅ NOUVEAU : Prédiction FA avec XGBoost
    
    Modifications STEP 2 :
    - Accepte 1500+ échantillons (au lieu de 5400+)
    - Signal lissé matériellement (sampleAverage=4)
    - Validé scientifiquement pour 60s @ 25Hz
    """
    
    logger.info("=" * 60)
    logger.info("🚀 ANALYSE SESSION - DÉMARRAGE")
    logger.info("=" * 60)
    logger.info(f"Patient ID : {data.patient_id}")
    logger.info(f"Timestamp  : {data.timestamp}")
    logger.info(f"SpO2       : {data.spo2}%")
    logger.info("=" * 60)
    
    try:
        # ── Convertir en numpy ────────────────────────────
        signal = np.array(data.ppg_values, dtype=np.float64)
        
        # ── ÉTAPE 1 : Validation ──────────────────────────
        logger.info("ÉTAPE 1/9 : Validation signal")
        validate_signal(signal)
        
        # ── ÉTAPE 2 : Calcul FS réelle ────────────────────
        logger.info("ÉTAPE 2/9 : Calcul fréquence d'échantillonnage")
        fs_real = calculate_real_fs(signal, data.timestamps_us)
        
        # ── ÉTAPE 3 : Re-échantillonnage 125 Hz ───────────
        logger.info("ÉTAPE 3/9 : Re-échantillonnage à 125 Hz")
        signal_125hz = resample_to_125hz(signal, fs_real)
        
        # ── ÉTAPE 4 : Filtrage Butterworth ────────────────
        logger.info("ÉTAPE 4/9 : Filtrage Butterworth")
        signal_filtered = filter_butterworth(signal_125hz, fs=125)
        
        # ── ÉTAPE 4.5 : Normalisation pour HeartPy ────────
        logger.info("ÉTAPE 5/9 : Normalisation Z-score")
        signal_mean = np.mean(signal_filtered)
        signal_std = np.std(signal_filtered)
        
        if signal_std > 0:
            signal_normalized = (signal_filtered - signal_mean) / signal_std
            logger.info(f"✅ Signal normalisé : mean=0, std=1")
        else:
            signal_normalized = signal_filtered
            logger.warning(f"⚠️ Signal std=0, normalisation ignorée")
        
        # ── ÉTAPE 5 : Détection pics HeartPy ──────────────
        logger.info("ÉTAPE 6/9 : Détection pics HeartPy")
        working_data, measures = detect_peaks_heartpy(signal_normalized, fs=125)
        
        # ── ÉTAPE 6 : Calcul IBI ──────────────────────────
        logger.info("ÉTAPE 7/9 : Calcul IBI (Inter-Beat Intervals)")
        ibi_ms = calculate_ibi(working_data, fs=125)
        
        # ── ÉTAPE 7 : Filtrage outliers ───────────────────
        logger.info("ÉTAPE 8/9 : Filtrage outliers")
        ibi_clean = filter_outliers(ibi_ms)
        
        # ── ÉTAPE 8 : Features HRV ────────────────────────
        logger.info("ÉTAPE 9/9 : Calcul features HRV")
        features = calculate_hrv_features(ibi_clean)
        
        # ── ÉTAPE 9 : PRÉDICTION IA FA ────────────────────
        prediction = predict_af(features)
        
        # ── Retour résultat ───────────────────────────────
        logger.info("=" * 60)
        logger.info("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        
        return HRVResponse(
            status="success",
            spo2=data.spo2,
            mean_bpm=features['Mean_BPM'],
            sdnn=features['SDNN'],
            rmssd=features['RMSSD'],
            pnn50=features['pNN50'],
            entropy=features['Entropy'],
            fs_real=round(fs_real, 1),
            n_samples=len(signal),
            # ✅ RÉSULTATS IA
            af_detected=prediction['label'],
            af_probability=prediction['probability'],
            af_risk=prediction['risk'],
            af_confidence=prediction['confidence']
        )
        
    except HTTPException:
        # Re-raise HTTPException directement
        raise
    
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ ERREUR INATTENDUE")
        logger.error("=" * 60)
        logger.error(f"Erreur : {str(e)}")
        logger.error("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pipeline HRV : {str(e)}"
        )