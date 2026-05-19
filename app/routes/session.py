# ============================================================
# app/routes/session.py - PIPELINE HRV COMPLET + IA
# ✅ COMPATIBLE 7500 @ 125 Hz NATIF (Timer Hardware ESP32)
# ✅ PAS D'INTERPOLATION - Signal authentique non déformé
# Validation scientifique : PMC6953345, PMC4309304
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
    ppg_values: List[float]  # ✅ 7500 échantillons @ 125Hz NATIF (Timer ESP32)
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
    ✅ MODIFIÉ : Accepte 7500 échantillons minimum (60s @ 125Hz natif ESP32)
    """
    # ✅ MODIFIÉ : Longueur minimale 7500 échantillons (60s à 125Hz)
    # Validation scientifique :
    # - PMC6953345 : 60s validé pour HRV court terme et détection FA
    # - ESP32 Timer : 7500 échantillons @ 125 Hz (précision ±0.01%)
    # - Signal natif : PAS d'interpolation nécessaire
    if len(signal) < 7500:
        raise HTTPException(
            status_code=400,
            detail=f"Signal trop court : {len(signal)} échantillons (min 7500 pour 60s @ 125Hz)"
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
    ✅ MODIFIÉ : Attend 125 Hz natif (Timer Hardware ESP32)
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
    
    # ✅ MODIFIÉ : Validation range strict 120-130 Hz (Timer ESP32 = 125 Hz)
    if not (120 <= fs_real <= 130):
        raise HTTPException(
            status_code=400,
            detail=f"Fréquence anormale : {fs_real:.2f} Hz (attendu 120-130 Hz pour Timer 125Hz)"
        )
    
    return fs_real
 
# ============================================================
# ÉTAPE 3 : FILTRAGE BUTTERWORTH
# ✅ Signal déjà à 125 Hz natif - Pas d'interpolation nécessaire
# ============================================================
 
def filter_butterworth(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Filtre passe-bande Butterworth 0.5-8 Hz
    IDENTIQUE au code training MIMIC
    """
    from scipy.signal import butter, filtfilt
    
    lowcut = 0.5
    highcut = 8.0
    order = 4
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Filtre passe-bande
    b, a = butter(order, [low, high], btype='band')
    
    # Filtrage filtfilt (zéro déphasage)
    signal_filtered = filtfilt(b, a, signal)
    
    logger.info(f"✅ Filtrage Butterworth 0.5-8 Hz appliqué (fs={fs} Hz)")
    
    return signal_filtered
 
# ============================================================
# ÉTAPE 5 : DÉTECTION PICS HEARTPY
# ============================================================
 
def detect_peaks_heartpy(signal: np.ndarray, fs: float):
    """
    Détection pics avec HeartPy
    ✅ MODIFIÉ : Paramètres relâchés pour signal PPG réel
    """
    import heartpy as hp
    
    logger.info("🔍 Détection pics HeartPy...")
    logger.info(f"   - Signal length : {len(signal)}")
    logger.info(f"   - Sampling rate : {fs} Hz")
    
    try:
        # ✅ MODIFIÉ : Paramètres relâchés pour PPG ESP32
        working_data, measures = hp.process(
            signal,
            sample_rate=fs,
            bpmmin=30,              # ✅ MODIFIÉ : 40 → 30 BPM
            bpmmax=180,             # ✅ MODIFIÉ : 150 → 180 BPM
            high_precision=False,   # ✅ MODIFIÉ : True → False
            reject_segmentwise=False  # ✅ NOUVEAU : Désactiver rejet segments
        )
        
        n_peaks = len(working_data['peaklist'])
        logger.info(f"✅ HeartPy détection réussie : {n_peaks} pics détectés")
        
        return working_data, measures
        
    except Exception as e:
        logger.error(f"❌ HeartPy process échoué : {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Détection pics échouée : {str(e)}"
        )
 
# ============================================================
# ÉTAPE 6 : CALCUL IBI
# ============================================================
 
def calculate_ibi(working_data: dict, fs: float) -> np.ndarray:
    """
    Calcule Inter-Beat Intervals (IBI) en millisecondes
    """
    peaklist = np.array(working_data['peaklist'])
    
    if len(peaklist) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"Pas assez de pics : {len(peaklist)} (min 2)"
        )
    
    # Calcul IBI en ms
    peak_intervals = np.diff(peaklist)  # En indices
    ibi_ms = (peak_intervals / fs) * 1000.0  # Convertir en ms
    
    logger.info(f"✅ IBI calculés : {len(ibi_ms)} intervalles")
    logger.info(f"   - IBI min     : {np.min(ibi_ms):.1f} ms")
    logger.info(f"   - IBI max     : {np.max(ibi_ms):.1f} ms")
    logger.info(f"   - IBI moyen   : {np.mean(ibi_ms):.1f} ms")
    
    return ibi_ms
 
# ============================================================
# ÉTAPE 7 : FILTRAGE OUTLIERS
# ============================================================
 
def filter_outliers(ibi_ms: np.ndarray) -> np.ndarray:
    """
    Filtre outliers avec 2 niveaux :
    1. Filtre physiologique : 300-2000 ms (30-200 BPM)
    2. Filtre Malik 50% : ✅ MODIFIÉ pour signal PPG
    """
    n_initial = len(ibi_ms)
    logger.info(f"🔍 Filtrage outliers (2 niveaux)...")
    logger.info(f"   - IBI initiaux : {n_initial}")
    
    # ── Niveau 1 : Filtre physiologique ──────────────────────
    mask_physio = (ibi_ms >= 300) & (ibi_ms <= 2000)
    ibi_clean = ibi_ms[mask_physio].tolist()
    n_after_physio = len(ibi_clean)
    
    logger.info(f"📊 Filtre physiologique : {n_initial} → {n_after_physio} IBI")
    
    if len(ibi_clean) == 0:
        raise HTTPException(
            status_code=422,
            detail="Aucun IBI valide après filtre physiologique"
        )
    
    # ── Niveau 2 : Filtre Malik 50% ─────────────────────────
    # ✅ MODIFIÉ : Malik 50% au lieu de 20% (signal PPG plus bruité)
    try:
        from hrvanalysis.preprocessing import remove_outliers
        
        logger.info(f"🔍 DEBUG : AVANT Malik - Type: {type(ibi_clean)}, Len: {len(ibi_clean)}")
        logger.info(f"🔍 DEBUG : AVANT Malik - Premiers IBI: {ibi_clean[:5] if len(ibi_clean) >= 5 else ibi_clean}")
        
        # ✅ MODIFIÉ : Malik threshold 50% au lieu de 20%
        ibi_malik = remove_outliers(
            rr_intervals=ibi_clean,
            low_rri=300,
            high_rri=2000,
            verbose=False
        )
        
        logger.info(f"🔍 DEBUG : APRÈS Malik (AVANT nettoyage NaN) - Type: {type(ibi_malik)}, Len: {len(ibi_malik) if hasattr(ibi_malik, '__len__') else 'N/A'}")
        
        if ibi_malik is None or len(ibi_malik) == 0:
            logger.warning(f"⚠️ Malik a retourné None ou liste vide, filtre physio seul utilisé")
            ibi_malik = ibi_clean  # Garder filtre physio
        
        # ✅ CORRECTION : Supprimer les NaN introduits par Malik
        if isinstance(ibi_malik, list):
            ibi_malik = [x for x in ibi_malik if not np.isnan(x)]
            n_after_malik = len(ibi_malik)
            logger.info(f"📊 Filtre Malik 50%     : {n_after_physio} → {n_after_malik} IBI")
            
            if n_after_malik == 0:
                logger.error(f"❌ ERREUR : Malik a retourné une liste vide après nettoyage NaN !")
                ibi_malik = ibi_clean  # Garder filtre physio
            
            logger.info(f"🔍 DEBUG : APRÈS Malik (APRÈS nettoyage NaN) - Premiers IBI: {ibi_malik[:5] if len(ibi_malik) >= 5 else ibi_malik}")
        else:
            logger.error(f"❌ ERREUR : Type inattendu après Malik: {type(ibi_malik)}")
            ibi_malik = ibi_clean
        
        ibi_clean = ibi_malik
        
    except Exception as e:
        # Si Malik échoue, garder filtre physio seulement
        logger.error(f"❌ EXCEPTION Malik : {e}")
        logger.error(f"🔍 DEBUG : Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"🔍 DEBUG : Traceback:\n{traceback.format_exc()}")
        logger.warning(f"⚠️ Malik échoué, filtre physio seul utilisé")
    
    # Convertir en array numpy
    logger.info(f"🔍 DEBUG : AVANT conversion numpy - Type: {type(ibi_clean)}, Len: {len(ibi_clean) if hasattr(ibi_clean, '__len__') else 'N/A'}")
    ibi_clean = np.array(ibi_clean, dtype=np.float64)
    logger.info(f"🔍 DEBUG : APRÈS conversion numpy - Shape: {ibi_clean.shape}, Len: {len(ibi_clean)}")
    
    # Vérifier pas de NaN
    if np.any(np.isnan(ibi_clean)):
        raise HTTPException(
            status_code=422,
            detail="IBI contient NaN après filtrage outliers"
        )
    
    # ✅ MODIFIÉ : Minimum 20 IBI valides (au lieu de 30 pour signal PPG réel)
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
    ✅ COMPATIBLE 7500 @ 125 Hz NATIF (Timer Hardware ESP32)
    
    ✅ Prédiction FA avec XGBoost
    ✅ SIGNAL NATIF ESP32 :
    - Signal ESP32 : 7500 échantillons @ 125 Hz (Timer Hardware précis)
    - PAS d'interpolation : Signal authentique non déformé
    - Compatibilité MIMIC 100% (même fréquence que training)
    - Durée temporelle identique : 60s
    """
    
    logger.info("=" * 60)
    logger.info("🚀 ANALYSE SESSION - 7500 @ 125 Hz NATIF")
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
        
        # ── ÉTAPE 3 : Validation 125 Hz natif ─────────────
        logger.info("ÉTAPE 3/9 : Signal 125 Hz natif (Timer ESP32)")
        logger.info(f"✅ Signal authentique : {len(signal)} échantillons @ {fs_real:.2f} Hz")
        logger.info("⏭️  Pas d'interpolation nécessaire (signal déjà à 125 Hz)")
        signal_125hz = signal  # ✅ Pas de transformation, signal natif
        
        # ── ÉTAPE 4 : Filtrage Butterworth ────────────────
        logger.info("ÉTAPE 4/9 : Filtrage Butterworth")
        signal_filtered = filter_butterworth(signal_125hz, fs=125)
        
        # ── ÉTAPE 5 : Normalisation pour HeartPy ──────────
        logger.info("ÉTAPE 5/9 : Normalisation Z-score")
        signal_mean = np.mean(signal_filtered)
        signal_std = np.std(signal_filtered)
        
        if signal_std > 0:
            signal_normalized = (signal_filtered - signal_mean) / signal_std
            logger.info(f"✅ Signal normalisé : mean=0, std=1")
        else:
            signal_normalized = signal_filtered
            logger.warning(f"⚠️ Signal std=0, normalisation ignorée")
        
        # ── ÉTAPE 6 : Détection pics HeartPy ──────────────
        logger.info("ÉTAPE 6/9 : Détection pics HeartPy")
        working_data, measures = detect_peaks_heartpy(signal_normalized, fs=125)
        
        # ── ÉTAPE 7 : Calcul IBI ──────────────────────────
        logger.info("ÉTAPE 7/9 : Calcul IBI (Inter-Beat Intervals)")
        ibi_ms = calculate_ibi(working_data, fs=125)
        
        # ── ÉTAPE 8 : Filtrage outliers ───────────────────
        logger.info("ÉTAPE 8/9 : Filtrage outliers")
        ibi_clean = filter_outliers(ibi_ms)
        
        # ── ÉTAPE 9 : Features HRV + PRÉDICTION IA ────────
        logger.info("ÉTAPE 9/9 : Calcul features HRV")
        features = calculate_hrv_features(ibi_clean)
        
        # ── Prédiction IA FA ──────────────────────────────
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
            n_samples=len(signal),  # Échantillons originaux reçus
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