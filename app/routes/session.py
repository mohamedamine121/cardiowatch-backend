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

# ✅ AJOUTÉ: Imports hrv-analysis pour alignment avec training MIMIC
from hrvanalysis import (
    remove_outliers,
    remove_ectopic_beats,
    interpolate_nan_values,
    get_time_domain_features
)

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
    ✅ ALIGNÉ TRAINING MIMIC : high_precision=True + clean_rr=True
    """
    import heartpy as hp
    
    logger.info("🔍 Détection pics HeartPy...")
    logger.info(f"   - Signal length : {len(signal)}")
    logger.info(f"   - Sampling rate : {fs} Hz")
    
    try:
        # ✅ MODIFIÉ : Paramètres alignés sur training MIMIC
        working_data, measures = hp.process(
            signal,
            sample_rate=fs,
            bpmmin=30,                  # Range large pour PPG
            bpmmax=180,                 # Range large pour PPG
            high_precision=True,        # ✅ MODIFIÉ: False → True (training MIMIC)
            clean_rr=True,              # ✅ AJOUTÉ: Nettoyage auto IBI (training MIMIC)
            clean_rr_method='iqr',      # ✅ AJOUTÉ: Méthode IQR (training MIMIC)
            reject_segmentwise=False    # Désactiver rejet segments
        )
        
        n_peaks = len(working_data['peaklist'])
        logger.info(f"✅ HeartPy détection réussie : {n_peaks} pics détectés")
        logger.info(f"   - high_precision : True (algorithme Pan-Tompkins amélioré)")
        logger.info(f"   - clean_rr       : True (nettoyage auto outliers IQR)")
        
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
    
    # ── Niveau 2 : Filtre Malik 20% (TRAINING MIMIC) ─────────
    # ✅ MODIFIÉ : Malik 20% (standard médical Task Force 1996)
    # Training MIMIC utilise remove_ectopic_beats avec Malik 20% implicite
    try:
        logger.info(f"🔍 Avant Malik - {len(ibi_clean)} IBI")
        
        # ÉTAPE 2a : Remove outliers (400-1500 ms comme training)
        ibi_outliers = remove_outliers(
            rr_intervals=ibi_clean,
            low_rri=400,
            high_rri=1500,
            verbose=False
        )
        
        # ÉTAPE 2b : Malik filter 20% (méthode training MIMIC)
        ibi_malik = remove_ectopic_beats(
            rr_intervals=ibi_outliers,
            method='malik',  # ✅ Malik 20% par défaut (training MIMIC)
            verbose=False
        )
        
        # Vérifier résultat
        if ibi_malik is None or len(ibi_malik) == 0:
            logger.warning(f"⚠️ Malik a retourné None ou liste vide")
            ibi_malik = ibi_clean
        else:
            logger.info(f"📊 Filtre Malik 20%     : {n_after_physio} → {len(ibi_malik)} IBI")
        
        # ✅ MODIFIÉ : INTERPOLATION NaN au lieu de SUPPRESSION (training MIMIC)
        # Training utilise interpolate_nan_values pour préserver continuité temporelle
        logger.info(f"🔍 Interpolation NaN (méthode training MIMIC)...")
        ibi_clean = interpolate_nan_values(
            rr_intervals=ibi_malik,
            interpolation_method='linear'  # Méthode training MIMIC
        )
        
        # Convertir en array numpy et supprimer NaN restants (edge cases)
        ibi_clean = np.array(ibi_clean, dtype=np.float64)
        ibi_clean = ibi_clean[~np.isnan(ibi_clean)]
        
        logger.info(f"📊 Après interpolation   : {len(ibi_clean)} IBI valides")
        
    except Exception as e:
        # Si Malik échoue, garder filtre physio seulement
        logger.error(f"❌ EXCEPTION Malik/Interpolation : {e}")
        import traceback
        logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
        logger.warning(f"⚠️ Utilisation filtre physio seul")
        ibi_clean = np.array(ibi_clean, dtype=np.float64)
    
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
    ✅ ALIGNÉ TRAINING MIMIC : get_time_domain_features + Entropie exacte
    """
    # Mean BPM depuis IBI
    mean_bpm = 60000.0 / np.mean(ibi_clean)
    
    # Features HRV avec bibliothèque hrv-analysis (training MIMIC)
    features = get_time_domain_features(ibi_clean.tolist())
    
    # ✅ Entropie Shannon EXACTE (training MIMIC)
    # Copié EXACTEMENT du notebook training
    hist, _ = np.histogram(ibi_clean, bins=20, density=True)  # ✅ bins=20 (training)
    hist = hist + 1e-10          # ✅ Éviter log(0)
    hist = hist / hist.sum()     # ✅ Normaliser
    shannon_entropy = float(-np.sum(hist * np.log2(hist)))  # ✅ Formule exacte
    
    result = {
        'Mean_BPM': round(mean_bpm, 1),
        'SDNN': round(features['sdnn'], 1),
        'RMSSD': round(features['rmssd'], 1),
        'pNN50': round(features['pnni_50'], 1),  # hrv-analysis utilise pnni_50
        'Entropy': round(shannon_entropy, 4)
    }
    
    logger.info("=" * 60)
    logger.info("📊 FEATURES HRV CALCULÉES (Training MIMIC)")
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
        logger.info("ÉTAPE 1/8 : Validation signal")
        validate_signal(signal)
        
        # ── ÉTAPE 2 : Calcul FS réelle ────────────────────
        logger.info("ÉTAPE 2/8 : Calcul fréquence d'échantillonnage")
        fs_real = calculate_real_fs(signal, data.timestamps_us)
        
        # ── ÉTAPE 3 : Validation 125 Hz natif ─────────────
        logger.info("ÉTAPE 3/8 : Signal 125 Hz natif (Timer ESP32)")
        logger.info(f"✅ Signal authentique : {len(signal)} échantillons @ {fs_real:.2f} Hz")
        logger.info("⏭️  Pas d'interpolation nécessaire (signal déjà à 125 Hz)")
        signal_125hz = signal  # ✅ Pas de transformation, signal natif
        
        # ── ÉTAPE 4 : Filtrage Butterworth ────────────────
        logger.info("ÉTAPE 4/8 : Filtrage Butterworth")
        signal_filtered = filter_butterworth(signal_125hz, fs=125)
        
        # ── ÉTAPE 5 : Détection pics HeartPy ──────────────
        # ✅ ALIGNÉ TRAINING MIMIC : high_precision=True + clean_rr=True
        # Normalisation Z-score SUPPRIMÉE (training n'en utilise pas)
        logger.info("ÉTAPE 5/8 : Détection pics HeartPy (signal brut)")
        working_data, measures = detect_peaks_heartpy(signal_filtered, fs=125)
        
        # ── ÉTAPE 6 : Calcul IBI ──────────────────────────
        logger.info("ÉTAPE 6/8 : Calcul IBI (Inter-Beat Intervals)")
        ibi_ms = calculate_ibi(working_data, fs=125)
        
        # ── ÉTAPE 7 : Filtrage outliers + Malik 20% ───────
        # ✅ ALIGNÉ TRAINING MIMIC : Malik 20% + interpolation NaN
        logger.info("ÉTAPE 7/8 : Filtrage outliers + Malik 20%")
        ibi_clean = filter_outliers(ibi_ms)
        
        # ── ÉTAPE 8 : Features HRV + PRÉDICTION IA ────────
        logger.info("ÉTAPE 8/8 : Calcul features HRV")
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