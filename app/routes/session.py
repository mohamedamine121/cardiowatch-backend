# ============================================================
# app/routes/session.py - PIPELINE HRV COMPLET
# Modifié pour accepter 1500+ échantillons (60s @ 25Hz)
# Validation scientifique : PMC6953345, PMC4309304
# ============================================================
 
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import logging
 
# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
router = APIRouter()
 
# ============================================================
# MODÈLE DE DONNÉES
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
    
    logger.info(f"✅ Validation OK : {len(signal)} pts, range={signal_range:.0f}")
 
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
        
        logger.info(f"FS calculée depuis timestamps : {fs_real:.2f} Hz")
    else:
        # Méthode 2 : Estimation depuis longueur signal (60s attendu)
        duration_s = 60.0
        fs_real = len(signal) / duration_s
        
        logger.info(f"FS estimée depuis longueur : {fs_real:.2f} Hz (assumant 60s)")
    
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
    
    logger.info(f"Re-sampling : {n_original} pts @ {fs_original:.1f}Hz → {n_target} pts @ 125Hz")
    
    # Validation longueur cible (~7500 pour 60s)
    expected = 125 * 60
    if abs(len(signal_125hz) - expected) > 200:
        logger.warning(f"Longueur après resample : {len(signal_125hz)} (attendu ~{expected})")
    
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
    
    logger.info(f"✅ Filtrage Butterworth OK")
    
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
        
        logger.info(f"✅ HeartPy OK : {len(working_data['peaklist'])} pics détectés")
        
    except Exception as e:
        # ✅ Log détaillé pour debug
        logger.error(f"❌ HeartPy échoué : {str(e)}")
        logger.error(f"   Signal stats : min={np.min(signal):.1f}, max={np.max(signal):.1f}, mean={np.mean(signal):.1f}")
        
        raise HTTPException(
            status_code=422,
            detail=f"HeartPy détection échouée : {str(e)}"
        )
    
    # Validation BPM ÉLARGIE
    bpm = float(measures['bpm'])
    if not (30 <= bpm <= 180):  # ✅ Range élargi
        raise HTTPException(
            status_code=422,
            detail=f"BPM hors range : {bpm:.1f} (attendu 30-180)"
        )
    
    logger.info(f"✅ HeartPy : BPM={bpm:.1f}")
    
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
    
    logger.info(f"IBI calculés : {len(ibi_ms)} intervalles")
    
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
    
    # Filtre physiologique
    ibi_clean = remove_outliers(
        rr_intervals=ibi_ms.tolist(),
        low_rri=400,   # Production : 400 ms (150 BPM)
        high_rri=1500  # Production : 1500 ms (40 BPM)
    )
    
    logger.info(f"📊 Après filtre physio : {len(ibi_ms)} → {len(ibi_clean)} IBI")
    
    # Filtre Malik TRÈS RELÂCHÉ pour PPG
    try:
        ibi_clean = remove_ectopic_beats(
            rr_intervals=ibi_clean,
            method="malik",
            custom_removing_rule=0.50  # ✅ 50% tolérance PPG
        )
        logger.info(f"📊 Après filtre Malik 50% : {len(ibi_clean)} IBI")
    except Exception as e:
        # Si Malik échoue, garder filtre physio seulement
        logger.warning(f"⚠️ Malik échoué, filtre physio seul : {e}")
    
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
    
    logger.info(f"✅ Outliers filtrés : {len(ibi_ms)} → {len(ibi_clean)} IBI")
    
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
        'Entropy': round(shannon_entropy, 2)
    }
    
    logger.info(f"✅ Features HRV calculées : BPM={result['Mean_BPM']}, SDNN={result['SDNN']}")
    
    return result
 
# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================
 
@router.post("/analyze", response_model=HRVResponse)
async def analyze_session(data: SessionData):
    """
    Pipeline HRV complet avec détection FA
    Support : 1500+ échantillons (25Hz ou 100Hz)
    
    Modifications STEP 2 :
    - Accepte 1500+ échantillons (au lieu de 5400+)
    - Signal lissé matériellement (sampleAverage=4)
    - Validé scientifiquement pour 60s @ 25Hz
    """
    
    logger.info(f"========================================")
    logger.info(f"ANALYSE SESSION - Patient {data.patient_id}")
    logger.info(f"========================================")
    
    try:
        # ── Convertir en numpy ────────────────────────────
        signal = np.array(data.ppg_values, dtype=np.float64)
        
        # ── ÉTAPE 1 : Validation ──────────────────────────
        validate_signal(signal)
        
        # ── ÉTAPE 2 : Calcul FS réelle ────────────────────
        fs_real = calculate_real_fs(signal, data.timestamps_us)
        
        # ── ÉTAPE 3 : Re-échantillonnage 125 Hz ───────────
        signal_125hz = resample_to_125hz(signal, fs_real)
        
        # ── ÉTAPE 4 : Filtrage Butterworth ────────────────
        signal_filtered = filter_butterworth(signal_125hz, fs=125)
        
        # ── ÉTAPE 4.5 : Normalisation pour HeartPy ────────
        # HeartPy fonctionne mieux avec signal normalisé
        signal_mean = np.mean(signal_filtered)
        signal_std = np.std(signal_filtered)
        
        if signal_std > 0:
            signal_normalized = (signal_filtered - signal_mean) / signal_std
            logger.info(f"📊 Signal normalisé : mean=0, std=1")
        else:
            signal_normalized = signal_filtered
            logger.warning(f"⚠️ Signal std=0, normalisation ignorée")
        
        # ── ÉTAPE 5 : Détection pics HeartPy ──────────────
        working_data, measures = detect_peaks_heartpy(signal_normalized, fs=125)
        
        # ── ÉTAPE 6 : Calcul IBI ──────────────────────────
        ibi_ms = calculate_ibi(working_data, fs=125)
        
        # ── ÉTAPE 7 : Filtrage outliers ───────────────────
        ibi_clean = filter_outliers(ibi_ms)
        
        # ── ÉTAPE 8 : Features HRV ────────────────────────
        features = calculate_hrv_features(ibi_clean)
        
        # ── Retour résultat ───────────────────────────────
        logger.info(f"✅ SUCCÈS - HRV calculées")
        logger.info(f"========================================")
        
        return HRVResponse(
            status="success",
            spo2=data.spo2,
            mean_bpm=features['Mean_BPM'],
            sdnn=features['SDNN'],
            rmssd=features['RMSSD'],
            pnn50=features['pNN50'],
            entropy=features['Entropy'],
            fs_real=round(fs_real, 1),
            n_samples=len(signal)
        )
        
    except HTTPException:
        # Re-raise HTTPException directement
        raise
    
    except Exception as e:
        logger.error(f"❌ ERREUR INATTENDUE : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pipeline HRV : {str(e)}"
        )