# ============================================================
# app/routes/session.py
# ============================================================

from fastapi import APIRouter, HTTPException
import numpy as np

router = APIRouter()

@router.post("/analyze")
async def analyze_session(data: dict):

    # ── 1. Récupérer données ──────────────────────
    ir_signal  = data.get("ir_signal", [])
    fs         = data.get("fs", 100)
    spo2       = data.get("spo2", 0)

    if len(ir_signal) < 50:
        raise HTTPException(
            status_code = 400,
            detail      = "Signal PPG insuffisant"
        )

    # ── 2. Convertir en numpy ─────────────────────
    signal = np.array(ir_signal, dtype=np.float64)

    # ── 3. Vérifier signal valide ─────────────────
    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 100:
        raise HTTPException(
            status_code = 422,
            detail      = "Signal PPG plat — pas de pulsation détectée"
        )

    # ── 4. Filtrage Butterworth 0.5-8 Hz ──────────
    try:
        from scipy.signal import butter, filtfilt
        b, a = butter(
            3,
            [0.5 / (fs/2), 8.0 / (fs/2)],
            btype = 'bandpass'
        )
        signal_filtre = filtfilt(b, a, signal)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Erreur filtrage : {e}"
        )

    # ── 5. Calcul BPM via HeartPy ─────────────────
    try:
        import heartpy as hp

        # Paramètres HeartPy optimisés pour PPG
        working_data, measures = hp.process(
            signal_filtre,
            sample_rate    = float(fs),
            bpmmin         = 40,
            bpmmax         = 180,
            hampel_correct = False,
            clean_rr       = True,
        )
        bpm = round(float(measures['bpm']), 1)

    except Exception as e:
        raise HTTPException(
            status_code = 422,
            detail      = f"Erreur HeartPy : {e}"
        )

    # ── 6. Retourner résultat ─────────────────────
    return {
        "success": True,
        "bpm"    : bpm,
        "spo2"   : spo2,
    }