# ============================================================
# app/routes/session.py
# Version test : BPM uniquement (sans HRV ni XGBoost)
# ============================================================

from fastapi  import APIRouter, HTTPException
import numpy  as np

router = APIRouter()

@router.post("/analyze")
async def analyze_session(data: dict):
    """
    Version test :
    → Reçoit signal PPG brut
    → Calcule BPM via HeartPy
    → Retourne BPM + SpO2
    """

    # ── 1. Récupérer données ──────────────────────
    ir_signal  = data.get("ir_signal", [])
    fs         = data.get("fs", 100)
    spo2       = data.get("spo2", 0)

    if len(ir_signal) < 500:
        raise HTTPException(
            status_code = 400,
            detail      = "Signal PPG insuffisant"
        )

    # ── 2. Convertir en numpy ─────────────────────
    signal = np.array(ir_signal, dtype=np.float64)

    # ── 3. Filtrage Butterworth 0.5-8 Hz ──────────
    try:
        from scipy.signal import butter, filtfilt
        b, a          = butter(
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

    # ── 4. Calcul BPM via HeartPy ─────────────────
    try:
        import heartpy as hp
        working_data, measures = hp.process(
            signal_filtre,
            sample_rate = float(fs)
        )
        bpm = round(float(measures['bpm']), 1)

    except Exception as e:
        raise HTTPException(
            status_code = 422,
            detail      = f"Erreur HeartPy : {e}"
        )

    # ── 5. Retourner résultat ─────────────────────
    return {
        "success": True,
        "bpm"    : bpm,
        "spo2"   : spo2,
    }