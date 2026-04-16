# ============================================================
# app/routes/session.py
# ============================================================

from fastapi import APIRouter, HTTPException
import numpy as np

router = APIRouter()

@router.post("/analyze")
async def analyze_session(data: dict):

    # ── 1. Récupérer données ──────────────────────
    ir_signal  = data.get("ir_signal",  [])
    fs         = data.get("fs",         100)
    spo2       = data.get("spo2",       0)
    duration   = data.get("duration_ms",0)
    fs_reel    = data.get("fs_reel",    0)

    if len(ir_signal) < 50:
        raise HTTPException(
            status_code = 400,
            detail      = "Signal PPG insuffisant"
        )

    # ── 2. Calculer fs réelle ─────────────────────
    # Priorité 1 : fs_reel depuis ESP32 (timestamps)
    # Priorité 2 : calculer depuis duration_ms
    # Priorité 3 : défaut 100 Hz
    if fs_reel > 0 and 20 <= fs_reel <= 150:
        fs_final = float(fs_reel)
    elif duration > 0:
        fs_calc  = len(ir_signal) / (duration / 1000.0)
        fs_final = fs_calc if 20 <= fs_calc <= 150 \
                   else float(fs)
    else:
        fs_final = float(fs)

    print(f"fs_reel reçue : {fs_reel} Hz")
    print(f"fs_final utilisée : {fs_final:.2f} Hz")
    print(f"n_pts : {len(ir_signal)}")

    # ── 3. Convertir en numpy ─────────────────────
    signal = np.array(ir_signal, dtype=np.float64)

    # ── 4. Vérifier signal valide ─────────────────
    signal_range = np.max(signal) - np.min(signal)
    if signal_range < 100:
        raise HTTPException(
            status_code = 422,
            detail      = "Signal PPG plat — "
                          "pas de pulsation détectée"
        )

    # ── 5. Filtrage Butterworth 0.5-8 Hz ──────────
    try:
        from scipy.signal import butter, filtfilt
        nyq  = fs_final / 2.0
        low  = 0.5  / nyq
        high = min(8.0 / nyq, 0.99)
        b, a = butter(3, [low, high], btype='bandpass')
        signal_filtre = filtfilt(b, a, signal)
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail      = f"Erreur filtrage : {e}"
        )

    # ── 6. Calcul BPM via HeartPy ─────────────────
    try:
        import heartpy as hp

        working_data, measures = hp.process(
            signal_filtre,
            sample_rate    = fs_final,
            bpmmin         = 40,
            bpmmax         = 180,
            hampel_correct = False,
            clean_rr       = True,
        )
        bpm = round(float(measures['bpm']), 1)
        print(f"BPM calculé : {bpm}")

    except Exception as e:
        raise HTTPException(
            status_code = 422,
            detail      = f"Erreur HeartPy : {e}"
        )

    # ── 7. Retourner résultat ─────────────────────
    return {
        "success" : True,
        "bpm"     : bpm,
        "spo2"    : spo2,
        "fs_reel" : round(fs_final, 1),
        "n_pts"   : len(ir_signal),
    }