import requests
import numpy as np

# ============================================================
# TEST BACKEND - 1500 ÉCHANTILLONS @ 25Hz
# Test du backend modifié pour accepter signal lissé (avg=4)
# ============================================================

# URL backend Render
BACKEND_URL = "https://cardiowatch-backend.onrender.com/api/session/analyze"

print("=" * 60)
print("TEST BACKEND - 1500 ÉCHANTILLONS @ 25Hz")
print("Simulation signal lissé (sampleAverage=4)")
print("=" * 60)

# ── Générer signal PPG fictif ──────────────────────────────
# Simule un signal cardiaque @ 25 Hz pendant 60 secondes
print("\n🔧 Génération signal PPG fictif...")

fs = 25  # Hz (avec sampleAverage=4 sur ESP32)
duration = 60  # secondes
n_samples = fs * duration  # 1500 échantillons

print(f"   Fréquence : {fs} Hz")
print(f"   Durée     : {duration}s")
print(f"   Échantillons : {n_samples}")

# Signal sinusoïdal (simule battements cardiaques @ 75 BPM)
t = np.linspace(0, duration, n_samples)
bpm = 75
frequency = bpm / 60  # Hz

# Signal de base (valeurs typiques MAX30102)
baseline = 185000
amplitude = 3000  # Amplitude réduite car signal lissé

signal = baseline + amplitude * np.sin(2 * np.pi * frequency * t)

# Ajouter bruit léger (signal déjà lissé par avg=4)
noise = np.random.normal(0, 150, n_samples)
signal = signal + noise

# Convertir en liste d'entiers
ppg_values = [int(v) for v in signal]

print(f"\n✅ Signal généré : {len(ppg_values)} échantillons @ {fs}Hz")
print(f"   Durée : {duration}s")
print(f"   Range : {min(ppg_values)} - {max(ppg_values)}")
print(f"   Amplitude : {max(ppg_values) - min(ppg_values)}")

# ── Préparer données POST ──────────────────────────────────
data = {
    "patient_id": "test_user_1500_25hz",
    "ppg_values": ppg_values,
    "spo2": 98,
    "timestamp": "2026-05-04T21:30:00Z"
}

print(f"\n📡 Envoi requête POST vers backend...")
print(f"URL : {BACKEND_URL}")
print(f"Payload : {len(ppg_values)} échantillons, SpO2={data['spo2']}%")

# ── Envoyer requête ────────────────────────────────────────
try:
    print("\n⏳ Envoi en cours... (peut prendre 10-30s)")
    
    response = requests.post(
        BACKEND_URL,
        json=data,
        timeout=60  # 60 secondes timeout
    )
    
    print(f"\n📥 Réponse reçue !")
    print(f"Status Code : {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ SUCCÈS ! Backend accepte 1500 échantillons !\n")
        print("=" * 60)
        print("RÉSULTATS HRV")
        print("=" * 60)
        print(f"Status       : {result.get('status')}")
        print(f"SpO2         : {result.get('spo2')} %")
        print(f"Mean BPM     : {result.get('mean_bpm')} BPM")
        print(f"SDNN         : {result.get('sdnn')} ms")
        print(f"RMSSD        : {result.get('rmssd')} ms")
        print(f"pNN50        : {result.get('pnn50')} %")
        print(f"Entropy      : {result.get('entropy')}")
        print(f"FS réelle    : {result.get('fs_real')} Hz")
        print(f"N samples    : {result.get('n_samples')}")
        print("=" * 60)
        print("\n🎉 TEST RÉUSSI - BACKEND MODIFIÉ FONCTIONNE !\n")
        print("✅ Backend accepte maintenant 1500 échantillons @ 25Hz")
        print("✅ Pipeline HRV complet validé")
        print("✅ Prêt pour test ESP32 réel")
        print("\n" + "=" * 60)
        
    elif response.status_code == 400:
        print(f"\n❌ ERREUR 400 - Validation échouée")
        print(f"Message : {response.text}")
        print("\n⚠️ Si erreur 'Signal trop court < 5400':")
        print("   → Le backend N'A PAS encore été mis à jour")
        print("   → Déployer le nouveau session.py sur Render")
        
    elif response.status_code == 422:
        print(f"\n❌ ERREUR 422 - Traitement échoué")
        print(f"Message : {response.text}")
        print("\n💡 Possible cause : HeartPy détection")
        
    else:
        print(f"\n❌ ERREUR {response.status_code}")
        print(f"Message : {response.text}")
        
except requests.exceptions.Timeout:
    print("\n⏱️ TIMEOUT - Le backend prend trop de temps (>60s)")
    print("Causes possibles :")
    print("  - Cold start Render (1er appel après inactivité)")
    print("  - Traitement HeartPy long")
    print("\n💡 Solution : Réessaye dans 1-2 minutes")
    
except requests.exceptions.ConnectionError:
    print("\n❌ ERREUR CONNEXION - Impossible de joindre le backend")
    print("Vérifier :")
    print("  - URL backend correcte")
    print("  - Backend Render actif")
    print("  - Connexion Internet")
    
except Exception as e:
    print(f"\n❌ ERREUR INATTENDUE : {e}")

print("\n" + "=" * 60)
print("FIN DU TEST")
print("=" * 60)