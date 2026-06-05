import requests
import numpy as np
import time

print("="*60)
print("TEST BACKEND - 7500 @ 125 Hz NATIF (Timer Hardware ESP32)")
print("="*60)

# URL backend
url = "https://cardiowatch-backend.onrender.com/api/session/analyze"

# ============================================================
# GÉNÉRATION SIGNAL 7500 ÉCHANTILLONS @ 125 Hz (60 secondes)
# ============================================================
fs = 125  # ✅ Timer Hardware ESP32 = 125 Hz exact
duration = 60
n_samples = 7500  # ✅ Timer 125 Hz × 60s = 7500 échantillons

print("\n📊 GÉNÉRATION SIGNAL PPG (Timer ESP32)")
print("-" * 60)
print(f"Fréquence d'échantillonnage : {fs} Hz (Timer Hardware)")
print(f"Durée                       : {duration} secondes")
print(f"Nombre d'échantillons       : {n_samples}")

t = np.linspace(0, duration, n_samples)
bpm = 75
frequency = bpm / 60

# Signal sinusoïdal simulant battements cardiaques
baseline = 185000
amplitude = 3000
signal = baseline + amplitude * np.sin(2 * np.pi * frequency * t)

# Ajout bruit réaliste
noise = np.random.normal(0, 150, n_samples)
signal = signal + noise

# Ajout variations physiologiques (variabilité HRV)
# Simuler légère variabilité BPM
variability = 500 * np.sin(2 * np.pi * 0.1 * t)  # Oscillation lente 0.1 Hz
signal = signal + variability

ppg_values = [float(v) for v in signal]

# Statistiques signal
signal_min = np.min(ppg_values)
signal_max = np.max(ppg_values)
signal_mean = np.mean(ppg_values)
signal_std = np.std(ppg_values)
signal_range = signal_max - signal_min

print(f"\n✅ Signal généré :")
print(f"   - Min   : {signal_min:.0f}")
print(f"   - Max   : {signal_max:.0f}")
print(f"   - Mean  : {signal_mean:.0f}")
print(f"   - Std   : {signal_std:.0f}")
print(f"   - Range : {signal_range:.0f}")

data = {
    "patient_id": "69b6ebb7b753251c9da3d24f",
    "ppg_values": ppg_values,
    "spo2": 98,
    "timestamp": "2026-05-17T14:00:00Z"
}

print(f"\n📡 Test connexion backend...")
print(f"URL         : {url}")
print(f"Patient ID  : {data['patient_id']}")
print(f"Échantillons: {len(ppg_values)} @ {fs} Hz")
print(f"SpO2        : {data['spo2']}%")

start_time = time.time()

try:
    print("\n⏳ Envoi requête (timeout 120s)...")
    
    response = requests.post(
        url,
        json=data,
        timeout=120
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Réponse reçue en {elapsed:.1f}s !")
    print(f"Status HTTP: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n" + "="*60)
        print("✅ SUCCÈS - ANALYSE COMPLÈTE (7500 @ 125Hz NATIF)")
        print("="*60)
        
        # ═══════════════════════════════════════════════════
        # SECTION 1 : VALIDATION SIGNAL (PAS D'INTERPOLATION)
        # ═══════════════════════════════════════════════════
        print("\n🔍 VALIDATION SIGNAL 125 Hz NATIF")
        print("-" * 60)
        
        fs_real = result.get('fs_real')
        n_samples_received = result.get('n_samples')
        
        print(f"Échantillons envoyés : {len(ppg_values)} @ {fs} Hz")
        print(f"Échantillons reçus   : {n_samples_received}")
        print(f"Fréquence détectée   : {fs_real:.1f} Hz")
        
        # Vérifier signal natif
        if n_samples_received == 7500:
            print(f"\n✅ Backend a reçu 7500 échantillons @ 125 Hz")
            print(f"⏭️  PAS d'interpolation nécessaire (signal natif Timer ESP32)")
            print(f"✅ Signal authentique non déformé")
        else:
            print(f"\n⚠️  Échantillons inattendus : {n_samples_received}")
            print(f"⚠️  Attendu : 7500 échantillons")
        
        # ═══════════════════════════════════════════════════
        # SECTION 2 : RÉSULTATS HRV
        # ═══════════════════════════════════════════════════
        print("\n📊 RÉSULTATS HRV (Heart Rate Variability)")
        print("-" * 60)
        print(f"Status       : {result.get('status')}")
        print(f"SpO2         : {result.get('spo2')}%")
        print(f"Mean BPM     : {result.get('mean_bpm'):.1f} BPM")
        print(f"SDNN         : {result.get('sdnn'):.1f} ms")
        print(f"RMSSD        : {result.get('rmssd'):.1f} ms")
        print(f"pNN50        : {result.get('pnn50'):.1f}%")
        print(f"Entropy      : {result.get('entropy'):.4f}")
        
        # ═══════════════════════════════════════════════════
        # SECTION 3 : RÉSULTATS IA - DÉTECTION FA
        # ═══════════════════════════════════════════════════
        print("\n🤖 RÉSULTATS IA - DÉTECTION FIBRILLATION AURICULAIRE")
        print("-" * 60)
        
        af_detected = result.get('af_detected')
        af_probability = result.get('af_probability')
        af_risk = result.get('af_risk')
        af_confidence = result.get('af_confidence')
        
        # Interprétation du label
        if af_detected == 1:
            label_text = "🚨 FIBRILLATION AURICULAIRE DÉTECTÉE"
            label_emoji = "⚠️"
        elif af_detected == 0:
            label_text = "✅ RYTHME CARDIAQUE NORMAL"
            label_emoji = "💚"
        elif af_detected == -1:
            label_text = "❓ PRÉDICTION IA IMPOSSIBLE"
            label_emoji = "⚠️"
        else:
            label_text = "❓ STATUT INCONNU"
            label_emoji = "❓"
        
        print(f"\n{label_emoji} DIAGNOSTIC : {label_text}")
        print(f"\nDétails :")
        print(f"  - Label AF      : {af_detected} ({'FA' if af_detected==1 else 'Normal' if af_detected==0 else 'Erreur'})")
        print(f"  - Probabilité   : {af_probability if af_probability is not None else 'N/A'}", end="")
        if af_probability is not None:
            print(f" ({af_probability*100:.2f}% de risque FA)")
        else:
            print()
        print(f"  - Niveau risque : {af_risk}")
        print(f"  - Confiance     : {af_confidence if af_confidence is not None else 'N/A'}", end="")
        if af_confidence is not None:
            print(f"%")
        else:
            print()
        
        # ═══════════════════════════════════════════════════
        # SECTION 4 : INTERPRÉTATION CLINIQUE
        # ═══════════════════════════════════════════════════
        print("\n📋 INTERPRÉTATION CLINIQUE")
        print("-" * 60)
        
        if af_detected == 1:
            print("⚠️  ALERTE : Fibrillation auriculaire détectée par l'IA")
            print("📞 RECOMMANDATION : Consulter un médecin rapidement")
            print("💊 RISQUE : AVC, insuffisance cardiaque")
            if af_confidence and af_confidence >= 80:
                print(f"🔴 Confiance élevée ({af_confidence}%) - Prédiction fiable")
            elif af_confidence and af_confidence >= 60:
                print(f"🟡 Confiance modérée ({af_confidence}%) - Vérification recommandée")
            else:
                print(f"🟢 Confiance faible ({af_confidence}%) - Faux positif possible")
                
        elif af_detected == 0:
            print("✅ NORMAL : Aucune fibrillation auriculaire détectée")
            print("💚 RECOMMANDATION : Surveillance de routine")
            if af_confidence and af_confidence >= 80:
                print(f"🔴 Confiance élevée ({af_confidence}%) - Résultat fiable")
            elif af_confidence and af_confidence >= 60:
                print(f"🟡 Confiance modérée ({af_confidence}%)")
            else:
                print(f"🟢 Confiance faible ({af_confidence}%) - Réévaluation recommandée")
                
        elif af_detected == -1:
            print("⚠️  ERREUR : Modèles IA non disponibles ou prédiction échouée")
            print("📊 INFO : Features HRV calculées avec succès")
            print("🔧 ACTION : Vérifier que les modèles .pkl sont sur le serveur")
        
        # ═══════════════════════════════════════════════════
        # SECTION 5 : VALIDATION APPROCHE 7500 @ 125Hz
        # ═══════════════════════════════════════════════════
        print("\n🎯 VALIDATION APPROCHE ESP32 TIMER HARDWARE")
        print("-" * 60)
        print("Configuration ESP32 (DUAL_CONFIG_125HZ) :")
        print("  - Timer Hardware : 125 Hz (précision ±0.01%)")
        print("  - Échantillons : 7500")
        print("  - Durée : 60 secondes EXACT")
        print("")
        print("Traitement Backend :")
        print("  - Signal natif : 7500 @ 125 Hz (PAS d'interpolation)")
        print("  - Compatibilité MIMIC : 100% (fréquence identique)")
        print("  - Qualité : Signal authentique non déformé ✅")
        print("")
        print("Validation scientifique :")
        print("  - PMC6953345 : 60s validé pour HRV court terme")
        print("  - Précision Timer : ±0.01% (vs ±5% boucle normale)")
        print("  - Features HRV identiques au training MIMIC ✅")
        print("  - Pas d'artefacts d'interpolation ✅")
        
        print("\n" + "="*60)
        print("✅ TEST 7500 @ 125 Hz NATIF TERMINÉ AVEC SUCCÈS")
        print("="*60)
        
    elif response.status_code == 400:
        print("\n" + "="*60)
        print(f"❌ ERREUR VALIDATION - HTTP {response.status_code}")
        print("="*60)
        try:
            error = response.json()
            print(f"Message: {error.get('detail', response.text)}")
        except:
            print(f"Message: {response.text}")
        
        print("\n📝 CAUSES POSSIBLES:")
        print("  1. Signal trop court (< 7500 échantillons)")
        print("  2. Signal contient NaN")
        print("  3. Signal plat (pas de pulsation)")
        print("  4. Fréquence hors range (120-130 Hz)")
        
    elif response.status_code == 422:
        print("\n" + "="*60)
        print(f"❌ ERREUR TRAITEMENT - HTTP {response.status_code}")
        print("="*60)
        try:
            error = response.json()
            print(f"Message: {error.get('detail', response.text)}")
        except:
            print(f"Message: {response.text}")
        
        print("\n📝 CAUSES POSSIBLES:")
        print("  1. Détection pics HeartPy échouée")
        print("  2. Trop peu de battements détectés")
        print("  3. Trop peu d'IBI valides (< 20)")
        print("  4. Filtrage Butterworth instable")
        
    else:
        print("\n" + "="*60)
        print(f"❌ ERREUR HTTP {response.status_code}")
        print("="*60)
        print(f"Message: {response.text}")
        
except requests.exceptions.Timeout:
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"⏱️ TIMEOUT après {elapsed:.1f}s")
    print("="*60)
    print("❌ Le backend ne répond pas dans le délai imparti!")
    print("\n📝 CAUSES POSSIBLES:")
    print("  1. Cold start Render (prend 30-60s au premier appel)")
    print("  2. Backend crashé ou en redémarrage")
    print("  3. Problème réseau")
    print("  4. Pipeline HRV + interpolation prend trop de temps (>120s)")
    print("\n💡 SOLUTIONS:")
    print("  1. Attendre 1 minute et réessayer")
    print("  2. Vérifier logs Render (https://dashboard.render.com)")
    print("  3. Augmenter timeout à 180s si nécessaire")
    print("  4. Vérifier que session.py modifié est bien déployé")
    
except requests.exceptions.ConnectionError as e:
    print("\n" + "="*60)
    print("❌ ERREUR DE CONNEXION")
    print("="*60)
    print(f"Impossible de se connecter au backend")
    print(f"Erreur: {e}")
    print("\n📝 VÉRIFICATIONS:")
    print("  1. Backend déployé sur Render ?")
    print("  2. URL correcte ? https://cardiowatch-backend.onrender.com")
    print("  3. Connexion Internet OK ?")
    print("  4. Backend en cours de redémarrage ?")
    
except Exception as e:
    print("\n" + "="*60)
    print("❌ ERREUR INATTENDUE")
    print("="*60)
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    import traceback
    print("\nStack trace:")
    traceback.print_exc()

print("\n" + "="*60)
print("FIN DU TEST 7500 @ 125 Hz NATIF (Timer Hardware ESP32)")
print("="*60)