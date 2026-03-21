from fastapi import APIRouter, HTTPException
from app.models.user import MedecinRegister, PatientRegister, LoginData, TokenResponse
from app.database import medecins_collection, patients_collection
from app.services.auth_service import hash_password, verify_password, create_jwt, generate_medecin_id
from datetime import datetime
from datetime import datetime, timedelta


router = APIRouter()

# ── Inscription Médecin ───────────────────────────
@router.post("/register/medecin")
async def register_medecin(data: MedecinRegister):

    # Vérifier si email existe déjà
    existing = await medecins_collection.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    # Générer identifiant unique médecin
    medecin_id = generate_medecin_id()

    # Créer le document médecin
    medecin = {
        "identifiant"  : medecin_id,
        "nom"          : data.nom,
        "email"        : data.email,
        "password"     : hash_password(data.password),
        "specialite"   : data.specialite,
        "telephone"    : data.telephone,
        "created_at"   : datetime.utcnow()
    }

    # Sauvegarder dans MongoDB
    await medecins_collection.insert_one(medecin)

    return {
        "message"      : "Médecin inscrit avec succès",
        "identifiant"  : medecin_id,
        "nom"          : data.nom,
        "email"        : data.email
    }


# ── Inscription Patient ───────────────────────────
@router.post("/register/patient")
async def register_patient(data: PatientRegister):

    # Vérifier si email existe déjà
    existing = await patients_collection.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    # Vérifier que le médecin existe
    medecin = await medecins_collection.find_one({"identifiant": data.medecin_id})
    if not medecin:
        raise HTTPException(status_code=404, detail="Identifiant médecin invalide")

    # Créer le document patient
    patient = {
        "nom"          : data.nom,
        "email"        : data.email,
        "password"     : hash_password(data.password),
        "age"          : data.age,
        "medecin_id"   : str(medecin["_id"]),
        "medecin_identifiant" : data.medecin_id,
        "created_at"   : datetime.utcnow()
    }

    # Sauvegarder dans MongoDB
    await patients_collection.insert_one(patient)

    return {
        "message"      : "Patient inscrit avec succès",
        "nom"          : data.nom,
        "email"        : data.email,
        "medecin"      : medecin["nom"]
    }


# ── Login ─────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(data: LoginData):

    # ── CAS 1 : LOGIN MÉDECIN ──────────────────────
    if data.role == "medecin":

        medecin = await medecins_collection.find_one({"email": data.email})
        if not medecin:
            raise HTTPException(status_code=404, detail="Médecin introuvable")

        if not verify_password(data.password, medecin["password"]):
            raise HTTPException(status_code=401, detail="Mot de passe incorrect")

        token = create_jwt({
            "medecin_id"   : str(medecin["_id"]),
            "identifiant"  : medecin["identifiant"],
            "role"         : "medecin"
        })

        return {
            "access_token" : token,
            "role"         : "medecin",
            "nom"          : medecin["nom"],
            "email"        : medecin["email"]
        }

    # ── CAS 2 : LOGIN PATIENT ──────────────────────
    elif data.role == "patient":

        if not data.medecin_id:
            raise HTTPException(status_code=400, detail="Identifiant médecin requis")

        # Vérifier médecin
        medecin = await medecins_collection.find_one({"identifiant": data.medecin_id})
        if not medecin:
            raise HTTPException(status_code=404, detail="Identifiant médecin invalide")

        # Vérifier patient
        patient = await patients_collection.find_one({"email": data.email})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient introuvable")

        # Vérifier que le patient appartient à ce médecin
        if patient["medecin_identifiant"] != data.medecin_id:
            raise HTTPException(status_code=403, detail="Ce patient n'appartient pas à ce médecin")

        if not verify_password(data.password, patient["password"]):
            raise HTTPException(status_code=401, detail="Mot de passe incorrect")

        token = create_jwt({
            "patient_id"   : str(patient["_id"]),
            "medecin_id"   : str(medecin["_id"]),
            "role"         : "patient"
        })

        return {
            "access_token" : token,
            "role"         : "patient",
            "nom"          : patient["nom"],
            "email"        : patient["email"],
            "medecinNom"   : medecin["nom"],
            "age"          : patient["age"],
            "patient_id"     : str(patient["_id"]),
            "telephone"      : patient.get("telephone", ""),
            "groupe_sanguin" : patient.get("groupe_sanguin", ""),
            "poids"          : patient.get("poids", ""),
            "taille"         : patient.get("taille", "")

        }

    else:
        raise HTTPException(status_code=400, detail="Rôle invalide")
 
 # ── Mise à jour profil patient ────────────────────
@router.put("/update/patient/{patient_id}")
async def update_patient(
    patient_id: str,
    data      : dict
):
    from bson import ObjectId

    # Champs autorisés à modifier
    update_data = {}
    if "telephone"    in data: update_data["telephone"]    = data["telephone"]
    if "groupe_sanguin" in data: update_data["groupe_sanguin"] = data["groupe_sanguin"]
    if "poids"        in data: update_data["poids"]        = data["poids"]
    if "taille"       in data: update_data["taille"]       = data["taille"]

    if not update_data:
        raise HTTPException(
            status_code=400,
            detail="Aucune donnée à mettre à jour"
        )

    await patients_collection.update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": update_data}
    )

    return {"message": "Profil mis à jour avec succès"}

    # ── Historique 7 jours ────────────────────────────
@router.get("/history/{patient_id}")
async def get_history(patient_id: str):
    from bson import ObjectId
    from app.database import db

    # 7 derniers jours
    today     = datetime.utcnow().replace(
                  hour=0, minute=0, second=0, microsecond=0)
    week_ago  = today - timedelta(days=6)

    # Récupérer toutes les fenêtres HRV des 7 derniers jours
    cursor = db.hrv_windows.find({
        "patient_id" : patient_id,
        "timestamp"  : {"$gte": week_ago}
    }).sort("timestamp", 1)

    windows = await cursor.to_list(length=1000)

    # Organiser par jour
    days_data = {}
    for w in windows:
        day_key = w["timestamp"].strftime("%Y-%m-%d")
        if day_key not in days_data:
            days_data[day_key] = []

        days_data[day_key].append({
            "minute"    : w.get("minute", 0),
            "timestamp" : w["timestamp"].strftime("%H:%M"),
            "bpm"       : w.get("mean_bpm", 0),
            "spo2"      : w.get("spo2", 0),
            "label"     : w.get("label", 0),
            "status"    : w.get("status", "Normal"),
        })

    # Construire les 7 jours
    result = []
    for i in range(7):
        day       = week_ago + timedelta(days=i)
        day_key   = day.strftime("%Y-%m-%d")
        sessions  = days_data.get(day_key, [])

        # Statut du jour
        if not sessions:
            day_status = "empty"
        elif any(s["label"] == 1 for s in sessions):
            day_status = "fa"
        else:
            day_status = "normal"

        result.append({
            "date"      : day_key,
            "day_name"  : day.strftime("%a"),
            "day_number": day.day,
            "status"    : day_status,
            "sessions"  : sessions,
        })

    return result
    # ── Alertes patient ───────────────────────────────
@router.get("/alerts/{patient_id}")
async def get_alerts(patient_id: str):
    from app.database import db

    # Récupérer directement les FA depuis hrv_windows
    cursor = db.hrv_windows.find({
        "patient_id" : patient_id,
        "label"      : 1  # 1 = FA détectée
    }).sort("timestamp", -1)

    windows = await cursor.to_list(length=100)

    result = []
    for w in windows:
        # Chercher message médecin si existe
        message = await db.messages_medecin.find_one({
            "window_id": str(w["_id"])
        })

        result.append({
            "id"               : str(w["_id"]),
            "bpm"              : w.get("mean_bpm", 0),
            "minute"           : w.get("minute", 0),
            "timestamp"        : w["timestamp"].strftime(
                                   "%d/%m/%Y à %H:%M"
                                 ) if "timestamp" in w else "",
            "message_medecin"  : message["contenu"]
                                 if message else None,
            "timestamp_message": message["timestamp"].strftime(
                                   "%d/%m/%Y à %H:%M"
                                 ) if message else None,
        })

    return result