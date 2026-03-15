from fastapi import APIRouter, HTTPException
from app.models.user import MedecinRegister, PatientRegister, LoginData, TokenResponse
from app.database import medecins_collection, patients_collection
from app.services.auth_service import hash_password, verify_password, create_jwt, generate_medecin_id
from datetime import datetime

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
            "medecinNom"   : medecin["nom"] 
        }

    else:
        raise HTTPException(status_code=400, detail="Rôle invalide")