from pydantic import BaseModel, EmailStr
from typing import Optional

# ── Modèle Inscription Médecin ────────────────────
class MedecinRegister(BaseModel):
    nom        : str
    email      : str
    password   : str
    specialite : str
    telephone  : str

# ── Modèle Inscription Patient ────────────────────
class PatientRegister(BaseModel):
    nom        : str
    email      : str
    password   : str
    age        : int
    medecin_id : str

# ── Modèle Login ──────────────────────────────────
class LoginData(BaseModel):
    email      : str
    password   : str
    role       : str
    medecin_id : Optional[str] = None

# ── Modèle Réponse Token ──────────────────────────
class TokenResponse(BaseModel):
    access_token         : str
    role                 : str
    nom                  : str
    email                : str
    medecinNom           : Optional[str]  = None
    medecinDisponibilite : Optional[dict] = None
    age                  : Optional[int]  = None
    patient_id           : Optional[str]  = None
    telephone            : Optional[str]  = None
    groupe_sanguin       : Optional[str]  = None
    poids                : Optional[str]  = None
    taille               : Optional[str]  = None
    identifiant          : Optional[str]  = None
    medecin_id           : Optional[str]  = None
    photo_url            : Optional[str]  = None  # ✅ AJOUTER