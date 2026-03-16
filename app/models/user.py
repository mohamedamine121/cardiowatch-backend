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
    medecin_id : str  # identifiant du médecin ex: MED-2024-001

# ── Modèle Login ──────────────────────────────────
class LoginData(BaseModel):
    email    : str
    password : str
    role     : str  # "patient" ou "medecin"
    medecin_id : Optional[str] = None  # seulement pour patient

# ── Modèle Réponse Token ──────────────────────────
class TokenResponse(BaseModel):
    access_token : str
    role         : str
    nom          : str
    email        : str
    medecinNom   : Optional[str] = None