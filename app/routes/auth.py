from fastapi import APIRouter, HTTPException
from app.models.user import MedecinRegister, PatientRegister, LoginData, TokenResponse
from app.database import medecins_collection, patients_collection
from app.services.auth_service import hash_password, verify_password, create_jwt, generate_medecin_id
from datetime import datetime, timedelta
from fastapi.responses import HTMLResponse
import secrets
import os

router = APIRouter()

# ── Stockage temporaire tokens reset ─────────────
reset_tokens: dict = {}

# ── Config email ──────────────────────────────────
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

conf = ConnectionConfig(
    MAIL_USERNAME   = os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD   = os.getenv("MAIL_PASSWORD"),
    MAIL_FROM       = os.getenv("MAIL_FROM"),
    MAIL_PORT       = 587,
    MAIL_SERVER     = "smtp.gmail.com",
    MAIL_STARTTLS   = True,
    MAIL_SSL_TLS    = False,
    USE_CREDENTIALS = True,
)

# ── Inscription Médecin ───────────────────────────
@router.post("/register/medecin")
async def register_medecin(data: MedecinRegister):
    existing = await medecins_collection.find_one(
        {"email": data.email}
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email déjà utilisé"
        )

    medecin_id = generate_medecin_id()

    medecin = {
        "identifiant": medecin_id,
        "nom"        : data.nom,
        "email"      : data.email,
        "password"   : hash_password(data.password),
        "specialite" : data.specialite,
        "telephone"  : data.telephone,
        "created_at" : datetime.utcnow()
    }

    await medecins_collection.insert_one(medecin)

    # ── Envoyer identifiant par email ─────────────
    try:
        message = MessageSchema(
            subject    = "CardioWatch — "
                         "Votre identifiant médecin",
            recipients = [data.email],
            body       = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;
             background:#f5f5f5;padding:20px">
  <div style="max-width:500px;margin:0 auto;
              background:white;border-radius:16px;
              padding:32px">
    <h2 style="color:#1A73E8;text-align:center">
      ❤️ CardioWatch
    </h2>
    <p style="color:#333">
      Bonjour Dr. {data.nom},
    </p>
    <p style="color:#333">
      Votre compte médecin a été créé avec succès !
      Voici votre identifiant unique à partager
      avec vos patients :
    </p>
    <div style="text-align:center;margin:32px 0">
      <div style="background:#f0f4ff;
                  border:2px solid #1A73E8;
                  border-radius:12px;
                  padding:20px;
                  display:inline-block">
        <p style="color:#999;font-size:13px;
                  margin:0 0 8px">
          Identifiant Médecin
        </p>
        <p style="color:#1A73E8;font-size:28px;
                  font-weight:bold;
                  letter-spacing:2px;margin:0">
          {medecin_id}
        </p>
      </div>
    </div>
    <p style="color:#333">
      Partagez cet identifiant avec vos patients
      pour qu'ils puissent s'inscrire sur
      CardioWatch et être associés à votre compte.
    </p>
    <div style="background:#fff3cd;
                border-radius:8px;
                padding:12px 16px;
                margin:16px 0">
      <p style="color:#856404;font-size:13px;
                margin:0">
        ⚠️ Gardez cet identifiant en lieu sûr.
        Il est unique et lié à votre compte.
      </p>
    </div>
    <hr style="border:none;border-top:1px solid #eee;
               margin:24px 0">
    <p style="color:#ccc;font-size:11px;
              text-align:center">
      CardioWatch — Système de surveillance cardiaque
    </p>
  </div>
</body>
</html>
            """,
            subtype="html",
        )
        fm = FastMail(conf)
        await fm.send_message(message)
        print(f"✅ Email identifiant envoyé à {data.email}")
    except Exception as e:
        # Ne pas bloquer l'inscription si email échoue
        print(f"⚠️ Erreur email identifiant : {e}")

    return {
        "message"    : "Médecin inscrit avec succès",
        "identifiant": medecin_id,
        "nom"        : data.nom,
        "email"      : data.email
    }


# ── Inscription Patient ───────────────────────────
@router.post("/register/patient")
async def register_patient(data: PatientRegister):
    existing = await patients_collection.find_one(
        {"email": data.email}
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email déjà utilisé"
        )

    medecin = await medecins_collection.find_one(
        {"identifiant": data.medecin_id}
    )
    if not medecin:
        raise HTTPException(
            status_code=404,
            detail="Identifiant médecin invalide"
        )

    patient = {
        "nom"                : data.nom,
        "email"              : data.email,
        "password"           : hash_password(data.password),
        "age"                : data.age,
        "medecin_id"         : str(medecin["_id"]),
        "medecin_identifiant": data.medecin_id,
        "created_at"         : datetime.utcnow()
    }

    await patients_collection.insert_one(patient)

    return {
        "message": "Patient inscrit avec succès",
        "nom"    : data.nom,
        "email"  : data.email,
        "medecin": medecin["nom"]
    }


# ── Login ─────────────────────────────────────────
@router.post("/login", response_model=TokenResponse)
async def login(data: LoginData):

    if data.role == "medecin":
        medecin = await medecins_collection.find_one(
            {"email": data.email}
        )
        if not medecin:
            raise HTTPException(
                status_code=404,
                detail="Médecin introuvable"
            )
        if not verify_password(
                data.password, medecin["password"]):
            raise HTTPException(
                status_code=401,
                detail="Mot de passe incorrect"
            )

        token = create_jwt({
            "medecin_id" : str(medecin["_id"]),
            "identifiant": medecin["identifiant"],
            "role"       : "medecin"
        })

        return {
            "access_token": token,
            "role"        : "medecin",
            "nom"         : medecin["nom"],
            "email"       : medecin["email"],
            "identifiant" : medecin["identifiant"],
            "medecin_id"  : str(medecin["_id"])
        }

    elif data.role == "patient":
        if not data.medecin_id:
            raise HTTPException(
                status_code=400,
                detail="Identifiant médecin requis"
            )

        medecin = await medecins_collection.find_one(
            {"identifiant": data.medecin_id}
        )
        if not medecin:
            raise HTTPException(
                status_code=404,
                detail="Identifiant médecin invalide"
            )

        patient = await patients_collection.find_one(
            {"email": data.email}
        )
        if not patient:
            raise HTTPException(
                status_code=404,
                detail="Patient introuvable"
            )

        if patient["medecin_identifiant"] != \
                data.medecin_id:
            raise HTTPException(
                status_code=403,
                detail="Ce patient n'appartient "
                       "pas à ce médecin"
            )

        if not verify_password(
                data.password, patient["password"]):
            raise HTTPException(
                status_code=401,
                detail="Mot de passe incorrect"
            )

        token = create_jwt({
            "patient_id": str(patient["_id"]),
            "medecin_id": str(medecin["_id"]),
            "role"      : "patient"
        })

        return {
            "access_token"        : token,
            "role"                : "patient",
            "nom"                 : patient["nom"],
            "email"               : patient["email"],
            "medecinNom"          : medecin["nom"],
            "medecinDisponibilite": dict(
                medecin.get("disponibilite", {})),
            "age"                 : patient["age"],
            "patient_id"          : str(patient["_id"]),
            "telephone"           : patient.get(
                "telephone", ""),
            "groupe_sanguin"      : patient.get(
                "groupe_sanguin", ""),
            "poids"               : patient.get(
                "poids", ""),
            "taille"              : patient.get(
                "taille", "")
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="Rôle invalide"
        )


# ── Mise à jour profil patient ────────────────────
@router.put("/update/patient/{patient_id}")
async def update_patient(
    patient_id: str,
    data      : dict
):
    from bson import ObjectId

    update_data = {}
    if "telephone"      in data:
        update_data["telephone"]      = data["telephone"]
    if "groupe_sanguin" in data:
        update_data["groupe_sanguin"] = data["groupe_sanguin"]
    if "poids"          in data:
        update_data["poids"]          = data["poids"]
    if "taille"         in data:
        update_data["taille"]         = data["taille"]

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


# ── Mise à jour disponibilité médecin ────────────
@router.put("/update/medecin/disponibilite/{medecin_id}")
async def update_disponibilite(
    medecin_id: str,
    data      : dict
):
    try:
        from bson import ObjectId

        disponibilite = data.get("disponibilite", {})

        await medecins_collection.update_one(
            {"_id": ObjectId(medecin_id)},
            {"$set": {"disponibilite": disponibilite}}
        )
        return {
            "success"      : True,
            "message"      : "Disponibilité mise à jour",
            "disponibilite": disponibilite
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


# ── Historique 7 jours ────────────────────────────
@router.get("/history/{patient_id}")
async def get_history(patient_id: str):
    from bson import ObjectId
    from app.database import db

    today    = datetime.utcnow().replace(
        hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=6)

    cursor = db.hrv_windows.find({
        "patient_id": patient_id,
        "timestamp" : {"$gte": week_ago}
    }).sort("timestamp", 1)

    windows = await cursor.to_list(length=1000)

    days_data = {}
    for w in windows:
        day_key = w["timestamp"].strftime("%Y-%m-%d")
        if day_key not in days_data:
            days_data[day_key] = []

        days_data[day_key].append({
            "minute"   : w.get("minute",   0),
            "timestamp": w["timestamp"].strftime(
                "%H:%M"),
            "bpm"      : w.get("mean_bpm", 0),
            "spo2"     : w.get("spo2",     0),
            "label"    : w.get("label",    0),
            "status"   : w.get("status",   "Normal"),
            "sdnn"     : w.get("sdnn",     0),
            "rmssd"    : w.get("rmssd",    0),
            "pnn50"    : w.get("pnn50",    0),
            "entropy"  : w.get("entropy",  0),
        })

    result = []
    for i in range(7):
        day      = week_ago + timedelta(days=i)
        day_key  = day.strftime("%Y-%m-%d")
        sessions = days_data.get(day_key, [])

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

    cursor = db.hrv_windows.find({
        "patient_id": patient_id,
        "label"     : 1
    }).sort("timestamp", -1)

    windows = await cursor.to_list(length=100)

    result = []
    for w in windows:
        message = await db.messages_medecin.find_one({
            "window_id": str(w["_id"])
        })

        result.append({
            "id"               : str(w["_id"]),
            "bpm"              : w.get("mean_bpm", 0),
            "minute"           : w.get("minute",   0),
            "timestamp"        : w["timestamp"].strftime(
                "%d/%m/%Y à %H:%M"
            ) if "timestamp" in w else "",
            "message_medecin"  : message["contenu"]
                if message else None,
            "timestamp_message": message["timestamp"]
                .strftime("%d/%m/%Y à %H:%M")
                if message else None,
        })

    return result


# ── Liste patients du médecin ─────────────────────
@router.get("/medecin/patients/{medecin_id}")
async def get_patients(medecin_id: str):
    from app.database import db

    medecin = await medecins_collection.find_one(
        {"identifiant": medecin_id}
    )
    if not medecin:
        raise HTTPException(
            status_code=404,
            detail="Médecin introuvable"
        )

    cursor = patients_collection.find({
        "medecin_id": str(medecin["_id"])
    })
    patients = await cursor.to_list(length=100)

    result = []
    for p in patients:
        last_window = await db.hrv_windows.find_one(
            {"patient_id": str(p["_id"])},
            sort=[("timestamp", -1)]
        )

        today    = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        fa_today = await db.hrv_windows.find_one({
            "patient_id": str(p["_id"]),
            "label"     : 1,
            "timestamp" : {"$gte": today}
        })

        result.append({
            "id"            : str(p["_id"]),
            "nom"           : p.get("nom",           ""),
            "age"           : p.get("age",            0),
            "email"         : p.get("email",          ""),
            "groupe_sanguin": p.get("groupe_sanguin", ""),
            "poids"         : p.get("poids",          ""),
            "taille"        : p.get("taille",         ""),
            "has_fa_today"  : fa_today is not None,
            "last_bpm"      : last_window.get(
                "mean_bpm", 0) if last_window else None,
            "last_spo2"     : last_window.get(
                "spo2", 0) if last_window else None,
            "last_session"  : last_window["timestamp"]
                .strftime("%d/%m/%Y à %H:%M")
                if last_window else None,
        })

    return result


# ── Alertes FA tous patients du médecin ───────────
@router.get("/medecin/alerts/{medecin_id}")
async def get_medecin_alerts(medecin_id: str):
    from app.database import db

    medecin = await medecins_collection.find_one(
        {"identifiant": medecin_id}
    )
    if not medecin:
        raise HTTPException(
            status_code=404,
            detail="Médecin introuvable"
        )

    cursor = patients_collection.find({
        "medecin_id": str(medecin["_id"])
    })
    patients    = await cursor.to_list(length=100)
    patient_ids = [str(p["_id"]) for p in patients]
    patient_map = {
        str(p["_id"]): p["nom"] for p in patients
    }

    cursor = db.hrv_windows.find({
        "patient_id": {"$in": patient_ids},
        "label"     : 1
    }).sort("timestamp", -1)
    windows = await cursor.to_list(length=500)

    result = []
    for w in windows:
        message = await db.messages_medecin.find_one({
            "window_id": str(w["_id"])
        })
        result.append({
            "id"            : str(w["_id"]),
            "patient_id"    : w["patient_id"],
            "patient_nom"   : patient_map.get(
                w["patient_id"], "Inconnu"),
            "bpm"           : w.get("mean_bpm", 0),
            "minute"        : w.get("minute",   0),
            "timestamp"     : w["timestamp"].strftime(
                "%d/%m/%Y à %H:%M"
            ) if "timestamp" in w else "",
            "traitee"       : message is not None,
            "message_envoye": message["contenu"]
                if message else None,
        })

    return result


# ── Envoyer message au patient ────────────────────
@router.post("/medecin/message")
async def send_message(data: dict):
    from app.database import db

    message = {
        "window_id" : data.get("window_id"),
        "patient_id": data.get("patient_id"),
        "medecin_id": data.get("medecin_id"),
        "contenu"   : data.get("contenu"),
        "couleur"   : data.get("couleur", "vert"),
        "timestamp" : datetime.utcnow(),
        "lu_patient": False
    }
    await db.messages_medecin.insert_one(message)
    return {"message": "Message envoyé avec succès"}


# ── DEBUG endpoint ────────────────────────────────
@router.get("/debug/login/test")
async def debug_login():
    medecin = await medecins_collection.find_one(
        {"identifiant": "MED-2026-4IWRX5"}
    )
    return {
        "disponibilite": dict(
            medecin.get("disponibilite", {})),
        "has_dispo"    : "disponibilite" in medecin
    }


# ── Profil médecin ────────────────────────────────
@router.get("/medecin/profil/{identifiant}")
async def get_medecin_profil(identifiant: str):
    medecin = await medecins_collection.find_one(
        {"identifiant": identifiant}
    )
    if not medecin:
        raise HTTPException(
            status_code=404,
            detail="Médecin introuvable"
        )
    return {
        "success"      : True,
        "id"           : str(medecin["_id"]),
        "nom"          : medecin["nom"],
        "email"        : medecin["email"],
        "specialite"   : medecin["specialite"],
        "telephone"    : medecin["telephone"],
        "identifiant"  : medecin["identifiant"],
        "disponibilite": medecin.get(
            "disponibilite", {}),
    }

# ── Sauvegarder photo profil patient ─────────────
@router.put("/update/patient/{patient_id}/photo")
async def update_patient_photo(
    patient_id: str,
    data      : dict
):
    from bson import ObjectId

    photo_url = data.get("photo_url", "")

    if not photo_url:
        raise HTTPException(
            status_code=400,
            detail="URL photo requise"
        )

    await patients_collection.update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": {"photo_url": photo_url}}
    )

    return {
        "success"  : True,
        "message"  : "Photo mise à jour",
        "photo_url": photo_url
    }
# ── Mot de passe oublié ───────────────────────────
@router.post("/forgot-password")
async def forgot_password(data: dict):
    email = data.get("email", "").strip().lower()

    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email requis"
        )

    patient = await patients_collection.find_one(
        {"email": email}
    )
    medecin = await medecins_collection.find_one(
        {"email": email}
    )

    if not patient and not medecin:
        return {
            "success": True,
            "message": "Si cet email existe, "
                       "un lien a été envoyé."
        }

    token     = secrets.token_urlsafe(32)
    expire_at = datetime.utcnow() + timedelta(hours=1)
    role      = "patient" if patient else "medecin"

    reset_tokens[token] = {
        "email"    : email,
        "role"     : role,
        "expire_at": expire_at,
    }

    reset_link = (
        "https://cardiowatch-backend.onrender.com"
        f"/api/auth/reset-password-page?token={token}"
    )

    try:
        message = MessageSchema(
            subject    = "CardioWatch — Réinitialisation"
                         " de votre mot de passe",
            recipients = [email],
            body       = f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;
             background:#f5f5f5;padding:20px">
  <div style="max-width:500px;margin:0 auto;
              background:white;border-radius:16px;
              padding:32px">
    <h2 style="color:#1A73E8;text-align:center">
      ❤️ CardioWatch
    </h2>
    <p style="color:#333">Bonjour,</p>
    <p style="color:#333">
      Vous avez demandé la réinitialisation de votre
      mot de passe. Cliquez sur le bouton ci-dessous :
    </p>
    <div style="text-align:center;margin:32px 0">
      <a href="{reset_link}"
         style="background:#1A73E8;color:white;
                padding:14px 32px;border-radius:8px;
                text-decoration:none;font-weight:bold">
        Réinitialiser mon mot de passe
      </a>
    </div>
    <p style="color:#999;font-size:12px">
      Ce lien expire dans <strong>1 heure</strong>.
      Si vous n'avez pas fait cette demande,
      ignorez cet email.
    </p>
  </div>
</body>
</html>
            """,
            subtype="html",
        )
        fm = FastMail(conf)
        await fm.send_message(message)
    except Exception as e:
        print(f"Erreur email : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur envoi email : {str(e)}"
        )

    return {
        "success": True,
        "message": "Si cet email existe, "
                   "un lien a été envoyé."
    }


# ── Page HTML reset ───────────────────────────────
@router.get("/reset-password-page")
async def reset_password_page(token: str):
    return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,
        initial-scale=1.0">
  <title>CardioWatch — Nouveau mot de passe</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:Arial,sans-serif;
         background:#f5f7fa;display:flex;
         align-items:center;justify-content:center;
         min-height:100vh;padding:20px}}
    .card{{background:white;border-radius:16px;
           padding:32px;width:100%;max-width:400px;
           box-shadow:0 4px 20px rgba(0,0,0,0.08)}}
    h2{{color:#1A73E8;text-align:center;
        margin-bottom:8px}}
    .sub{{color:#999;font-size:14px;
          text-align:center;margin-bottom:24px}}
    label{{display:block;font-size:14px;color:#555;
           margin-bottom:6px;font-weight:500}}
    input{{width:100%;padding:12px 16px;
           border:1.5px solid #ddd;
           border-radius:8px;font-size:15px;
           margin-bottom:16px;outline:none}}
    input:focus{{border-color:#1A73E8}}
    button{{width:100%;padding:14px;
            background:#1A73E8;color:white;
            border:none;border-radius:8px;
            font-size:16px;font-weight:bold;
            cursor:pointer}}
    button:disabled{{background:#aaa;
                     cursor:not-allowed}}
    .msg{{margin-top:16px;padding:12px 16px;
          border-radius:8px;font-size:14px;
          display:none}}
    .success{{background:#e8f5e9;color:#2e7d32;
              border:1px solid #a5d6a7}}
    .error{{background:#ffebee;color:#c62828;
            border:1px solid #ef9a9a}}
  </style>
</head>
<body>
  <div class="card">
    <h2>❤️ CardioWatch</h2>
    <p class="sub">Nouveau mot de passe</p>
    <form id="form">
      <label>Nouveau mot de passe</label>
      <input type="password" id="pwd1"
             placeholder="Min. 6 caractères"
             required minlength="6">
      <label>Confirmer le mot de passe</label>
      <input type="password" id="pwd2"
             placeholder="Répétez le mot de passe"
             required minlength="6">
      <button type="submit" id="btn">
        Réinitialiser
      </button>
    </form>
    <div class="msg" id="msg"></div>
  </div>
  <script>
    document.getElementById('form')
      .addEventListener('submit', async (e) => {{
        e.preventDefault();
        const p1=document.getElementById('pwd1').value;
        const p2=document.getElementById('pwd2').value;
        const btn=document.getElementById('btn');
        const msg=document.getElementById('msg');
        if(p1!==p2){{
          msg.className='msg error';
          msg.style.display='block';
          msg.textContent=
            'Les mots de passe ne correspondent pas.';
          return;
        }}
        btn.disabled=true;
        btn.textContent='Envoi...';
        try{{
          const res=await fetch(
            '/api/auth/reset-password',
            {{
              method:'POST',
              headers:{{'Content-Type':
                'application/json'}},
              body:JSON.stringify({{
                token:'{token}',password:p1
              }})
            }}
          );
          const data=await res.json();
          if(data.success){{
            msg.className='msg success';
            msg.style.display='block';
            msg.textContent=
              '✅ Mot de passe modifié ! '
              'Retournez à l\'application.';
            document.getElementById('form')
              .style.display='none';
          }}else{{
            msg.className='msg error';
            msg.style.display='block';
            msg.textContent=
              data.detail||'Erreur. Réessayez.';
            btn.disabled=false;
            btn.textContent='Réinitialiser';
          }}
        }}catch(err){{
          msg.className='msg error';
          msg.style.display='block';
          msg.textContent='Erreur réseau.';
          btn.disabled=false;
          btn.textContent='Réinitialiser';
        }}
      }});
  </script>
</body>
</html>
    """)


# ── Traiter reset ─────────────────────────────────
@router.post("/reset-password")
async def reset_password(data: dict):
    token    = data.get("token",    "")
    password = data.get("password", "")

    if not token or not password:
        raise HTTPException(
            status_code=400,
            detail="Token et mot de passe requis"
        )

    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Minimum 6 caractères"
        )

    token_data = reset_tokens.get(token)
    if not token_data:
        raise HTTPException(
            status_code=400,
            detail="Lien invalide ou expiré"
        )

    if datetime.utcnow() > token_data["expire_at"]:
        del reset_tokens[token]
        raise HTTPException(
            status_code=400,
            detail="Lien expiré. "
                   "Faites une nouvelle demande."
        )

    from passlib.context import CryptContext
    pwd_context = CryptContext(
        schemes=["bcrypt"], deprecated="auto"
    )
    hashed = pwd_context.hash(password)

    collection = (
        patients_collection
        if token_data["role"] == "patient"
        else medecins_collection
    )
    await collection.update_one(
        {"email": token_data["email"]},
        {"$set" : {"password": hashed}}
    )

    del reset_tokens[token]

    return {
        "success": True,
        "message": "Mot de passe réinitialisé !"
    }