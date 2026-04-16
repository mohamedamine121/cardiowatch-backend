from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import auth
from app.routes import session

# ── Créer l'application FastAPI ───────────────────
app = FastAPI(
    title       = "CardioWatch API",
    description = "Backend pour l'application CardioWatch",
    version     = "1.0.0"
)

# ── CORS (autoriser Flutter à communiquer) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Inclure les routes ────────────────────────────
app.include_router(
    auth.router,
    prefix = "/api/auth",
    tags   = ["Authentification"]
)
app.include_router(
    session.router,
    prefix = "/api/session",
    tags   = ["Session PPG"]
)

# ── Route de test ─────────────────────────────────
@app.get("/")
async def root():
    return {
        "message" : "CardioWatch API fonctionne !",
        "version" : "1.0.0"
    }