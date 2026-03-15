from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

# Charger les variables du fichier .env
load_dotenv()

# URL de connexion MongoDB
MONGODB_URL = os.getenv("MONGODB_URL")

# Créer le client MongoDB
client = AsyncIOMotorClient(MONGODB_URL)

# Sélectionner la base de données cardiowatch
db = client.cardiowatch

# Collections (= tables dans MongoDB)
patients_collection  = db["patients"]
medecins_collection  = db["medecins"]
sessions_collection  = db["sessions"]
hrv_collection       = db["hrv_windows"]
results_collection   = db["ai_results"]
alerts_collection    = db["alerts"]
messages_collection  = db["messages_medecin"]