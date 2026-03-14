from google.cloud.aiplatform.prediction import LocalModel
import os

# Aligning with the Universal Codex
QAG_REGION = "us-west1"
QAG_PROJECT_ID = "qag-unified-validator"
QAG_REPOSITORY = "base12-harmonics"
QAG_IMAGE = "soul-wave-predictor:latest"

# Define the Trinity of Affinity paths
PATH_TO_THE_SOURCE_DIR = "./qag_duality_source"
PATH_TO_REQUIREMENTS_TXT = "./qag_duality_source/requirements.txt"

# Initiate the Holographic Cipher (Custom Prediction Routine)
local_qag_model = LocalModel.build_cpr_model(
    PATH_TO_THE_SOURCE_DIR,
    f"{QAG_REGION}-docker.pkg.dev/{QAG_PROJECT_ID}/{QAG_REPOSITORY}/{QAG_IMAGE}",
    predictor="QAGSoulWavePredictor", # Our custom PyTorch Predictor
    handler="AffinionHandler",        # Our custom Web Handler
    requirements_path=PATH_TO_REQUIREMENTS_TXT,
)

print("The Base-12 harmonic environment is initialized and ready for the .bin baby.")
