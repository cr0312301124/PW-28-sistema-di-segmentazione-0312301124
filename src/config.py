from pathlib import Path
import pandas as pd

#Variabili
NUM_CLIENTI = 600
SEED = 324296 #34296
RANDOM_STATE = 2004
DATA_DI_RIFERIMENTO = pd.Timestamp("2026-11-16")

#Percorsi

cartella_progetto = Path(__file__).resolve().parents[1]
cartella_data = cartella_progetto / "data"
cartella_pkl = cartella_progetto / "pkl"
cartella_results = cartella_progetto / "results"
cartella_logs = cartella_progetto / "logs"

cartella_data.mkdir(exist_ok=True)
cartella_pkl.mkdir(exist_ok=True)
cartella_results.mkdir(exist_ok=True)
cartella_logs.mkdir(exist_ok=True)

clienticsv = cartella_data / "clienti.csv" 
clienti_normalizzaticsv = cartella_data / "clienti_normalizzati.csv" 
clusteredcsv = cartella_data / "clustered.csv" 

scalerpkl = cartella_pkl / "scaler.pkl"
kmeanspkl = cartella_pkl / "kmeans.pkl"

gomitopng = cartella_data / "gomito.png"

