from pathlib import Path

cartella_progetto = Path(__file__).resolve().parents[1]
cartella_data = cartella_progetto / "data"
cartella_pkl = cartella_progetto / "pkl"
cartella_results = cartella_progetto / "results"

cartella_data.mkdir(exist_ok=True)
cartella_pkl.mkdir(exist_ok=True)
cartella_results.mkdir(exist_ok=True)

clienticsv = cartella_data / "clienti.csv" 
clienti_normalizzaticsv = cartella_data / "clienti_normalizzati.csv" 
clusteredcsv = cartella_data / "clustered.csv" 

scalerpkl = cartella_pkl / "scaler.pkl"
kmeanspkl = cartella_pkl / "kmeans.pkl"

gomitopng = cartella_data / "gomito.png"

