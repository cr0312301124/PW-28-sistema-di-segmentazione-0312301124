import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans

#percorsi
cartella_progetto = Path(__file__).resolve().parents[1]
percorso_csv = cartella_progetto / "data" / "clienti_normalizzati.csv"
cartella_risultati = cartella_progetto / "results"
cartella_risultati.mkdir(exist_ok=True)
df = pd.read_csv(percorso_csv)
