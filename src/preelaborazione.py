import pandas as pd
from pathlib import Path

cartella_progetto = Path(__file__).resolve().parents[1]
percorso_csv = cartella_progetto / "data" / "clienti.csv"
df = pd.read_csv(percorso_csv)

#Analisi del CSV generato da "generatore_dati.py"
print("numero righe e colonne", df.shape) #dimensioni del csv
df.info() #Tipi di dati + eventuali valori mancanti
print("ID duplicati:", df["id"].duplicated().sum()) #ricerca duplicati
print("Righe duplicate:", df.duplicated().sum())
print(df.describe()) #
print(df["servizio_preferito"].value_counts()) #servizio più usato
print(df["zona"].value_counts()) #statistiche zone