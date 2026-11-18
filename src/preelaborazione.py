import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cartella_progetto = Path(__file__).resolve().parents[1]
percorso_csv = cartella_progetto / "data" / "clienti.csv"
percorso_scaler = cartella_progetto / "pkl" / "scaler.pkl"

df = pd.read_csv(percorso_csv)


#Analisi del CSV generato da "generatore_dati.py"
print("numero righe e colonne", df.shape) #dimensioni del csv
df.info() #Tipi di dati + eventuali valori mancanti
print("ID duplicati:", df["id"].duplicated().sum()) #ricerca duplicati
print("Righe duplicate:", df.duplicated().sum())
print(df.describe()) #
print(df["servizio_preferito"].value_counts()) #servizio più usato
print(df["zona"].value_counts()) #statistiche zone

#pulizia di base
df["data_iscrizione"] = pd.to_datetime(df["data_iscrizione"]) #conversione delle date da stringhe a date riconosciute da pandas
print(df["data_iscrizione"].dtype)

#selezione variabili
record = [
    "richieste_ultimi_90_giorni",
    "importo_medio_eur",
    "giorni_dall_ultima_richiesta",
    "giorni_dall_iscrizione"
]

x = df[record].copy() #matrice
print(x.head())
print(x.shape)

#normalizzazione
x_normalizzato = scaler.fit_transform(x)
x_normalizzato = pd.DataFrame(
    x_normalizzato,
    columns=record,
    index=x.index
)

print(x_normalizzato.head())
print(x_normalizzato.shape)
print(x_normalizzato.describe())

#Esportazione dati normalizzati
percorso_output = cartella_progetto / "data" / "clienti_normalizzati.csv"
x_normalizzato.to_csv(percorso_output, index=False)
print("dati normalizzati salvati in data")

#joblib
joblib.dump(scaler, percorso_scaler)