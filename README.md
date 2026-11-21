# PW-28-sistema-di-segmentazione

Progetto di machine learning per la segmentazione automatica della clientela di un'azienda di servizi, utilizzando l'algoritmo K-Means


## Esecuzione Online
È possibile usufruire di questo link per utilizzare la dashboard online:


## Esecuzione in locale

```bash
# Scarica il progetto
cd PW-28---sistema-di-segmentazione

# Crea ambiente virtuale (non obbligatorio)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# oppure
venv\Scripts\activate             # Windows

# Installa le dipendenze
pip install streamlit pandas scikit-learn joblib seaborn matplotlib altair numpy

# Come eseguire l'intera pipeline (generazione → clustering)
python src/main.py

# Come avviare la dashboard
streamlit run src/app.py
```






## Parametri configurabili (`config.py`)

```python
NUM_CLIENTI = 600
SEED = 34296
RANDOM_STATE = 2004
DATA_DI_RIFERIMENTO = pd.Timestamp("2026-11-16")
```
SEED e RANDOM_STATE permettono di generare all'esecuzione gli stessi dati, se si vuole generare nuovi dati randomici, è necessario modificare almeno una delle due variabii


## Struttura del progetto

```
├── src/
│   ├──app.py                   # Dashboard Streamlit
│   ├── main.py                 # Pipeline completa (generazione → clustering)
│   ├── config.py               # Percorsi e parametri globali
│   ├── generatore_dati.py      # Generazione dataset sintetico
│   ├── preelaborazione.py      # Pulizia + StandardScaler
│   └──raggruppamento.py        # gomito, Silhouette, clustering finale
├── data/
│   ├── clienti.csv
│   ├── clienti_normalizzati.csv
│   ├── clustered.csv
│   └── gomito.png
├── pkl/
│   ├── scaler.pkl
│   └── kmeans.pkl
└── logs/
    └── report_esecuzione_*.txt
```
