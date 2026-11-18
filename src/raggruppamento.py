import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans

#percorsi
cartella_progetto = Path(__file__).resolve().parents[1]
percorso_csv = cartella_progetto / "data" / "clienti_normalizzati.csv"
cartella_risultati = cartella_progetto / "results"
cartella_risultati.mkdir(exist_ok=True)
percorso_grafico = cartella_risultati / "gomito.png"
df = pd.read_csv(percorso_csv)

#gomito
valori_g = range(2,9)
inerzie = []

for g in valori_g:
    modello = KMeans(
        n_clusters=g,
        random_state=2004,
        n_init=10
    )

    modello.fit(df)
    inerzie.append(modello.inertia_)
    print(f"cluster: {g} - Inerzia: {modello.inertia_:2f}")

#grafico
plt.figure(figsize=(8,5))
plt.plot(valori_g, inerzie, marker="o")
plt.xlabel("metodo del gomito")
plt.ylabel("Inerzia")
plt.xticks(list(valori_g))
plt.grid(True)

plt.savefig(percorso_grafico, dpi=300, bbox_inches="tight")
plt.show