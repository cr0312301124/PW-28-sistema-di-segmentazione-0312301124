import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#percorsi
cartella_progetto = Path(__file__).resolve().parents[1]
percorso_csv_nominale = cartella_progetto / "data" / "clienti_normalizzati.csv"
percorso_csv_originale = cartella_progetto / "data" / "clienti.csv"
percorso_kmeans = cartella_progetto / "pkl" / "kmeans.pkl"
cartella_risultati = cartella_progetto / "results"
cartella_risultati.mkdir(exist_ok=True)
percorso_grafico = cartella_risultati / "gomito.png"
percorso_cluster = cartella_risultati / "cluster3.csv"
df = pd.read_csv(percorso_csv_nominale)
df_originale = pd.read_csv(percorso_csv_originale)

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

#silhoouette
valori_cluster = [3, 4, 5]
for g in valori_cluster:
    modello = KMeans(
        n_clusters=g,
        random_state=2004,
        n_init=10
    )
        
    etichette = modello.fit_predict(df)
    silhouette = silhouette_score(df,etichette)
    print(f"cluster: {g} - Inerzia: {modello.inertia_:2f} - SIlhouette: {silhouette:4f}")

#Cluster 3 è migliore (SEED:34296 | Random_state: 2004)
cluster_c = KMeans(
    n_clusters=3,
    random_state=2004,
    n_init=10
)

etichette_finali = cluster_c.fit_predict(df)
df_originale["cluster"] = etichette_finali
df_originale.to_csv(percorso_cluster, index=False)
print(df_originale["cluster"].value_counts().sort_index())
print(df_originale.groupby("cluster").mean(numeric_only=True).round(2))
print(df_originale.groupby("cluster").mean(numeric_only=True).round(2).T)

#analisi servizi e zone

servizi_cluster = pd.crosstab(
    df_originale["cluster"],
    df_originale["servizio_preferito"],
    normalize="index"
) * 100

zone_cluster = pd.crosstab(
    df_originale["cluster"],
    df_originale["zona"],
    normalize="index"
) * 100

print("servizi:\n", servizi_cluster.round(2) )
print("zone:\n", zone_cluster.round(2) )

#joblib
joblib.dump(cluster_c, percorso_kmeans)
