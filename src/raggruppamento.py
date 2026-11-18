import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from config import clienticsv,clienti_normalizzaticsv,clusteredcsv,kmeanspkl,gomitopng


def gomito(RANDOM_STATE=2004):
    df = pd.read_csv(clienti_normalizzaticsv)
    print(RANDOM_STATE)
    print(df.shape)
    #gomito
    valori_g = range(2,9)
    inerzie = []

    for g in valori_g:
        modello = KMeans(
            n_clusters=g,
            random_state=RANDOM_STATE,
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

    plt.savefig(gomitopng, dpi=300, bbox_inches="tight")
    plt.show
    return df

def silhouette(RANDOM_STATE=2004,df=pd.read_csv(clienti_normalizzaticsv)):
    #silhoouette
    valori_cluster = [3, 4, 5]
    risultati = {}
    for g in valori_cluster:
        modello = KMeans(
            n_clusters=g,
            random_state=RANDOM_STATE,
            n_init=10
        )
            
        etichette = modello.fit_predict(df)
        punteggio = silhouette_score(df,etichette)
        risultati[g] = punteggio
        print(f"cluster: {g} - Inerzia: {modello.inertia_:2f} - SIlhouette: {punteggio:4f}")

        cluster_migliore = max(risultati, key=risultati.get)
        print("cluster migliore:", cluster_migliore)

    return df, cluster_migliore

def elab_finale(RANDOM_STATE=2004,df=pd.read_csv(clienti_normalizzaticsv),nclust = 0):
#Cluster nclust è migliore (SEED:34296 | Random_state: 2004)
    df_originale = pd.read_csv(clienticsv)
    cluster_c = KMeans(
        n_clusters=nclust,
        random_state=RANDOM_STATE,
        n_init=10
    )

    etichette_finali = cluster_c.fit_predict(df)
    df_originale["cluster"] = etichette_finali
    df_originale.to_csv(clusteredcsv, index=False)
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
    joblib.dump(cluster_c, kmeanspkl)
    return