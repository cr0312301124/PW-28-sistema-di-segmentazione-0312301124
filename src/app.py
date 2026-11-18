import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from config import clusteredcsv,kmeanspkl,scalerpkl,gomitopng
from generatore_dati import generatore
from preelaborazione import prelab
from raggruppamento import gomito,silhouette,elab_finale


st.set_page_config(
    page_title="Project Work", 
)


tab_A, tab_B = st.tabs(["a","b"])
with tab_A:
    def a():
        df = pd.read_csv(clusteredcsv)
        scaler = joblib.load(scalerpkl)
        modello = joblib.load(kmeanspkl)
        st.header("Profili dei gruppi individuati")

        colonne = [
            "richieste_ultimi_90_giorni",
            "importo_medio_eur",
            "importo_totale_eur",
            "giorni_dall_ultima_richiesta",
            "giorni_dall_iscrizione"
        ]

        tipologia_cliente = {
            0: "Recenti",
            1: "Molto attivi",
            2: "occasionali/inattivi"
        }

        numero_clienti = df.groupby("cluster")["id"].nunique()
        profili = df.groupby("cluster")[colonne].mean()
        profili.insert(0,"tipologia",tipologia_cliente)
        profili.insert(0,"numero_clienti", numero_clienti)
        st.dataframe(profili.round(2), use_container_width=True)

        st.header("Grafico a dispersione")
        tab_statico, tab_interattivo = st.tabs(["statico","interattivo"])
        with tab_statico:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(
                data=df,
                x="richieste_ultimi_90_giorni",
                y="importo_medio_eur",
                hue="cluster",
                palette="viridis",
                s=70,
                ax=ax
            )


            ax.set_title("Segmentazione dei clienti")
            ax.set_xlabel("Richieste negli ultimi 90 giorni")
            ax.set_ylabel("Importo medio speso (€)")
            ax.legend(title="Cluster")
            st.pyplot(fig)
            plt.close(fig)

        with tab_interattivo:
            st.scatter_chart(
                df,
                x="richieste_ultimi_90_giorni",
                y="importo_medio_eur",
                color="cluster"
            )
        return
    a()

with tab_B:
    st.header("Genera")
    Seed = st.number_input("seed", value=34296)
    Random_state = st.number_input("random_state", value=2004)
    Clienti = st.number_input("clienti", value=600)

    if st.button("test"):
        generatore(Seed,Clienti)
        print("generazione completata")
        prelab()
        print("prelaborazione eseguita")
        x = gomito(Random_state)
        print("gomito generato")
        y, num_cluster = silhouette(Random_state, x)
        print("silhouette completata")
        elab_finale(Random_state, y, num_cluster)
        print("elaborato finale")
        st.image(gomitopng)
        a()