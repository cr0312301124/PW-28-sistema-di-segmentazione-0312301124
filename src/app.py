import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from contextlib import redirect_stdout
from datetime import datetime
from config import clusteredcsv,kmeanspkl,scalerpkl,gomitopng,cartella_logs
from generatore_dati import generatore
from preelaborazione import prelab
from raggruppamento import gomito,silhouette,elab_finale

nome_file = cartella_logs / f"report_esecuzione_{datetime.now():%Y%m%d_%H%M%S}.txt"


st.set_page_config(
    page_title="Project Work",
    layout="wide" 
)

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

    numero_clienti = df.groupby("cluster")["id"].nunique()
    profili = df.groupby("cluster")[colonne].mean()
    profili.insert(0,"numero_clienti", numero_clienti)
    st.dataframe(profili.round(2), use_container_width=True)

    st.header("Grafici a dispersione")
    tab_interattivo, tab_statico = st.tabs(["statico","interattivo"])    
#PRIMO GRAFICO
    with tab_interattivo:
        grafico = alt.Chart(df).mark_circle(size=70).encode(
            x=alt.X(
                "giorni_dall_ultima_richiesta:Q",
                title="Giorni dall'ultima richiesta"
            ),
            y=alt.Y(
                "importo_medio_eur:Q",
                title="Importo medio (€)"
            ),
            color=alt.Color(
                "cluster:N",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title="Cluster")
            ),
            tooltip=[
                "cluster",
                "giorni_dall_ultima_richiesta",
                "importo_medio_eur"
            ]
        ).properties(
            height=400
        ).interactive()

        st.altair_chart(grafico, use_container_width=True)

        with tab_statico:
                fig, ax = plt.subplots(figsize=(18,5))
                sns.scatterplot(
                    data=df,
                    x="giorni_dall_ultima_richiesta",
                    y="importo_medio_eur",
                    hue="cluster",
                    palette="viridis",
                    s=70,
                    ax=ax
                )
        
        
                ax.set_title("Segmentazione dei clienti")
                ax.set_xlabel("giorni_dall_ultima_richiesta")
                ax.set_ylabel("importo_medio_eur")
                ax.legend(title="Cluster")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
    return

#navbar
col1, col2, col3 = st.columns([3, 2, 1])
with col1:
    st.write("test1")
with col2:
        st.write("test2")
with col3:
    mostra_editor = st.checkbox("Debug Mode")

if mostra_editor:
    st.header("Debug Mode")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        Seed = st.number_input("Seed", value=34296)
    with col2:
        Random_state = st.number_input("random_state", value=2004)
    with col3:
        Clienti = st.number_input("Numero clienti", value=600, min_value=300, max_value=600)
    
    
    

    if st.button("Genera"):
        with open(nome_file, "w", encoding="utf-8") as file:
            with redirect_stdout(file):
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
                #st.image(gomitopng)
                a()
else:
     a()