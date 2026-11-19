import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import streamlit as st
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


def nuovo_cliente(scaler, modello):

    with st.form("nuovo_cliente"):
        richieste = st.number_input("Richieste ultimi 90 giorni", min_value=0)
        importo = st.number_input("Importo medio in euro", min_value=0.0)
        ultima = st.number_input("giorni dall'ultima richiesta", min_value=0)
        iscrizione = st.number_input("giorni dall'iscrizione", min_value=0)
        invia = st.form_submit_button("Assegna")

    if invia:
        dati = pd.DataFrame([[richieste, importo, ultima, iscrizione]],
            columns=[
            "richieste_ultimi_90_giorni",
            "importo_medio_eur",
            "giorni_dall_ultima_richiesta",
            "giorni_dall_iscrizione"
            ]
        )

        cluster = modello.predict(scaler.transform(dati))[0]
        st.success(f"Cliente appartente al cluster {cluster}")

def filtri_clienti(df):

    df["data_iscrizione"] = pd.to_datetime(df["data_iscrizione"])

    col1, col2, col3 = st.columns(3)

    with col1:
        zona = st.selectbox("Zona", ["Tutte"] + sorted(df["zona"].unique()))
    with col2:
        servizio = st.selectbox("Servizio preferito", ["Tutti"] + sorted(df["servizio_preferito"].unique()))
    with col3:
        periodo = st.date_input("Periodo di iscrizione", (
            df["data_iscrizione"].min().date(),
            df["data_iscrizione"].max().date()
        ))

    if zona != "Tutte":
        df = df[df["zona"] == zona]
    if servizio != "Tutti":
        df = df[df["servizio_preferito"] == servizio]
    if isinstance(periodo, tuple) and len(periodo) == 2:
        df = df[df["data_iscrizione"].dt.date.between(*periodo)]

    return df

def a(df):


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

    st.header("Grafico a dispersione")

    tab_interattivo, tab_statico = st.tabs(["interattivo","statico"]) 

    #GRAFICO INTERATTIVO
    with tab_interattivo:
        st.write("Mettendo il cursore sopra a un puntino è possibile leggere informazioni aggiuntive a esso")
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
                "giorni_dall_ultima_richiesta",
                "importo_medio_eur",
                "id",
                "cluster",
                "zona",
                "servizio_preferito",
                "richieste_ultimi_90_giorni",
                "importo_totale_eur",
                "giorni_dall_iscrizione"
            ]
        ).properties(
            height=400
        ).interactive()

        st.altair_chart(grafico, use_container_width=True)

        #GRAFICO STATICO
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
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    df = pd.read_csv(clusteredcsv)
    df = filtri_clienti(df)
with col2:
    st.write("")
    st.write("")
    with st.popover("Aggiungi cliente"):
        scaler = joblib.load(scalerpkl)
        modello = joblib.load(kmeanspkl)
        nuovo_cliente(scaler, modello)
with col3:
    mostra_editor = st.checkbox("Debug Mode")

if mostra_editor:
    st.header("Debug Mode")
    st.write("Questa è una funzionalità extra del progetto, utilizzata per effettuare test e generare nuovi dataset. Non è necessaria per il normale utilizzo della dashboard.")

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
                st.success("Nuovi dati generati!")

                a(df)
else:
     a(df)