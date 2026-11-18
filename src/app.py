import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from config import clusteredcsv,kmeanspkl,scalerpkl




st.set_page_config(
    page_title="Project Work", 
)

#tab_a, tab_b = st.tabs(["A","B"])

df = pd.read_csv(clusteredcsv)
scaler = joblib.load(scalerpkl)
modello = joblib.load(kmeanspkl)

#with tab_a:
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

#with tab_b:
#    col1, col2 = st.columns(2)
#
#    with col1:
#        a = st.number_input(
#            "test",
#            value=34296
#        )
#    with col2:
#        b = st.number_input(
#            "test",
#            value=2004
#        )

