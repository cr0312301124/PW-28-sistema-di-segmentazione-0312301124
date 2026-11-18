import numpy as np
import pandas as pd
from config import clienticsv
from config import DATA_DI_RIFERIMENTO

def generatore(SEED=34296, NUM_CLIENTI=600):
    clienti = []

    rng = np.random.default_rng(SEED)

    #DEBUG
    #print("primo numero rng:", rng.random())
    #print("secondo numero rng:", rng.random())

    #Zone geografiche dei clienti
    zone = ["Nord-Ovest","Nord-Est","Centro","Sud","Isole","Estero"]

    #Servizi disponibili
    servizi = ["Consegne","Consulenza","Assistenza","Formazione","Abbonamenti"]

    #Status clienti
    status = ["Molto attivi","Occasionali","Recenti","Inattivi"]
    prob_status = [0.30,0.35,0.15,0.20]

    for i in range(NUM_CLIENTI):
        zona = rng.choice(zone)
        stato = rng.choice(status, p=prob_status)
        servizio = rng.choice(servizi)

        if stato == "Molto attivi":
            richieste_ultimi_90_giorni = rng.integers(12,35)
            giorni_dall_ultima_richiesta = rng.integers(0,10)
            richieste_precedenti_90_giorni = rng.integers(10,60)
            spesa_media_per_richiesta = rng.uniform(25,90)
            giorni_dall_iscrizione = rng.integers(100,700) #non possono essere recenti

        elif stato == "Recenti":
            giorni_dall_iscrizione = rng.integers(1,40)
            if giorni_dall_iscrizione <= 7:
                richieste_ultimi_90_giorni = rng.integers(1, giorni_dall_iscrizione + 1)
            else:
                richieste_ultimi_90_giorni = rng.integers(1,7)
            richieste_precedenti_90_giorni = 0
            spesa_media_per_richiesta = rng.uniform(10,50)
            giorni_dall_ultima_richiesta = rng.integers(1, giorni_dall_iscrizione + 1)

        elif stato == "Occasionali":
            richieste_ultimi_90_giorni = rng.integers(1,9)
            giorni_dall_ultima_richiesta = rng.integers(7,75)
            richieste_precedenti_90_giorni = rng.integers(0,31)
            spesa_media_per_richiesta = rng.uniform(15,65)
            giorni_dall_iscrizione = rng.integers(100,700)

        else: #Inattivi
            richieste_ultimi_90_giorni = 0
            giorni_dall_ultima_richiesta = rng.integers(90,180)
            richieste_precedenti_90_giorni = rng.integers(1,30)
            spesa_media_per_richiesta = rng.uniform(15,75)
            giorni_dall_iscrizione = rng.integers(180,700)

        data_iscrizione = (
            DATA_DI_RIFERIMENTO - pd.Timedelta(days=int(giorni_dall_iscrizione))
        )

        richieste_totali = richieste_ultimi_90_giorni + richieste_precedenti_90_giorni
        spesa_totale = spesa_media_per_richiesta * richieste_totali

        cliente = {
            "id": f"C{i + 1:04d}",
            "data_iscrizione": data_iscrizione,
            "zona": zona,
            "servizio_preferito": servizio,
            "profilo simulato": stato,
            "richieste_ultimi_90_giorni": richieste_ultimi_90_giorni,
            "richieste_totali": richieste_totali,
            "importo_medio_eur": round(spesa_media_per_richiesta, 2),
            "importo_totale_eur": round(spesa_totale, 2),
            "giorni_dall_ultima_richiesta": giorni_dall_ultima_richiesta,
            "giorni_dall_iscrizione": giorni_dall_iscrizione
        }

        clienti.append(cliente)

    #dataframe
    df = pd.DataFrame(clienti)

    df.to_csv(clienticsv, index=False)

    print("Generazione Completata")
    print(f"Clienti Generati: {len(df)}")
    print(f"File salvato in: {clienticsv}")

    print("\nControllo anomalie:")
    print("ID duplicati:", df["id"].duplicated().sum())
    print("Importi negativi:", (df["importo_totale_eur"] < 0).sum())
    print("Richieste negative:", (df["richieste_totali"] < 0).sum())

    # Ultima richiesta antecedente all'iscrizione
    giorni_iscrizione = (DATA_DI_RIFERIMENTO - df["data_iscrizione"]).dt.days
    date_anomale = df[df["giorni_dall_ultima_richiesta"] > giorni_iscrizione]
    print("\nUltima richiesta antecedente all'iscrizione:")
    print(date_anomale["id"].tolist() if not date_anomale.empty else "Nessuna anomalia")
    return