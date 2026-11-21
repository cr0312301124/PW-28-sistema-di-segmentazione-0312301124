from config import SEED,NUM_CLIENTI,RANDOM_STATE,nome_file
from generatore_dati import generatore
from preelaborazione import prelab
from raggruppamento import gomito,silhouette,elab_finale
from contextlib import redirect_stdout

with open(nome_file, "w", encoding="utf-8") as file:
    with redirect_stdout(file):
        print("SEED: ", SEED)
        print("RANDOM_STATE: ", RANDOM_STATE)
        print("NUM_CLIENTI:", NUM_CLIENTI)
        generatore(SEED,NUM_CLIENTI)
        print("generazione completata")
        prelab()
        print("prelaborazione eseguita")
        x = gomito(RANDOM_STATE)
        print("gomito generato")
        y, num_cluster = silhouette(RANDOM_STATE, x)
        print("silhouette completata")
        elab_finale(RANDOM_STATE, y, num_cluster)
        print("elaborato finale")
