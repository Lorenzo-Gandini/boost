import math

# Source : https://www.bikeitalia.it/calcolare-le-misure-della-bicicletta/
# Formule di Cyrille Guimard e indicazioni di BikeItalia

def calcola_setup_bici_corsa(cavallo, busto, braccia):
    # Calcola l'altezza della sella utilizzando la formula di Cyrille Guimard
    altezza_sella = cavallo * 0.883  # Comes from Cyrille Guimard studies

    # Calcola la distanza sella-manubrio basata sulla lunghezza del busto e delle braccia
    distanza_sella_manubrio = (busto + braccia) * 0.50

    # Calcola il dislivello sella-attacco (svettamento sella)
    dislivello_sella_attacco = altezza_sella * 0.235

    # Output dei risultati con descrizione
    risultati = {
        "Altezza sella (cm)": f"{round(altezza_sella, 2)}. Misurare dal centro della pedivella alla parte superiore della sella seguendo il piantone.",
        "Distanza sella-manubrio (cm)": f"{round(distanza_sella_manubrio, 2)}. Misurare dalla punta della sella al centro del manubrio.",
        "Altezza sella-manubrio (cm)": f"{round(dislivello_sella_attacco, 2)}. Differenza in altezza tra la sommità della sella e il punto più alto del manubrio."
    }

    return risultati

# Input dell'utente
cavallo = float(input("Inserisci l'altezza del cavallo (in cm): "))
busto = float(input("Inserisci la lunghezza del busto (in cm): "))
braccia = float(input("Inserisci la lunghezza delle braccia (in cm): "))

# Calcolo del setup ideale per bici da corsa
setup = calcola_setup_bici_corsa(altezza, cavallo, busto, braccia)

# Stampa i risultati
print("\n--- Setup Ideale della Bicicletta da Corsa ---")
for key, value in setup.items():
    print(f"{key}: {value} cm")