# Source of formulas : https://www.bikeitalia.it/calcolare-le-misure-della-bicicletta/

def calcola_setup_bici_corsa(cavallo, busto):
    altezza_sella = cavallo * 0.885

    coefficiente_raggiungimento = 1.11  # Race bike coefficient, different from MTB or Gravel
    distanza_sella_manubrio = busto * coefficiente_raggiungimento

    dislivello_sella_manubrio = cavallo * 0.06

    # Print of the results and instruction to get it
    risultati = {
        "Altezza sella (cm)": f"{round(altezza_sella, 2)}. Misurare dal centro della pedivella alla parte superiore della sella seguendo il piantone.",
        "Distanza sella-manubrio (cm)": f"{round(distanza_sella_manubrio, 2)}. Misurare dalla punta della sella al centro del manubrio.",
        "Altezza sella-manubrio (cm)": f"{round(dislivello_sella_manubrio, 2)}. Differenza in altezza tra la sommità della sella e il punto più alto del manubrio."
    }

    return risultati

# Input of measures
cavallo = float(input("Inserisci l'altezza del cavallo (in cm): "))
busto = float(input("Inserisci la lunghezza del busto (in cm): "))

setup = calcola_setup_bici_corsa(cavallo, busto)

print("\n--- Setup Ideale della Bicicletta da Corsa ---")
for key, value in setup.items():
    print(f"{key}: {value}")
