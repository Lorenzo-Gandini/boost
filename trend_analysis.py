import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


# Load data
file_path = "Sessioni allenamento/Gandini Lorenzo/Gandini Lorenzo - Avvio rapido - 2024-11-22 18-01.csv"

# Read the file, eliminating trailing semicolons since Technogym leave a ";" at the end of every line exept the first one
with open(file_path, 'r', encoding='utf-16') as file:
    lines = [line.rstrip(';\n') for line in file]

cleaned_data = io.StringIO('\n'.join(lines))

data = pd.read_csv(cleaned_data, encoding='utf-16', delimiter=';', skiprows=1,engine='python', on_bad_lines='warn')
data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)  #convert all float with "," with "." since in the file are strings
data = data.apply(pd.to_numeric, errors='coerce')   

# Extract name of columns
data.columns = [col.strip() for col in data.columns]

check = False
for col in data.columns:
    if "Tempo (sec)" == col:
        check = True
        break

if not(check):
    raise KeyError("La colonna 'Tempo (sec)' non è presente nel DataFrame. Verifica il file CSV e assicurati che i nomi delle colonne siano corretti.")

