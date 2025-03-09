import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io


# Load data
file_path = "training_data/Gandini Lorenzo/Gandini Lorenzo - Avvio rapido - 2024-11-22 18-01.csv"

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

# Time as x axes
data.set_index("Tempo (sec)", inplace=True)

# beginning of training phases
fine_riscaldamento = 300    # first 5 minutes
inizio_defaticamento = data.index.max() - 300 # last 5 minutes
inizio_allenamento = fine_riscaldamento + 1

# Define training phases
data['Fase'] = 'Fase Allenamento'
data.loc[data.index <= fine_riscaldamento, 'Fase'] = 'Riscaldamento'
data.loc[data.index > fine_riscaldamento, 'Fase'] = 'Allenamento'
data.loc[data.index >= inizio_defaticamento, 'Fase'] = 'Defaticamento'

print(f"Prime 10 righe prima del filtro per {file}: \n", data.head(10))

to_ignore = ['Distanza (m)','Cadenza (rpm)','Pendenza (%)']
data = data.drop(columns = to_ignore, errors='ignore')

print(f"Prime 10 righe dopo il filtro per {file}: \n", data.head(10))

for col in data.columns[:-1]:
    data[col] = data[col].replace(0, np.nan)  #To interpolate change 0 with NaN
    data[col] = data[col].interpolate(method='linear')  

    metriche_fasi = data.groupby('Fase')[col].agg(['mean', 'max', 'min']).reset_index()

    # Draw training phases
    plt.figure(figsize=(15, 5))
    plt.plot(data.index, data[col], label=col, color='blue')
    plt.fill_between(data.index, 0, data[col], where=(data['Fase'] == 'Riscaldamento'), color='green', alpha=0.1, label='Riscaldamento')
    plt.fill_between(data.index, 0, data[col], where=(data['Fase'] == 'Allenamento'), color='red', alpha=0.1, label='Allenamento')
    plt.fill_between(data.index, 0, data[col], where=(data['Fase'] == 'Defaticamento'), color='blue', alpha=0.1, label='Defaticamento')

    allenamento_stats = metriche_fasi[metriche_fasi['Fase'] == 'Allenamento']
    if not allenamento_stats.empty:
        max_value = allenamento_stats['max'].values[0]
        min_value = allenamento_stats['min'].values[0]

        max_index = data[(data['Fase'] == 'Allenamento') & (data[col] == max_value)].index[0]
        min_index = data[(data['Fase'] == 'Allenamento') & (data[col] == min_value)].index[0]

        # Draw max and min on the graph
        plt.scatter(max_index, max_value, color='red', s=75)
        plt.scatter(min_index, min_value, color='darkgreen', s=75)
        
        plt.annotate(text=f'Max : {int(max_value)}', xy=(max_index, max_value), xytext=(10, 10), textcoords='offset points', fontsize=10, color='black')
        plt.annotate(text=f'Min : {int(min_value)}', xy=(min_index, min_value), xytext=(10, -10), textcoords='offset points', fontsize=10, color='black')

    # Ad mean to the legend
    allenamento_mean = allenamento_stats['mean'].values[0] if not allenamento_stats.empty else None
    if allenamento_mean is not None:
        plt.plot([], [], ' ', label=f'Media Allenamento: {allenamento_mean:.2f}')

    # Draw graph
    plt.xlabel('Tempo (sec)')
    plt.ylabel(col)
    plt.ylim(0, data[col].max())
    plt.yticks(np.arange(0, (data[col].max() * 1.1) + (max(1, round((data[col].max() * 1.1) / 10))), max(1, round((data[col].max() * 1.1) / 10))))  
    plt.title('Andamento della ' + col.split(" ")[0] + ' e Fasi di Allenamento')
    plt.legend()
    plt.grid()
    plt.show()