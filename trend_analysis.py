import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import datetime

# Define folder path
athlete_name = "Masutti Erik"
folder_path = f"training_data/{athlete_name}"

# Regex to extract training type and date
file_pattern = re.compile(rf"{re.escape(athlete_name)} - (.*?) - (\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}-\d{{2}})\.csv")

# List of files to process
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Filter files with "Avvio rapido" in the name
selected_files = []
for file in files:
    match = file_pattern.match(file)
    if match and "Avvio rapido" in match.group(1):
        training_date = datetime.strptime(match.group(2), "%Y-%m-%d %H-%M")
        selected_files.append((file, training_date))

# Sort files by date
selected_files.sort(key=lambda x: x[1])

# Initialize results dictionary
results = []

# Process each file
for file, training_date in selected_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path, 'r', encoding='utf-16') as f:
        lines = [line.rstrip(';\n') for line in f]

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

    to_ignore = ['Distanza (m)','Cadenza (rpm)','Pendenza (%)']
    data = data.drop(columns = to_ignore, errors='ignore')

    # Interpolate data and calculate metrics for Allenamento phase
    metrics = {}
    for col in data.columns[:-1]:
        if col in data.columns:
            data[col] = data[col].replace(0, np.nan)  # Replace 0 with NaN
            data[col] = data[col].interpolate(method='linear')  # Interpolate missing values

            metriche_fasi = data.groupby('Fase')[col].agg(['mean', 'max', 'min']).reset_index()

            allenamento_stats = metriche_fasi[metriche_fasi['Fase'] == 'Allenamento']

            if not allenamento_stats.empty:
                metrics[col] = {
                    'max': allenamento_stats['max'].values[0],
                    'min': allenamento_stats['min'].values[0],
                    'mean': allenamento_stats['mean'].values[0]
                }
            # print(f"Data column : {col} and stats :\n {allenamento_stats}")

    results.append({
        'file': file,
        'date': training_date,
        'metrics': metrics
    })


# Create a summary DataFrame
summary_data = []
for result in results:
    file_date = result['date']
    for metric, values in result['metrics'].items():
        summary_data.append({
            'File': result['file'],
            'Date': file_date,
            'Metric': metric,
            'Max': values['max'],
            'Min': values['min'],
            'Mean': values['mean']
        })

summary_df = pd.DataFrame(summary_data)

# Save the summary to a CSV file
summary_df.to_csv("output/training_analysis_summary.csv", index=False)
print("Il riepilogo è stato salvato come 'training_analysis_summary.csv'.")


# Plot the results
metrics_to_plot = ['Potenza (watt)', 'Velocità (km/h)', 'Frequenza Cardiaca (bpm)']
for metric in metrics_to_plot:
    metric_data = summary_df[summary_df['Metric'] == metric]

    plt.figure(figsize=(10, 6))
    plt.plot(metric_data['Date'], metric_data['Max'], label='Max', marker='o', linestyle='-')
    plt.plot(metric_data['Date'], metric_data['Min'], label='Min', marker='o', linestyle='-')
    plt.plot(metric_data['Date'], metric_data['Mean'], label='Mean', marker='o', linestyle='-')
    
    plt.title(f"Trend of {metric} over time")
    plt.xlabel("Date")
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())  # Show one label per day
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format dates as YYYY-MM-DD
    ax.set_xticks(metric_data['Date'])  # Explicitly set the ticks to match the data

    # Rotate the labels
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()