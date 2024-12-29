import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
from datetime import datetime

def load_files(folder_path, athlete):
    file_pattern = re.compile(rf"{re.escape(athlete)} - (.*?) - (\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}-\d{{2}})\.csv")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    selected_files = []
    for file in files:
        match = file_pattern.match(file)
        if match and "Avvio rapido" in match.group(1):
            training_date = datetime.strptime(match.group(2), "%Y-%m-%d %H-%M")
            selected_files.append((file, training_date))
    selected_files.sort(key=lambda x: x[1])
    return selected_files

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-16') as f:
        lines = [line.rstrip(';\n') for line in f]
    cleaned_data = io.StringIO('\n'.join(lines))
    data = pd.read_csv(cleaned_data, encoding='utf-16', delimiter=';', skiprows=1, engine='python', on_bad_lines='warn')
    data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.columns = [col.strip() for col in data.columns]
    if "Tempo (sec)" not in data.columns:
        raise KeyError("La colonna 'Tempo (sec)' non è presente nel DataFrame. Verifica il file CSV e assicurati che i nomi delle colonne siano corretti.")
    data.set_index("Tempo (sec)", inplace=True)
    return data

def analyze_data(data):
    fine_riscaldamento = 300
    inizio_defaticamento = data.index.max() - 300
    data['Fase'] = 'Fase Allenamento'
    data.loc[data.index <= fine_riscaldamento, 'Fase'] = 'Riscaldamento'
    data.loc[data.index > fine_riscaldamento, 'Fase'] = 'Allenamento'
    data.loc[data.index >= inizio_defaticamento, 'Fase'] = 'Defaticamento'
    to_ignore = ['Distanza (m)', 'Cadenza (rpm)', 'Pendenza (%)']
    data = data.drop(columns=to_ignore, errors='ignore')
    metrics = {}
    for col in data.columns[:-1]:
        if col in data.columns:
            data[col] = data[col].replace(0, np.nan)
            data[col] = data[col].interpolate(method='linear')
            metriche_fasi = data.groupby('Fase')[col].agg(['mean', 'max', 'min']).reset_index()
            allenamento_stats = metriche_fasi[metriche_fasi['Fase'] == 'Allenamento']
            if not allenamento_stats.empty:
                metrics[col] = {
                    'max': allenamento_stats['max'].values[0],
                    'min': allenamento_stats['min'].values[0],
                    'mean': allenamento_stats['mean'].values[0]
                }
    return metrics

def save_summary(results, athlete, athlete_mod_uc):
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
    file_name = f"{athlete_mod_uc}_training_summary.csv"
    summary_df.to_csv(f"output/{athlete}/stats/{file_name}", index=False)
    print(f"Il riepilogo è stato salvato come {file_name}.")
    return summary_df

def plot_results(summary_df, athlete, athlete_mod_uc, show_plots):
    metrics_to_plot = ['Potenza (watt)', 'Velocita (km/h)', 'Frequenza Cardiaca (bpm)']
    
    for metric in metrics_to_plot:
        metric_name = re.sub(r'\s*\(.*?\)', '', metric)
        output_file = os.path.join(f"output/{athlete}/plots/", f"{athlete_mod_uc}_training_{metric_name}.png")
        metric_data = summary_df[summary_df['Metric'] == metric]
        plt.figure(figsize=(10, 6))
        plt.plot(metric_data['Date'], metric_data['Max'], label='Max', marker='o', linestyle='-')
        plt.plot(metric_data['Date'], metric_data['Mean'], label='Mean', marker='o', linestyle='-')
        plt.title(f"Trend of {metric} over time")
        plt.xlabel("Date")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xticks(metric_data['Date'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save and optionally show the plot

        plt.savefig(output_file)
        if show_plots:
            plt.show()
        plt.close()

def run_training_analysis(athlete, athlete_mod_uc, show_plots):
    folder_path = f"training_data/{athlete}"
    selected_files = load_files(folder_path, athlete)
    results = []
    for file, training_date in selected_files:
        file_path = os.path.join(folder_path, file)
        print(file_path)
        data = process_file(file_path)
        metrics = analyze_data(data)
        results.append({
            'file': file,
            'date': training_date,
            'metrics': metrics
        })
    summary_df = save_summary(results, athlete, athlete_mod_uc)
    plot_results(summary_df, athlete, athlete_mod_uc, show_plots)