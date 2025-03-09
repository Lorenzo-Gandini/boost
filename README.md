# Boost

Welcome to the **Boost** repository! This project leverages advanced sports technology to analyze biomechanical and performance data, helping athletes and researchers optimize training, improve performances and reduce injury risk.

## Getting Started

Follow the steps below to set up and run the project on your local machine.

---

### Prerequisites

Ensure you have the following software installed:

- [Python 3.x](https://www.python.org/downloads/) (required)
- [pip](https://pip.pypa.io/en/stable/installation/) (for package management)

---

### Installation

1. **Clone the Repository**  
   Clone the Boost repository to your local machine:
   ```bash
   git clone https://github.com/Lorenzo-Gandini/boost
   ```
2. **Navigate to the Project Directory**  
   Move into the project folder:
   ```bash
   cd boost
   ```
3. **Install Dependencies**  
   Install all required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

### Running the Project

To execute the main application, run the following command:
```bash
python main.py
```

---

## How to Use Boost

### Interactive Workflow

Once the application starts, you will be guided through an interactive menu:

1. **Select Athlete**  
   Choose an athlete for analysis. Athletes are loaded from the `training_data` folder:
   ```bash
   🤔 Which athlete do you want to analyze?
      1. Athlete1
      2. Athlete2
      ...
   ```

2. **Choose Analysis Type**  
   Select the type(s) of analysis you want to run:
   ```bash
   🤔 What type of analysis do you want to run?
      1. Spine movements
      2. Knee angles
      3. Ankle angles
      4. Training sessions
      5. All of them
   ```

3. **Show Plots**  
   Specify whether to display plots during the analysis:
   ```bash
   🤔 Do you want to see the plots during the analysis? (yes/no)
   ```

4. **Review Your Choices**  
   The program will recap your selections and ask for confirmation:
   ```bash
   ---- RECAP OF YOUR CHOICES ----
      Athlete: Athlete1
      SPINE Analysis: Enabled
      LEG Analysis: Disabled
      ANKLE Analysis: Disabled
      TRAINING Analysis: Disabled
      PDF Report: Yes

   🤔 Do you want to proceed with these choices? (yes/no)
   ```

5. **Side Selection (for Knee and Ankle Analyses)**  
   If analyzing knees or ankles, choose the side(s) to focus on:
   ```bash
   🤔 Which knee do you want to analyze?
      1. Right
      2. Left
      3. Both
   ```

---

### Output

Results will be saved in the `output/{athlete}` directory, organized by analysis type. If enabled, visual plots and a PDF report will also be generated.

The directory structure for output will look like this:
```bash
output/
├── Athlete1/
│   ├── spine_analysis/
│   │   ├── plot1.png
│   │   └── report.pdf
│   ├── knee_analysis/
│   │   ├── plot1.png
│   │   └── plot2.png
│   └── ...
├── Athlete2/
│   └── ...
```

---

### Run a Single Training Session Analysis

Instead of running the entire analysis, you can perform an analysis of a single session. In the file `single_session_analysis.py`, there is a variable named `file_path`.  
You can set this variable to the desired `.csv` file located in the `training_data` folder to generate graphs for that specific training session.

### Customize Pedal Cycle Plotting in Knee and Ankle Analyses

In the knee and ankle analysis scripts (`knee_analysis.py` and `ankle_analysis.py`), you can define which specific pedal cycle to plot by setting the variable `cycle_index`.  

This variable allows you to focus on a particular cycle of the pedaling motion for detailed analysis. Simply edit the `cycle_index` variable in the corresponding script to the desired cycle number before running the analysis.

For example:
```python
cycle_index = 75  # Plots the 75th pedal cycle
