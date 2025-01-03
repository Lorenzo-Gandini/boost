# Boost

Welcome to the Boost repository! This project is focused on enhancing sports technology through innovative software solutions.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have the following software installed on your machine:

- [Python 3.x](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/boost.git
    ```
2. Navigate to the project directory:
    ```sh
    cd boost
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Code

To run the main application, use the following command:
```sh
python main.py
```
User Interaction
After running python main.py, you will be prompted to provide some information and make choices for the analysis:

Select Athlete: You will be asked to select an athlete from the list of available athletes in the training_data folder.

🤔 Which athlete do you want to analyze?
   1. Athlete1
   2. Athlete2
   ...

2. Select Analysis Type: You will be asked to choose the type of analysis you want to run.
🤔 What type of analysis do you want to run?
   1. Spine movements
   2. Knee angles
   3. Ankle angles
   4. Training sessions
   5. All of them.

3. Show Plots: You will be asked if you want to see the plots during the analysis.
🤔 Do you want to see the plots during the analysis? (yes/no)

Recap and Confirmation: A recap of your choices will be displayed, and you will be asked to confirm if you want to proceed with the analysis.
---- RECAP OF YOUR CHOICES ----
   Athlete: Athlete1
   SPINE Analysis: Enabled
   LEG Analysis: Disabled
   ANKLE Analysis: Disabled
   TRAINING Analysis: Disabled
   PDF Report: Yes

🤔 Do you want to proceed with these choices? (yes/no)

Joint Side Selection: For knee and ankle analyses, you will be asked to select which side (right, left, or both) you want to analyze.


🤔 Which knee do you want to analyze?
   1. Right
   2. Left
   3. Both

   Follow the prompts to complete the analysis. The results will be saved in the output/{athlete} directory. ```