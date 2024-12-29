from utils import ask_athlete, ask_option, ask_yesno

def main():
    # Richiesta delle informazioni sull'atleta
    athlete = ask_athlete("Which athlete do you want to analyze?")
    athlete_mod = athlete.replace(" ", "_")
    athlete_mod_uc = athlete_mod.upper()

    # Opzioni selezionate dall'utente
    option = ask_option(
        "\nWhat type of analysis do you want to run? \n"
        "OPTIONS:\n"
        "1. Spine movements\n"
        "2. Knee angles\n"
        "3. Ankle angles\n"
        "4. Training sessions\n"
        "5. All of them.\n"
        "---> "
    )
    want_pdf = ask_yesno("\nDo you want a report in pdf? (yes / no)\n--->")
    show_plots = ask_yesno("Do you want to see the plots during the analysis? (yes/no):")

    # FIX THIS
    # show_animation = ask_yesno("Do you want to see the animation of the athlete? (yes/no):")
    # if show_animation:
    #     show_animation(take, bones_pos, body_edges, colors, [0, 1, 2])


    # Mappa delle opzioni
    SPINE = option in {1, 5}
    KNEE = option in {2, 5}
    ANKLE = option in {3, 5}
    TRAINING = option in {4, 5}
    PDF = want_pdf

    # Esegui le analisi con i parametri
    if SPINE:
        import spine_analysis
        spine_analysis.run_spine_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots)

    if KNEE:
        import knee_analysis
        knee_analysis.run_knee_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots)
    
    if ANKLE:
        import ankle_analysis
        ankle_analysis.run_ankle_analysis(athlete, athlete_mod, athlete_mod_uc, show_plots)

    if TRAINING:
        import training_analysis
        training_analysis.run_training_analysis(athlete, athlete_mod_uc, show_plots)

    if PDF:
        import pdf_generator
        pdf_generator.generate_report(athlete)

    # Recap finale
    print("\n---- RECAP OF YOUR CHOICES ----")
    print(f"Athlete: {athlete}")
    print(f"SPINE Analysis: {'Enabled' if SPINE else 'Disabled'}")
    print(f"LEG Analysis: {'Enabled' if KNEE else 'Disabled'}")
    print(f"ANKLE Analysis: {'Enabled' if ANKLE else 'Disabled'}")
    print(f"TRAINING Analysis: {'Enabled' if TRAINING else 'Disabled'}")
    print(f"PDF Report: {'Yes' if PDF else 'No'}")

    print("\nAnalysis completed.")

if __name__ == "__main__":
    main()
