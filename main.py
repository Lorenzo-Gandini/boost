from utils import ask_athlete, ask_option, ask_yesno, user_message, print_recap

def main():
    # Interaction with the user about athlets and desired output
    print("\nWelcome to BOOST, the biomechanial tool perfect for your analysis! \nLet's start by selecting the athlete and the type of analysis you want to run. \nPlease follow the instructions below. \n")
    athlete = ask_athlete("Which athlete do you want to analyze?\n(We suggest you athlete #3 or #6, since they are the most complete).")
    option = ask_option("What type of analysis do you want to run?")
    # want_pdf = ask_yesno("Do you want a PDF report?") #WORKINPROGRESS
    show_plots = ask_yesno("Do you want to see the plots during the analysis?")

    SPINE = option in {1, 5}
    KNEE = option in {2, 5}
    ANKLE = option in {3, 5}
    TRAINING = option in {4, 5}
    # PDF = want_pdf

    choices = {
        "athlete": athlete,
        "spine": SPINE,
        "leg": KNEE,
        "ankle": ANKLE,
        "training": TRAINING
        # "pdf": PDF
    }
    print_recap(choices)
    proceed = ask_yesno("Do you want to proceed with these choices?")
    if not proceed:
        user_message("Analysis aborted. Please restart the program.", "error")
        exit()
    else:
        user_message("Ok, let's go and proceed with the analysis.", "success")
        
    # Esegui le analisi con i parametri
    if SPINE:
        user_message("SPINE analysis will be performed.", "info")
        import spine_analysis
        spine_analysis.run_spine_analysis(athlete, show_plots)

    if KNEE:
        user_message("KNEE analysis will be performed.", "info")
        import knee_analysis
        knee_analysis.run_knee_analysis(athlete, show_plots)

    if ANKLE:
        user_message("ANKLE analysis will be performed.", "info")
        import ankle_analysis
        ankle_analysis.run_ankle_analysis(athlete, show_plots)

    if TRAINING:
        user_message("TRAINING analysis will be performed.", "info")
        import training_analysis
        training_analysis.run_training_analysis(athlete, show_plots)

    # if PDF:
    #     user_message("A PDF report will be generated.", "info")
    #     import pdf_generator
    #     pdf_generator.generate_report(athlete)

    
if __name__ == "__main__":
        main()
