import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.utils.helper import get_main_config, has_values

# Authentication configuration
LOGIN_CREDENTIALS = [
    ("daniel", "daniel"),
    ("michael", "michael"),
    ("marcel", "marcel"),
    ("veton", "veton"),
]

# Get configuration
config = get_main_config()

MODELS = [
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4o-2024-08-06",
]

opinion_analyzer = OpinionAnalyzer(model_name_or_path=config["models"]["llm"])
analyzer_dict = {"opinion_analyzer": opinion_analyzer}

# Shared state for cancellation
cancel_status = {"cancel": False}

COLUMN_RENAMING = deepcopy(config["app"]["column_renaming"])
SELECTED_SOURCES = opinion_analyzer.get_unique_field_values("business_id")
SIMILARITY_THRESHOLD = config["thresholds"]["sentence_similarity"]


def prepare_pipeline(model_name_or_path: str, dropdown_source_selection: list[str], selected_columns: list[str]):
    """
    Initializes the OpinionAnalyzer and updates the model used for analysis.
    Parameters:
    model_name_or_path (str): The name or path of the model to be used for opinion analysis.
    selected_columns (list): List of selected columns to display in the DataFrame.
    Side Effects:
    - Updates the global 'analyzer_dict' with a new instance of OpinionAnalyzer.
    - Updates the global 'selected_columns'.
    - Prints a message indicating the change of the model.
    """

    global COLUMN_RENAMING
    global SELECTED_SOURCES
    global SIMILARITY_THRESHOLD

    COLUMN_RENAMING = {
        k: v
        for k, v in config["app"]["column_renaming"].items()
        if v in selected_columns
    }

    SELECTED_SOURCES = dropdown_source_selection

    analyzer_dict["opinion_analyzer"] = OpinionAnalyzer(
        model_name_or_path=model_name_or_path
    )
    print(
        "Changing the model to: ", analyzer_dict["opinion_analyzer"].model_name_or_path
    )
    print("Displaying columns: ", selected_columns)


def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns in a given DataFrame and reorders the DataFrame to match the new column order.
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be modified.
    columns_renamings (dict): A dictionary specifying the columns to be renamed in the form {old_name: new_name}.
    Returns:
    pd.DataFrame: The modified DataFrame with renamed and reordered columns.
    """
    dataframe.rename(columns=COLUMN_RENAMING, inplace=True)
    dataframe = dataframe[[new_col for old_col, new_col in COLUMN_RENAMING.items()]]
    return dataframe


def update_progress(total_num_matches: int, analyzed_matches: int, num_arguments: int, canceled=False) -> str:
    """
    Updates the progress bar with the current analysis status.

    :param total_num_matches: The total number of thematic matches found.
    :type total_num_matches: int
    :param analyzed_matches: The number of matches that have been checked for arguments.
    :type analyzed_matches: int
    :return: A string containing the progress information.
    :rtype: str

    Example:
    >>> update_progress(5, 3)
    'Die Argumentsuche läuft! Es gibt insgesamt <PROGRESS_MAX> thematische Übereinstimmungen. Davon wurden bereits 3 auf Argumente überprüft. Es wurden 5 Argumente gefunden.'
    """

    if canceled:
        progress_info = "Die Argumentensuche wurde abgebrochen!"
    elif total_num_matches == 0:
        progress_info = "Es wurden keine Argumente gefunden."
    elif total_num_matches == analyzed_matches and total_num_matches > 0:
        progress_info = (
            f"Die Argumentensuche ist abgeschlossen! Es gibt insgesamt {total_num_matches} thematische Übereinstimmungen. "
            f"Davon wurden bereits {analyzed_matches} auf Argumente überprüft. "
            f"Es wurden {num_arguments} Argumente gefunden."
        )
    else:
        progress_info = (
            f"Die Argumentsuche läuft! Es gibt insgesamt {total_num_matches} thematische Übereinstimmungen. "
            f"Davon wurden bereits {analyzed_matches} auf Argumente überprüft. "
            f"Es wurden {num_arguments} Argumente gefunden."
        )
    return progress_info


# Function to perform argument mining and yield results incrementally
def perform_argument_mining(input_text: str, similarity_threshold: float = None):
    """
    Performs argument mining on the provided input text.
    Parameters:
    - input_text (str): The text to be analyzed for arguments.
    - similarity_threshold (float, optional): A threshold value for similarity to filter arguments.

    The function handles:
    - Initializing a DataFrame with renamed columns.
    - Iterating through found arguments and checking for cancellation status.
    - Filtering arguments based on labels and appending to the accumulated DataFrame.
    - Yielding the accumulated DataFrame for updates.
    - Simulating processing time.
    """

    accumulated_df = pd.DataFrame(
        columns=[new_col for old_col, new_col in COLUMN_RENAMING.items()]
    )
    accumulated_df = rename_columns(accumulated_df)
    accumulated_df.style.set_table_styles(
        [dict(selector="th", props=[("max-width", "50px")])]
    )
    # Simulate finding arguments incrementally for the sake of demonstration
    print("Using this model: ", analyzer_dict["opinion_analyzer"].model_name_or_path)
    arguments = analyzer_dict["opinion_analyzer"].find_arguments(
        topic_text=input_text, similarity_threshold=similarity_threshold, allowed_business_ids=SELECTED_SOURCES
    )

    matches_found, _ = has_values(arguments)
    if not matches_found:
        progress_info = update_progress(0, 0, len(accumulated_df))
        yield accumulated_df, progress_info

    neutral_count = 0
    already_analyzed = 0

    for idx, row in enumerate(arguments):
        if cancel_status["cancel"] or neutral_count >= config["app"]["neutral_limit"]:
            progress_info = update_progress(row["num_matches"], already_analyzed, len(accumulated_df),
                                            canceled=cancel_status["cancel"])
            yield accumulated_df, progress_info
            break
        if row["label"] in ["pro", "contra"]:
            current_row_df = pd.DataFrame([row])
            current_row_df = rename_columns(current_row_df)
            # Append the current row to the accumulated DataFrame
            accumulated_df = pd.concat(
                [accumulated_df, current_row_df], ignore_index=True
            )
            accumulated_df.style.set_table_styles(
                [dict(selector="th", props=[("max-width", "50px")])]
            )
            neutral_count = 0
        else:
            neutral_count += 1
            print(f"Number of neutral stances: {neutral_count}")
        already_analyzed += 1
        progress_info = update_progress(row["num_matches"], already_analyzed, len(accumulated_df),
                                        canceled=cancel_status["cancel"])

        yield accumulated_df, progress_info


# Function to reset the output table
def reset_output():
    """
    Creates and returns an empty pandas DataFrame with pre-defined columns.
    Returns:
        pd.DataFrame: An empty DataFrame with specific columns for storing various labels, arguments, reasoning and metadata.
    """
    return pd.DataFrame(
        columns=[new_col for old_col, new_col in COLUMN_RENAMING.items()]
    )


# Function to set the cancel flag
def cancel_operation():
    """
    Cancels an ongoing operation by setting the cancel status to True.
    Modifies the global variable cancel_status by setting its "cancel" key to True, indicating that an operation should be halted.
    """
    cancel_status["cancel"] = True


# Function to clear the cancel flag and reset
def setup_operation():
    """
    Initializes the setup operation by resetting the cancel status to False.
    This function is intended to prepare the environment or context
    for a specific operation that requires a cancellation mechanism.
    By setting the cancel status to False, it ensures that any previous
    cancellation states are cleared.
    Parameters:
    None
    Returns:
    None
    """
    cancel_status["cancel"] = False


# Function to save the DataFrame as an Excel file
def save_as_excel(dataframe):
    """
    Saves the provided DataFrame as an Excel file.
    Parameters:
    dataframe (pd.DataFrame): The DataFrame to be saved.
    Returns:
    str: Path to the saved Excel file.
    """
    dataframe["similarity_threshold"] = SIMILARITY_THRESHOLD
    output_path = Path(config["paths"]["user_data"])
    os.makedirs(output_path, exist_ok=True)
    file_path = output_path / f"saved_results_{datetime.now().isoformat()}.xlsx"
    dataframe.to_excel(file_path, index=False)
    return str(file_path)


def get_username(request: gr.Request):
    # https://github.com/gradio-app/gradio/issues/3259
    # https://stackoverflow.com/questions/77494902/how-to-retrieve-users-username-after-authenticating-in-gradio
    print("The current user is: ", {request.username})
    return {request.username}


# Gradio app interface with vertical layout
with gr.Blocks() as interface:
    gr.Markdown("# Argument Mining App")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Spezifiziere Einstellungen")
            dropdown_model_selection = gr.Dropdown(choices=MODELS, label="Modellauswahl", info="Wähle ein Model aus")

            dropdown_source_selection = gr.Dropdown(choices=opinion_analyzer.get_unique_field_values("business_id"),
                                                    label="Quellenauswahl",
                                                    info="Wähle die Geschäfte aus, die durchsucht werden sollen.",
                                                    multiselect=True)

            column_selection = gr.CheckboxGroup(
                choices=list(COLUMN_RENAMING.values()),
                value=list(
                    COLUMN_RENAMING.values()
                ),  # Default to selecting all columns
                label="Spaltenauswahl",
                info="Wähle die Spalten, die in der Ergebnistabelle dargestellt werden sollen.",
            )
            # Slider for similarity threshold
            similarity_threshold_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=config["thresholds"]["sentence_similarity"],
                step=0.01,
                label="Schwellenwert für die semantischen Ähnlichkeit"
            )
            hyperparam_button = gr.Button("Einstellungen übernehmen")
            hyperparam_button.click(
                fn=prepare_pipeline,  # Reset cancel status before starting
                inputs=[dropdown_model_selection, dropdown_source_selection, column_selection],
                outputs=[],
            )

    with gr.Row():
        with gr.Column():
            # Input text box
            gr.Markdown("## Suche nach Pro- und Contra-Argumenten")
            input_text = gr.Textbox(
                lines=2,
                placeholder="Füge ein Text ein, um entsprechende Argumente zu finden...",
            )
            # Buttons to trigger the analysis and cancel the operation
            with gr.Row():
                submit_button = gr.Button("Beginne Argumentensuche!")
                cancel_button = gr.Button("Abbrechen")

            # Progress bar display
            progress_display = gr.Textbox(label="Status der Argumentensuche", interactive=False,
                                          lines=1, visible=False)

            # Output DataFrame
            output_table = gr.Dataframe(wrap=True, visible=False)
            # Export button to save the DataFrame as an Excel file
            export_button = gr.Button("Exportiere die Ergebnisse als Excel", visible=False)
            # Link input and output with the function
            submit_button.click(
                fn=setup_operation,
                inputs=[],
                outputs=[],
            )


            def change_visibility():
                return {progress_display: gr.Textbox(visible=True),
                        output_table: gr.Dataframe(visible=True),
                        export_button: gr.Button(visible=True)
                        }


            submit_button.click(fn=change_visibility, outputs=[progress_display, output_table, export_button])

            submit_button.click(
                fn=perform_argument_mining,
                inputs=[input_text, similarity_threshold_slider],
                outputs=[output_table, progress_display]
            )

            # Reset the output table when new input is provided
            input_text.change(
                fn=reset_output,
                inputs=[],
                outputs=output_table,
            )
            # Bind the cancel button
            cancel_button.click(
                fn=cancel_operation,
                inputs=[],
            )
            # Bind the export button
            export_button.click(
                fn=save_as_excel,
                inputs=[output_table],
                outputs=gr.File(label="Sichere Ergebnisse als Excel-Datei"),
            )
# Launch the app with authentication
if __name__ == "__main__":
    interface.launch(auth=LOGIN_CREDENTIALS, share=True)
