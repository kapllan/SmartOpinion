import time
import gradio as gr
import pandas as pd
from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.utils.helper import get_main_config

# Get configuration
config = get_main_config()
MODELS = [
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "gpt-4o-2024-08-06",
]
opinion_analyzer = OpinionAnalyzer(model_name_or_path=config["models"]["llm"])
analyzer_dict = {"opinion_analyzer": opinion_analyzer}
# Shared state for cancellation
cancel_status = {"cancel": False}


COLUMN_RENAMING = {
    "topic_original": "Worum geht es?",
    "label": "Haltung",
    "argument_original": "Ansicht",
    "argument_reason": "Modellbegründung",
    "reasoning": "Beweis",
    "reasoning_segment": "Abschnitt mit Beweis",
    "person": "Person",
    "party": "Partei",
    "canton": "Kanton",
    "similarity": "Sem. Ähnlichkeit",
    "model_name": "LLM",
}


def prepare_pipeline(model_name_or_path):
    """
    Initializes the OpinionAnalyzer and updates the model used for analysis.
    Parameters:
    model_name_or_path (str): The name or path of the model to be used for opinion analysis.
    Side Effects:
    - Updates the global 'analyzer_dict' with a new instance of OpinionAnalyzer.
    - Prints a message indicating the change of the model.
    """
    analyzer_dict["opinion_analyzer"] = OpinionAnalyzer(
        model_name_or_path=model_name_or_path
    )
    print(
        "Changing the model to: ", analyzer_dict["opinion_analyzer"].model_name_or_path
    )


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


# Function to perform argument mining and yield results incrementally
def perform_argument_mining(input_text: str, similarity_threshold: float = None):
    """
    Performs argument mining on the provided input text.
    Parameters:
    - input_text (str): The text to be analyzed for arguments.
    - similarity_threshold (float, optional): A threshold value for similarity to filter arguments.
    A dictionary is used to rename columns for better readability. An empty DataFrame is initialized to accumulate results. The method iteratively processes the text to find arguments using an opinion analyzer. If a cancellation signal is detected, the processing halts. Valid arguments are appended to the accumulated DataFrame, which is then periodically yielded for updates.
    The function handles:
    - Initializing a DataFrame with renamed columns.
    - Iterating through found arguments and checking for cancellation status.
    - Filtering arguments based on labels and appending to the accumulated DataFrame.
    - Yielding the accumulated DataFrame for updates.
    - Simulating processing time.
    """

    # Initialize an empty DataFrame to accumulate results
    accumulated_df = pd.DataFrame(
        columns=[new_col for old_col, new_col in COLUMN_RENAMING.items()]
    )
    accumulated_df = rename_columns(accumulated_df)
    # Simulate finding arguments incrementally for the sake of demonstration
    print("Using this model: ", analyzer_dict["opinion_analyzer"].model_name_or_path)
    arguments = analyzer_dict["opinion_analyzer"].find_arguments(
        topic_text=input_text, similarity_threshold=similarity_threshold
    )
    # For this example, simulate processing each argument one by one
    for row in arguments:
        print(f"Current cancel status: ", cancel_status["cancel"])
        if cancel_status["cancel"]:
            break  # Check for cancellation before processing each row
        if row["label"] in ["pro", "contra"]:
            current_row_df = pd.DataFrame([row])
            current_row_df = rename_columns(current_row_df)
            # Append the current row to the accumulated DataFrame
            accumulated_df = pd.concat(
                [accumulated_df, current_row_df], ignore_index=True
            )
            # Yield the accumulated DataFrame to update the Gradio output
            yield accumulated_df
            # Simulate processing time (remove or adjust as needed)
            time.sleep(0.5)


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
    print(cancel_status)


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
    file_path = "/tmp/output.xlsx"
    dataframe.to_excel(file_path, index=False)
    return file_path


# Gradio app interface with vertical layout
with gr.Blocks() as interface:
    gr.Markdown("# Argument Mining App")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Spezifiziere Einstellungen")
            dropdown = gr.Dropdown(choices=MODELS, label="Wähle ein Model aus")
            # Slider for similarity threshold
            similarity_threshold_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=config["thresholds"]["sentence_similarity"],
                step=0.01,
                label="Schwellenwert für die semantischen Ähnlichkeit",
            )
            hyperparam_button = gr.Button("Einstellungen sichern")
            hyperparam_button.click(
                fn=prepare_pipeline,  # Reset cancel status before starting
                inputs=[dropdown],
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
            # Button to trigger the analysis
            submit_button = gr.Button("Beginne Argumentensuche!")
            # Button to cancel the operation
            cancel_button = gr.Button("Abbrechen")
            # Output DataFrame
            output_table = gr.Dataframe(wrap=True)
            # Export button to save the DataFrame as an Excel file
            export_button = gr.Button("Exportieren als Excel")
            # Link input and output with the function
            submit_button.click(
                fn=setup_operation,
                inputs=[],
                outputs=[],
            )
            submit_button.click(
                fn=perform_argument_mining,  # Reset cancel status before starting
                inputs=[input_text, similarity_threshold_slider],
                outputs=output_table,
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
                outputs=[],
            )
            # Bind the export button
            export_button.click(
                fn=save_as_excel,
                inputs=[output_table],
                outputs=gr.File(label="Sichere Ergebnisse als Excel-Datei"),
            )
# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
