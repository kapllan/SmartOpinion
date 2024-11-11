import gradio as gr
import pandas as pd
from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.utils.helper import get_main_config
from pprint import pprint
import pandas as pd
import time
import threading  # Import threading

# Get configuration
config = get_main_config()

MODELS = [
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

opinion_analyzer = OpinionAnalyzer(
    model_name_or_path=config["models"]["llm"], model_client="together"
)

analyzer_dict = {"opinion_analyzer": opinion_analyzer}

# Shared state for cancellation
cancel_status = {"cancel": False}


def prepare_pipeline(model_name_or_path):
    """
    Prepare the opinion analyzer pipeline with specified model.

    Args:
        model_name_or_path (str): The path or name of the model to load into the OpinionAnalyzer.

    Side Effects:
        Updates the global analyzer_dict with an instance of OpinionAnalyzer using the specified model.

    Prints:
        Confirmation message with the name of the model being loaded into the analyzer.
    """
    analyzer_dict["opinion_analyzer"] = OpinionAnalyzer(
        model_name_or_path=model_name_or_path, model_client="together"
    )
    print(
        "Changing the model to: ", analyzer_dict["opinion_analyzer"].model_name_or_path
    )


def rename_columns(dataframe: pd.DataFrame, columns_renamings: dict) -> pd.DataFrame:
    """
    Renames and reorders columns in a given DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame whose columns need to be renamed.
        columns_renamings (dict): A dictionary where keys are the current column names
        and values are the new column names.

    Returns:
        pd.DataFrame: DataFrame with renamed and reordered columns.
    """
    dataframe.rename(columns=columns_renamings, inplace=True)
    dataframe = dataframe[[new_col for old_col, new_col in columns_renamings.items()]]
    return dataframe


# Function to perform argument mining and yield results incrementally
def perform_argument_mining(input_text: str, similarity_threshold: float = None):
    """
    Performs argument mining on the given input text, finding and processing arguments
    based on the specified similarity threshold.

    Args:
        input_text (str): The input text to analyze for arguments.
        similarity_threshold (float, optional): The threshold for semantic similarity. Defaults to None.

    Initializes an empty DataFrame with renamed columns to accumulate results.
    Simulates finding arguments incrementally and processes each argument.
    Interrupts processing if cancel status is set to True.
    Yields accumulated DataFrame for update after processing each valid argument.
    """
    columns_renamings = {
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

    # Initialize an empty DataFrame to accumulate results
    accumulated_df = pd.DataFrame(
        columns=[new_col for old_col, new_col in columns_renamings.items()]
    )
    accumulated_df = rename_columns(accumulated_df, columns_renamings)
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
            current_row_df = rename_columns(current_row_df, columns_renamings)
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

    reset_output initializes and returns an empty pandas DataFrame with predefined columns.

    Returns:
        pd.DataFrame: A DataFrame with specified columns for storing various argument details.
    """
    return pd.DataFrame(
        columns=[
            "label",
            "argument_rewritten",
            "argument_original",
            "argument_reason",
            "reasoning",
            "reasoning_segment",
            "person",
            "party",
            "canton",
            "similarity",
        ]
    )


# Function to set the cancel flag
def cancel_operation():
    """

    Cancel the current operation by setting the cancel_status flag to True and printing the status.

    This function is used to signal that an ongoing operation should be cancelled by modifying a shared cancel_status dictionary. After setting the flag, it prints the current state of the cancel_status.
    """
    cancel_status["cancel"] = True
    print(cancel_status)


# Function to clear the cancel flag and reset
def setup_operation():
    """
    Sets up the initial operation state by resetting the cancel status.

    - Initializes the cancel status dictionary
    - Sets the 'cancel' flag to False

    Parameters:
        None

    Returns:
        None
    """
    cancel_status["cancel"] = False


# Gradio app interface with vertical layout
with gr.Blocks() as interface:
    gr.Markdown("# Argument Mining App")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Spezifiziere Hyperparameter.")

            dropdown = gr.Dropdown(choices=MODELS, label="Wähle ein Model aus")
            # Slider for similarity threshold
            similarity_threshold_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=config["thresholds"]["sentence_similarity"],
                step=0.01,
                label="Wert der semantischen Ähnlichkeit",
            )

            hyperparam_button = gr.Button("Festlegen")
            hyperparam_button.click(
                fn=prepare_pipeline,  # Reset cancel status before starting
                inputs=[dropdown],
                outputs=[],
            )

    with gr.Row():
        with gr.Column():
            # Input text box
            gr.Markdown("## Suche nach Pro- und Contra-Argumenten.")
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
# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)
