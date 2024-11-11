import gradio as gr
import pandas as pd
from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.utils.helper import get_main_config
from pprint import pprint
import time
import threading  # Import threading

# Get configuration
config = get_main_config()

opinion_analyzer = OpinionAnalyzer(
    model_name_or_path=config["models"]["llm"], model_client="together"
)

# Shared state for cancellation
cancel_status = {"cancel": False}


# Function to perform argument mining and yield results incrementally
def perform_argument_mining(input_text: str, similarity_threshold: float = None):
    columns = [
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
    # Initialize an empty DataFrame to accumulate results
    accumulated_df = pd.DataFrame(columns=columns)
    # Simulate finding arguments incrementally for the sake of demonstration
    arguments = opinion_analyzer.find_arguments(
        topic_text=input_text, similarity_threshold=similarity_threshold
    )
    # For this example, simulate processing each argument one by one
    for row in arguments:
        print(f"Current cancel status: ", cancel_status["cancel"])
        if cancel_status["cancel"]:
            break  # Check for cancellation before processing each row

        if row["label"] in ["pro", "contra"]:
            current_row_df = pd.DataFrame([row])[columns]
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
    cancel_status["cancel"] = True
    print(cancel_status)


# Function to clear the cancel flag and reset
def setup_operation():
    cancel_status["cancel"] = False


# Gradio app interface with vertical layout
with gr.Blocks() as interface:
    gr.Markdown("# Argument Mining App")
    gr.Markdown("Enter a topic and get a table with mock claims and evidence.")
    # Input text box
    input_text = gr.Textbox(lines=2, placeholder="Enter a topic or argument text...")
    # Slider for similarity threshold
    similarity_threshold_slider = gr.Slider(
        minimum=0,
        maximum=1,
        value=config["thresholds"]["sentence_similarity"],
        step=0.01,
        label="Similarity Threshold",
    )
    # Button to trigger the analysis
    submit_button = gr.Button("Submit")
    # Button to cancel the operation
    cancel_button = gr.Button("Cancel")
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
