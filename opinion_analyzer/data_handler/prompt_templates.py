""" This file contains prompt templates for the LLMs. """

template_dict = {
    "prompt_template_1": "<s> [INST] {prompt} [/INST]",
    "prompt_template_2": "<|input|>\n{prompt}<|output|>\n",
    "prompt_template_3": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                            {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            """,
}
