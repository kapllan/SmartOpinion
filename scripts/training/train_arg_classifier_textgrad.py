# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import textgrad

# Step 1: Load the VAGOsolutions model and tokenizer using Hugging Face
model_name = "VAGOsolutions/SauerkrautLM-Phi-3-medium"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 2: Initialize the TextGrad optimizer
optimizer = textgrad.optimizer(model=model, tokenizer=tokenizer)

# Step 3: Define a base prompt template
prompt_template = """
Das Thema ist: {topic}
Das Argument lautet: {argument}
Was ist die Haltung dieses Arguments zum Thema? Antworten Sie mit "1" für Pro und "-1" für Contra.
"""

# Step 4: Define example inputs for prompt generation
examples = [
    {
        "topic": "Klimawandel",
        "argument": "Die CO2-Emissionen müssen reduziert werden, um das Klima zu schützen.",
        "expected_output": "1",
    },
    {
        "topic": "Klimawandel",
        "argument": "CO2-Reduktion ist zu teuer und beeinträchtigt die Wirtschaft.",
        "expected_output": "-1",
    },
]

# Step 5: Generate and optimize prompts using TextGrad
optimized_prompts = []

for example in examples:
    # Create the initial prompt using the template and the example
    prompt = prompt_template.format(
        topic=example["topic"], argument=example["argument"]
    )

    # Optimize the prompt using TextGrad based on the expected output
    optimized_prompt = optimizer.optimize(
        prompt, example["expected_output"], max_iterations=5
    )

    # Store the optimized prompt
    optimized_prompts.append(optimized_prompt)


# Step 6: Function to get model's response for the given prompt
def get_response(prompt, tokenizer, model, max_length=100):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the response using the model
    outputs = model.generate(
        inputs["input_ids"], max_length=max_length, pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens back into text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Step 7: Run the optimized prompts through the model and print the responses
for prompt in optimized_prompts:
    response = get_response(prompt, tokenizer, model)
    print(f"Optimized Prompt: {prompt}")
    print(f"Model's Response: {response}")
    print("\n")
