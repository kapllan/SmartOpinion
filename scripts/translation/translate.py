from transformers import MBartForConditionalGeneration, MBart50Tokenizer, pipeline
from opinion_analyzer.utils.helper import get_main_config
import pandas as pd
from tqdm import tqdm
from opinion_analyzer.inference.client_handler import ClientHandler

tqdm.pandas()

"""# Load the model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Specify source and target languages
src_lang = "en_XX"  # Source language (English)
tgt_lang = "de_DE"  # Target language (French)

# Create a translation pipeline
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Text to translate
text = "A cure or treatment may be discovered shortly after having ended someone's life unnecessarily."

# Tokenizer needs to know source and target language
tokenizer.src_lang = src_lang

# Translate the text
translated = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)

# Print the result
print(translated[0]['translation_text'])"""


def make_first_char_capital(text: str) -> str:
    return text[0].upper() + text[1:]


def llm_translate(text: str, llm_pipeline) -> str:
    prompt = """
                Übersetze folgenden Text ins Deutsche. Sag sonst nichts weiter.
                
                Hier ist der Text: {text}
            
            """

    return llm_pipeline.generate(prompt.format(text=text))


if __name__ == "__main__":
    # Load your dataset (CSV)

    config = get_main_config()

    ch = ClientHandler("VAGOsolutions/Llama-3.1-SauerkrautLM-70b-Instruct")

    # Load the model and tokenizer
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    # Specify source and target languages
    src_lang = "en_XX"  # Source language (English)
    tgt_lang = "de_DE"  # Target language (French)

    # Create a translation pipeline
    translator = pipeline("translation", model=model, tokenizer=tokenizer)

    # Tokenizer needs to know source and target language
    tokenizer.src_lang = src_lang

    argument_training_dataset_path = config["paths"]["datasets"] / "IBM_Debater_(R)_XArgMining" / "Machine Translations"
    df = pd.read_csv(argument_training_dataset_path / "Arguments_6L_MT.csv")
    df["topic_EN"] = df["topic_EN"].progress_apply(make_first_char_capital)
    df["argument_EN"] = df["argument_EN"].progress_apply(make_first_char_capital)
    """df["topic_DE_own"] = df["topic_EN"].progress_apply(
        lambda x: translator(x, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    )
    df["argument_DE_own"] = (df["argument_EN"].progress_apply(
        lambda x: translator(x, src_lang=src_lang, tgt_lang=tgt_lang)[0]["translation_text"]
    ))"""

    df["topic_DE_own"] = df["topic_EN"].progress_apply(lambda x: llm_translate(x, ch)
                                                       )
    df["argument_DE_own"] = (df["argument_EN"].progress_apply(lambda x: llm_translate(x, ch)))

    df.to_csv(argument_training_dataset_path / "Arguments_6L_MT_own_translations.csv", index=False)
