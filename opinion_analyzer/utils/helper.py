import os
import re
from ast import literal_eval
from pathlib import Path
from typing import Union, Literal
from itertools import tee
import pandas as pd
import spacy
import yaml
from PIL import Image
from spacy.tokens.doc import Doc
import numpy as np
import shutil
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from bs4 import BeautifulSoup
from transformers import EvalPrediction
import fitz

nlp = spacy.load("de_dep_news_trf")


def get_main_config() -> dict:
    """
    Reads the main configuration dictionary from a YAML file and updates the paths.

    The function reads the configuration from ``config.yaml`` located in the ``configs`` directory,
    relative to the script's parent directory. It converts paths in the ``paths`` section
    to absolute paths and ensures the ``document_name_component`` in the ``procurement_analyzer``
    section is a string, if it is not None.

    Returns:
        dict: The configuration dictionary with updated paths.
    """
    source_directory = Path(__file__).parent.parent
    path_to_config_file = source_directory / "configs" / "config.yaml"
    with open(path_to_config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    paths = config["paths"]
    for key, path in paths.items():
        os.makedirs(source_directory / path, exist_ok=True)
        paths[key] = source_directory / path
    models = config["models"]
    for key, path in models.items():
        if ".." in path:
            os.makedirs(source_directory / path, exist_ok=True)
            models[key] = source_directory / path
        else:
            models[key] = path
    return config


def has_values(iterator):
    """
    Check whether an iterator has any values and return a duplicate.

    :param iterator: An iterator to be checked for values.
    :type iterator: iterator
    :return: A tuple containing a boolean indicating if the iterator has values,
             and a duplicate of the original iterator.
    :rtype: tuple (bool, iterator)

    This function duplicates the given iterator and checks if the
    original iterator has any values by trying to get the next item.
    It then returns a boolean indicating the presence of values along
    with a duplicate iterator to continue iteration without exhausting
    the original.
    """
    it1, it2 = tee(iterator)  # Duplicate the iterator
    try:
        next(it1)  # Check if the first duplicate has any values
        return True, it2  # Return the second duplicate to continue iterating
    except StopIteration:
        return False, it2


def extract_text_from_pdf(pdf_file: str | Path) -> list[str]:
    """Goes through all the pages of a PDF file and extracts the text using OCR.

    :param pdf_file: The path to the PDF file
    :return: A list of strings representing the content of each page
    """
    content = []
    with fitz.open(pdf_file) as doc:
        for page in doc:
            content.append(page.get_text())
    return content


def remove_subfolders(directory):
    # Iterate through all the items in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # If the item is a directory, remove it
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the directory and its contents


def get_model_client_config() -> dict:
    """
    Returns the configuration dictionary that tells which client or pipeline to use for which model.

    Returns:
        dict: The configuration dictionary containing the model client configuration.
    """
    source_directory = Path(__file__).parent.parent
    path_to_config_file = source_directory / "configs" / "model_client_config.yaml"
    with open(path_to_config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def concatenate_images_vertically(
        images: Union[list[Image.Image], list[Union[str, os.PathLike]], str, os.PathLike]
) -> Image.Image:
    """
    Concatenate multiple images vertically.

    Args:
        images (Union[list[Image.Image], list[Union[str, os.PathLike]], str, os.PathLike]): List of PIL Image objects or list of image file paths.

    Returns:
        PIL.Image.Image: Concatenated image.

    Raises:
        ValueError: If the input list is empty.
        FileNotFoundError: If one or more image files are not found.

    Example:
        >>> image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        >>> concatenated_image = concatenate_images_vertically(image_paths)
        >>> concatenated_image.show()
        >>> image_objects = [Image.open('image1.jpg'), Image.open('image2.jpg'), Image.open('image3.jpg')]
        >>> concatenated_image = concatenate_images_vertically(image_objects)
        >>> concatenated_image.show()
    """
    if not images:
        raise ValueError("Input list cannot be empty")

    if isinstance(images, list) and all(isinstance(img, Image.Image) for img in images):
        images_to_concatenate = images
    elif isinstance(images, (os.PathLike, str)):
        if os.path.isfile(images):
            images = [Path(images)]
        else:
            images = [x for x in Path(images).rglob("*") if x.is_file()]
        images = sorted(images, key=lambda x: re.findall(r"\d+", x.name)[0])
        images_to_concatenate = [
            Image.open(img) for img in images if os.path.exists(img)
        ]
        if len(images_to_concatenate) != len(images):
            raise FileNotFoundError("One or more image files not found")
    elif all(isinstance(img, (str, os.PathLike)) for img in images):
        images_to_concatenate = [
            Image.open(img) for img in images if os.path.exists(img)
        ]
        if len(images_to_concatenate) != len(images):
            raise FileNotFoundError("One or more image files not found")
    else:
        raise ValueError(
            "Invalid input type. Input must be a list of PIL Image objects or file paths."
        )

    total_height = sum(img.height for img in images_to_concatenate)
    max_width = max(img.width for img in images_to_concatenate)

    output_img = Image.new("RGB", (max_width, total_height))

    y_offset = 0
    for img in images_to_concatenate:
        output_img.paste(img, (0, y_offset))
        y_offset += img.height

    return output_img


def extract_dictionary_from_string(sample_string: str) -> list[dict]:
    """
    Extracts a list of dictionaries from a string.

    Args:
        sample_string (str): The string containing the list of dictionaries.

    Returns:
        list[dict]: The list of dictionaries.
    """
    sample_string = re.sub(r"`", "", sample_string)
    sample_string = re.sub(r"null", "''", sample_string)
    sample_string = re.sub(r"-\n", "", sample_string)
    sample_string = re.sub(r"\n", " ", sample_string)
    sample_string = re.sub(r"\] \]", "]", sample_string)
    sample_string = re.sub(r"\]\]", "]", sample_string)

    regex = r"(\[[^\[\]]*\])"
    if len(re.findall(regex, sample_string)) > 0:
        sample_string = re.findall(regex, sample_string)[0]
    sample_string = sample_string.strip()
    if not sample_string.startswith("["):
        sample_string = "[" + sample_string
    if not sample_string.endswith("]"):
        sample_string = sample_string + "]"
    sample_string = literal_eval(sample_string)
    sample_string = [x for x in sample_string if isinstance(x, dict)]
    return sample_string


def make_spacy_doc(text: Union[str, Doc]) -> Doc:
    """
    Converts a given text into a spaCy Doc object.

    Args:
        text (Union[str, Doc]): Text to be converted.

    Returns:
        spacy.tokens.Doc: spaCy Doc object.
    """
    if isinstance(text, str):
        text = nlp(text)
    return text


def find_ambiguous_words(text: Union[str, Doc]) -> list[str]:
    """
    Finds ambiguous words in a given text.

    Args:
        text (Union[str, spacy.tokens.Doc]): The text to analyze.

    Returns:
        list[str]: List of ambiguous words in the text.
    """
    text = make_spacy_doc(text)
    ambiguous_words = []

    tags_to_ignore = [
        "attributive possessive pronoun",
        "non-reflexive personal pronoun",
    ]
    for w in text:
        if spacy.explain(w.tag_) not in tags_to_ignore:
            if (
                    re.search(r"(pronoun|pronominal)", spacy.explain(w.tag_))
                    and w.text not in ambiguous_words
            ):
                ambiguous_words.append(w.text)
    return ambiguous_words


def find_ambiguous_sentence(text: Union[str, Doc]) -> list[str]:
    """
    Finds sentences with ambiguous words in a given text.

    Args:
        text (Union[str, spacy.tokens.Doc]): The text to analyze.

    Returns:
        list[str]: List of sentences with ambiguous words.
    """
    text = make_spacy_doc(text)
    ambiguous_sentences = []

    for sent in text.sents:
        ambiguous_words = find_ambiguous_words(sent)
        if ambiguous_words:
            ambiguous_sentences.append(sent.text)
    return ambiguous_sentences


def make_sentences_concrete(
        text: Union[str, Doc],
        client_handler,
        expand_sentence: str,
        method: Literal["llm", "context"] = "llm",
        check_ambiguity: bool = False,
) -> list[str]:
    """
    Disambiguates and rewrites ambiguous sentences in a given text.

    Args:
        text (Union[str, spacy.tokens.Doc]): The text to analyze.
        client_handler: Client handler to generate new sentences.
        expand_sentence: The prompt to expand the sentence, see prompt database.

    Returns:
        list[str]: DataFrame containing original sentences and their disambiguated versions.
    """
    text = make_spacy_doc(text)
    ambiguous_sentences = find_ambiguous_sentence(text)

    print(f"{len(ambiguous_sentences)} ambiguous sentences.")
    # print("These are the ambiguous sentences:")
    # for s in ambiguous_sentences:
    # print(s)
    # print('+++++++++++++++++')

    all_sentences = [sent.text for sent in text.sents]
    new_sentences = []
    contexts = []
    for n, sent in enumerate(text.sents):
        if sent.text in ambiguous_sentences or check_ambiguity:
            ambiguous_words = ", ".join(find_ambiguous_words(sent))
            if method == "llm":
                context = " ".join(all_sentences[n - 20: n + 20])
                prompt = expand_sentence.format(
                    sentence=sent.text, context=context, ambiguous_words=ambiguous_words
                )
                new_sentence = client_handler.generate(prompt)
            elif method == "context":
                new_sentence = " ".join(all_sentences[n - 2: n + 2])
            elif method is None:
                new_sentence = sent.text
            else:
                raise f"No method {method} available. Choose between 'llm' or 'context'."
            new_sentences.append(new_sentence)
        else:
            new_sentences.append(sent.text)
        general_context = " ".join(all_sentences[n - 5: n + 5])
        contexts.append(general_context)

    results = [
        {
            "original_sentence": all_sentences[n],
            "new_sentence": new_sentences[n],
            "context": contexts[n],
        }
        for n, _ in enumerate(all_sentences)
    ]
    results = pd.DataFrame(results)
    return results


config = get_main_config()


def compute_metrics_multi_class(p: EvalPrediction, y_true=None, y_pred=None):
    if p is not None:
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        y_pred = np.argmax(logits, axis=1)
        y_true = p.label_ids

    macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(
        y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0
    )
    accuracy_not_normalized = accuracy_score(
        y_true=y_true, y_pred=y_pred, normalize=False
    )
    accuracy_normalized = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=True)
    precision_macro = precision_score(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    )
    precision_micro = precision_score(
        y_true=y_true, y_pred=y_pred, average="micro", zero_division=0
    )
    precision_weighted = precision_score(
        y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0
    )
    recall_score_macro = recall_score(
        y_true=y_true, y_pred=y_pred, average="macro", zero_division=0
    )
    recall_score_micro = recall_score(
        y_true=y_true, y_pred=y_pred, average="micro", zero_division=0
    )
    recall_score_weighted = recall_score(
        y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0
    )
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    return {
        "macro-f1": macro_f1,
        "micro-f1": micro_f1,
        "weighted-f1": weighted_f1,
        "matthews_correlation": mcc,
        "accuracy_normalized": accuracy_normalized,
        "accuracy_not_normalized": accuracy_not_normalized,
        "macro-precision": precision_macro,
        "micro-precision": precision_micro,
        "weighted-precision": precision_weighted,
        "macro-recall": recall_score_macro,
        "micro-recall": recall_score_micro,
        "weighted-recall": recall_score_weighted,
    }


def adjust_labels(label: str, score: float, threshold: float):
    if threshold is None:
        return str(label).lower()
    if score is None:
        return str(label).lower()
    if score >= threshold:
        return str(label).lower()
    return "Neutral"


def clean_text_for_excel(text: str) -> str:
    if not isinstance(text, str):
        return text

    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r" +", " ", text)
    return text


ILLEGAL_XLSX_CHARS = {
    "\x00": "\\x00",  # NULL
    "\x01": "\\x01",  # SOH
    "\x02": "\\x02",  # STX
    "\x03": "\\x03",  # ETX
    "\x04": "\\x04",  # EOT
    "\x05": "\\x05",  # ENQ
    "\x06": "\\x06",  # ACK
    "\x07": "\\x07",  # BELL
    "\x08": "\\x08",  # BS
    "\x0b": "\\x0b",  # VT
    "\x0c": "\\x0c",  # FF
    "\x0e": "\\x0e",  # SO
    "\x0f": "\\x0f",  # SI
    "\x10": "\\x10",  # DLE
    "\x11": "\\x11",  # DC1
    "\x12": "\\x12",  # DC2
    "\x13": "\\x13",  # DC3
    "\x14": "\\x14",  # DC4
    "\x15": "\\x15",  # NAK
    "\x16": "\\x16",  # SYN
    "\x17": "\\x17",  # ETB
    "\x18": "\\x18",  # CAN
    "\x19": "\\x19",  # EM
    "\x1a": "\\x1a",  # SUB
    "\x1b": "\\x1b",  # ESC
    "\x1c": "\\x1c",  # FS
    "\x1d": "\\x1d",  # GS
    "\x1e": "\\x1e",  # RS
    "\x1f": "\\x1f",  # US
}


def escape_xlsx_char(ch: str) -> str:
    """
    Escape illegal characters for XLSX format.
    :param ch: The character to escape.
    :type ch: str
    :return: The escaped character.
    :rtype: str
    """
    return ILLEGAL_XLSX_CHARS.get(ch, ch)


def escape_xlsx_string(st: str) -> str:
    """
    Escape all illegal characters in a string for XLSX format.
    :param st: The input string to escape.
    :type st: str
    :return: The escaped string.
    :rtype: str
    """
    if not isinstance(st, str):
        return st

    return "".join([escape_xlsx_char(ch) for ch in st])
