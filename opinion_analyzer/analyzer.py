import argparse
import os
import re
from pathlib import Path
from pprint import pprint

import spacy
from bs4 import BeautifulSoup
from openpyxl.styles.builtins import output
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

import pandas as pd
from opinion_analyzer.data_handler.prompt_database import prompt_dict
from opinion_analyzer.inference.client_handler import ClientHandler
from opinion_analyzer.utils.helper import (
    extract_dictionary_from_string,
    get_main_config,
    make_sentences_concrete,
    adjust_labels,
    extract_text_from_pdf,
    clean_text_for_excel,
    escape_xlsx_string)

from typing import Literal

nlp = spacy.load("de_dep_news_trf")

config = get_main_config()


# log = get_logger()


class OpinionAnalyzer(ClientHandler):
    stance_classifier = pipeline("text-classification",
                                 model=config["models"]["stance_classifier"])

    stance_class_threshold = 0.9
    """def __init__(self):
        # Call the parent class's __init__ method
        super().__init__()
        self.stance_classifier = pipeline("text-classification",
                                          model=config["models"]["stance_classifier"])

        self.stance_class_threshold = 0.5"""

    def segment_argument_text(self, text: str, prompt: str = None):
        if prompt is None:
            prompt = prompt_dict["segment_argument_text"]
        prompt = prompt.format(
            argument_text=text,
            example_json_argument_segments=prompt_dict[
                "example_json_argument_segments"
            ],
        )

        output = self.generate(prompt=prompt)

        return extract_dictionary_from_string(output)

    def get_stance(self, topic_text: str, text_sample: str):

        text = topic_text + f" {self.stance_classifier.tokenizer.sep_token} " + text_sample
        result = self.stance_classifier(text)[0]

        return result

    def is_argument(self, topic_text: str, text_sample: str):

        prompt = prompt_dict["is_argument"].format(topic_text=topic_text, text_sample=text_sample)

        result = self.generate(prompt=prompt)

        return result

    def categorize_argument(self, topic_text: str, text_sample: str, prompt: str = None,
                            method: Literal["llm", "finetuned"] = None):

        """is_argument = self.is_argument(topic_text, text_sample)

        if is_argument.lower().strip() == "ja":"""
        if prompt is None:
            prompt = prompt_dict["categorize_argument_few_shot"]
        prompt = prompt.format(
            topic_text=topic_text,
            text_sample=text_sample,
        )

        if method in [None, "llm"]:
            label = self.generate(prompt=prompt)
            label = label.strip().lower()
            if label in ["pro", "contra", "neutral"]:
                output = {"label": label, "score": None}
            elif label.startswith("pro") or label.startswith("contra") or label.startswith("neutral"):
                output = {
                    "label": label.split(" ")[0],
                    "score": None
                }
            elif label.endswith("pro") or label.endswith("contra") or label.endswith("neutral"):
                output = {
                    "label": label.split(" ")[-1],
                    "score": None
                }
            else:
                output = {
                    "label": "neutral",
                    "score": None
                }

        elif method == "finetuned":
            output = self.get_stance(topic_text, text_sample)
        else:
            raise f"Method {method} does not exist. Choose between 'llm' or 'finetuned'."

        return output

    def is_debatable(self, text: str) -> str:
        prompt = prompt_dict["is_debatable"]
        prompt = prompt.format(
            text=text
        )
        return self.generate(prompt=prompt)

    def extract_arguments(
            self, topic_text: str, argument_text: str, prompt: str = None
    ):

        if prompt is None:
            prompt = prompt_dict["extract_arguments_1"]
        prompt = prompt.format(
            topic_text=topic_text,
            argument_text=argument_text,
            # example_json_arguments=prompt_dict["example_json_arguments"],
        )

        return self.generate(prompt=prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        default=None,
        type=str,
        help="Specify the model name.",
    )

    parser.add_argument(
        "-m",
        "--method",
        type=str,
        choices=["llm", "finetuned"],
        help="Specify if you want to use LLMs or a fine-tuned model.",
    )

    parser.add_argument(
        "-rw",
        "--rewrite",
        action="store_true",
        help="Decide if you want to rewrite each sentence or not. Defaults to False.",
    )

    parser.add_argument(
        "-bi",
        "--business_id",
        type=int,
        help="Decide which business you want to analyze.",
    )

    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="Proceedings.txt",
        help="Specify the file name you want to analyze.",
    )

    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        default=None,
        help="Specify the topic.",
    )

    parser.add_argument(
        "-dc",
        "--debatability_check",
        action="store_true",
        help="Check if a given text is debatable or not. Defaults to False.",
    )

    args = parser.parse_args()

    args.business_id = str(args.business_id)

    sent_model = SentenceTransformer(config["models"]["sentence_similarity"])

    opinion_analyzer = OpinionAnalyzer(args.model_name_or_path)

    results = []

    if args.topic is not None:
        topic_text = args.topic
    else:
        topic_text = open(config["paths"]["data"] / "referendums" / args.business_id / "InitialSituation.txt").read()
        topic_text = BeautifulSoup(topic_text, features="lxml").text

    path_to_argument_file = config["paths"]["data"] / "referendums" / args.business_id / args.file
    if path_to_argument_file.name.endswith("txt"):
        argument_text = open(config["paths"]["data"] / "referendums" / args.business_id / args.file).read()
        argument_text = BeautifulSoup(argument_text, features="lxml").text
    elif path_to_argument_file.name.endswith("pdf"):
        argument_text = " ".join(extract_text_from_pdf(path_to_argument_file))

    if args.rewrite:
        topic_text_segmented = make_sentences_concrete(text=topic_text,
                                                       client_handler=opinion_analyzer,
                                                       expand_sentence=prompt_dict["expand_sentence"]).new_sentence.tolist()
        argument_text_segmented = make_sentences_concrete(text=argument_text,
                                                          client_handler=opinion_analyzer,
                                                          expand_sentence=prompt_dict["expand_sentence"]).new_sentence.tolist()
    else:
        if len(topic_text) > 500:
            topic_text_segmented = [sent.text for sent in nlp(topic_text).sents if len(sent.text) > 10]
        else:
            topic_text_segmented = [topic_text]
        if len(argument_text) > 500:
            argument_text_segmented = [sent.text for sent in nlp(argument_text).sents if len(sent.text) > 10]
        else:
            argument_text_segmented = [argument_text]

    topic_text_segmented = [sent for sent in topic_text_segmented if
                            opinion_analyzer.is_debatable(sent).lower().strip() == "ja"]
    argument_text_segmented_emb = sent_model.encode(argument_text_segmented)

    is_argument_list = []
    # is_debatable_set = {}

    for segment in topic_text_segmented:
        segment_emb = sent_model.encode(segment)
        print("Extracting the arguments for the following topic:\n")
        print(segment)
        print("--------")
        results_semantic_search = util.semantic_search(segment_emb, argument_text_segmented_emb)[0]
        results_semantic_search = [x for x in results_semantic_search if x["score"] > 0.8]
        for idx, sent in enumerate(argument_text_segmented):
            if idx in [x["corpus_id"] for x in results_semantic_search]:
                # TODO: Add a prompt to distinguish whether something is a Beschluss or not
                # TODO: Add a prompt to distinguish whether something could be an argument or not.
                if args.debatability_check:
                    debatable = opinion_analyzer.is_debatable(sent).lower().strip() == "ja"
                else:
                    debatable = "ja"
                if debatable == "ja":
                    # if [segment, sent] in is_argument_list or opinion_analyzer.is_argument(topic_text=segment, text_sample=sent).lower().strip() == "ja":
                    # is_argument_list.append([segment, sent])
                    result = opinion_analyzer.categorize_argument(
                        topic_text=segment, text_sample=sent, method=args.method
                    )
                    label = adjust_labels(label=result["label"], score=result["score"],
                                          threshold=opinion_analyzer.stance_class_threshold)

                    entry = {
                        "topic": segment,
                        "text_sample": sent,
                        "label": label,
                        "similarity": [x for x in results_semantic_search if x["corpus_id"] == idx][0]["score"],
                    }
                    results.append(entry)
                    pprint(entry)

    results = pd.DataFrame(results)
    results = results.applymap(escape_xlsx_string)
    output_dir = Path("argument_mining_results")
    os.makedirs(output_dir, exist_ok=True)
    if args.method == "finetuned":
        model_name = Path(opinion_analyzer.stance_classifier.model.name_or_path).name.replace("/", "_")
    else:
        model_name = args.model_name_or_path.replace("/", "_")
    results.to_excel(output_dir / f"argument_analysis__business_id_{args.business_id}__{args.file}__{model_name}.xlsx",
                     index=False)
