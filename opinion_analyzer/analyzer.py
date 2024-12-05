# Pipeline to extract arguments from referendums.


import argparse
import datetime
import os
from pathlib import Path
from pprint import pprint
from typing import Literal
from ast import literal_eval
import chromadb
import pandas as pd
import spacy
import torch
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from difflib import SequenceMatcher

from opinion_analyzer.data_handler.prompt_database import prompt_dict
from opinion_analyzer.inference.client_handler import ClientHandler
from opinion_analyzer.utils.helper import (
    extract_dictionary_from_string,
    get_main_config,
    make_sentences_concrete,
    adjust_labels,
    extract_text_from_pdf,
    escape_xlsx_string,
)
from opinion_analyzer.utils.log import get_logger


nlp = spacy.load("de_dep_news_trf")

config = get_main_config()

log = get_logger()


def find_matching_sentence(string: str, context: str) -> str:
    """
    Find the sentence in the context that is most similar to the given string.

    This function splits the context into sentences and uses a simple similarity
    measurement (Levenshtein-like) to find the most similar sentence.

    :param string: The sentence to compare against the context.
    :type string: str
    :param context: The block of text from which to find the most similar sentence.
    :type context: str
    :return: The sentence from the context that is most similar to the string.
    :rtype: str

    :Example:

    ::

        context = \"\"\"
        Python is an interpreted, high-level and general-purpose programming language.
        Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
        \"\"\"

        sentence = "Python is easy to read."
        result = find_matching_sentence(sentence, context)
        print(result)
    """
    # Tokenize the context into sentences
    sentences = [sent.text for sent in nlp(context).sents]

    # Initialize variables to store the best match
    highest_similarity = -1
    best_match = ""

    for sentence in sentences:
        # Calculate a simple similarity ratio using SequenceMatcher
        similarity = SequenceMatcher(None, string, sentence).ratio()

        # Update best match if we find a higher similarity score
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = sentence

    return best_match


def prepare_documents(
        text: Path | str, method: Literal["llm", "context", None], client_handler
) -> list[dict]:
    """
    Prepare documents for text processing.

    This function handles text extraction from different types of documents and processes the text
    according to the specified method. The current implementation supports text files (.txt) and
    PDF documents (.pdf).

    :param path: The file path of the document to be processed.
    :type path: Path
    :param method: The method to be used for processing the text. It can be "llm", "context", or None.
                   Defaults to "llm".
    :type method: Literal["llm", "context", None], optional
    :param client_handler: The client handler object used for processing the text.
    :type client_handler: ClientHandler

    :raises ValueError: If the document extension is not supported (i.e., not .txt or .pdf).

    :return: A list of dictionaries containing the processed and expanded text.
    :rtype: list[dict]
    """
    if os.path.exists(text):
        text = Path(text)
        if text.name.endswith("txt"):
            text = open(text).read()
            text = BeautifulSoup(text, features="lxml").text
        elif text.name.endswith("pdf"):
            text = " ".join(extract_text_from_pdf(text))
        else:
            raise ValueError(f"Can handle only PDF or txt documents.")

    text_expended_context = make_sentences_concrete(
        text=text,
        client_handler=client_handler,
        expand_sentence=prompt_dict[config["prompts"]["make_sentence_concrete"]],
        method=method,
    )
    return text_expended_context


class OpinionAnalyzer(ClientHandler):
    stance_classifier = pipeline(
        "text-classification", model=config["models"]["stance_classifier"]
    )

    stance_class_threshold = 0.9

    def __init__(
            self,
            model_name_or_path: str = None,
            tokenizer_model: str = None,
            model_client: Literal["together", "openai"] = None,
    ):
        # Call the parent class's __init__ method
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_model=tokenizer_model,
            model_client=model_client,
        )

        self.stance_classifier = pipeline(
            "text-classification", model=config["models"]["stance_classifier"]
        )

        self.stance_class_threshold = config["thresholds"]["stance_classification"]

        self.sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config["models"]["sentence_similarity"]
            )
        )

        self.chroma_client = chromadb.PersistentClient(
            str(config["paths"]["data"] / config["database"]["chromadb"])
        )

        self.existing_collections = self.chroma_client.list_collections()

        if len(self.existing_collections) > 0 and config["database"]["chromadb"] in [
            ec.name for ec in self.existing_collections
        ]:
            self.collection = self.chroma_client.get_collection(
                config["database"]["chromadb"],
                embedding_function=self.sentence_transformer_ef,
            )

        self.sent_model = SentenceTransformer(config["models"]["sentence_similarity"])

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

        text = (
                topic_text + f" {self.stance_classifier.tokenizer.sep_token} " + text_sample
        )
        result = self.stance_classifier(text)[0]

        return result

    def is_argument(self, topic_text: str, text_sample: str):

        prompt = prompt_dict["is_argument"].format(
            topic_text=topic_text, text_sample=text_sample
        )

        result = self.generate(prompt=prompt)

        return result

    def categorize_argument(
            self,
            topic_text: str,
            text_sample: str,
            prompt: str = None,
            method: Literal["llm", "finetuned"] = None,
    ):
        """is_argument = self.is_argument(topic_text, text_sample)

        if is_argument.lower().strip() == "ja":"""

        if prompt is None:
            prompt = prompt_dict[config["prompts"]["stance_detection"]]

        prompt = prompt.format(
            topic_text=topic_text,
            text_sample=text_sample,
        )

        if method in [None, "llm"]:
            model_generation = self.generate(prompt=prompt)
            label = model_generation.strip().lower()
            if label in ["pro", "contra", "neutral"]:
                output = {"label": label, "score": None}
            elif (
                    label.startswith("pro")
                    or label.startswith("contra")
                    or label.startswith("neutral")
            ):
                output = {"label": label.split(" ")[0], "score": None}
            elif (
                    label.endswith("pro")
                    or label.endswith("contra")
                    or label.endswith("neutral")
            ):
                output = {"label": label.split(" ")[-1], "score": None}
            else:
                output = {"label": "neutral", "score": None}
            output["model_generation"] = model_generation

        elif method == "finetuned":
            output = self.get_stance(topic_text, text_sample)
            output["score"] = round(output["score"], 2)
            output["model_generation"] = output["score"]

        else:
            raise f"Method {method} does not exist. Choose between 'llm' or 'finetuned'."

        return output

    def extract_evidence(self, topic: str, claim: str, stance: str, context: str):
        """
        Extracts evidence based on a given topic, claim, stance, and context. Uses a configured
        prompt to generate evidence and attempts to parse it.

        :param topic: The topic related to the claim.
        :type topic: str
        :param claim: The claim for which evidence is being sought.
        :type claim: str
        :param stance: The stance towards the claim.
        :type stance: str
        :param context: Additional context to aid evidence extraction.
        :type context: str

        :return: A dictionary containing the extracted reasoning segment and reasoning.
        :rtype: dict

        :raises SyntaxError: If the generated evidence cannot be parsed into a dictionary.

        This function logs errors when the generated output does not contain the expected fields,
        'reasoning_segment' and 'reasoning'.
        """

        if config["prompts"]["find_reasoning"] == "find_reasoning_3":
            evidence = self.generate(
                prompt=prompt_dict[config["prompts"]["find_reasoning"]].format(
                    topic=topic,
                    claim=claim,
                    stance=stance,
                    context=context
                )
            )
        elif config["prompts"]["find_reasoning"] == "find_reasoning_4":
            evidence = self.generate(
                prompt=prompt_dict[config["prompts"]["find_reasoning"]].format(
                    topic=topic,
                    # claim=claim,
                    stance=stance,
                    context=context
                )
            )
        try:
            evidence = literal_eval(evidence)
        except SyntaxError as se:
            print(f"SyntaxError: {se}")
            log.error(se)
            evidence = {"reasoning_segment": "", "reasoning": ""}
        if "reasoning_segment" not in evidence.keys():
            log.error("No field 'reasoning_segment' in reasoning output")
        if "reasoning" not in evidence.keys():
            log.error("No field 'reasoning' in reasoning output")
        reasoning_segment = find_matching_sentence(string=evidence["reasoning_segment"], context=context)
        evidence["reasoning_segment"] = reasoning_segment
        return evidence

    def is_debatable(self, text: str) -> str:
        prompt = prompt_dict["is_debatable"]
        prompt = prompt.format(text=text)
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

    def convert_to_sub_dicts(self, data_dict):
        """
        Convert the given dictionary to a list of sub-dictionaries.

        Args:
            data_dict (dict): The input dictionary containing lists of data.

        Returns:
            list: A list of sub-dictionaries with 'id', 'document', 'metadata', and 'distance'.
        """
        # Extracting lists from the dictionary
        ids = data_dict["ids"][0]
        documents = data_dict["documents"][0]
        metadatas = data_dict["metadatas"][0]
        distances = data_dict["distances"][0]

        # Creating a list of sub-dictionaries
        sub_dicts = [
            {"id": id_, "document": doc, "metadata": meta, "distance": dist}
            for id_, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]

        return sub_dicts

    def get_unique_field_values(self, field_name: str, collection_field: str = "metadatas"):
        """
        Retrieve all unique values of a specified field from a ChromaDB collection.

        This function extracts unique values from a specified field within a ChromaDB collection.

        :param collection: The ChromaDB collection object.
        :type collection: chromadb.Collection
        :param field_name: The name of the field for which unique values are to be retrieved.
        :type field_name: str
        :return: A set containing all unique values of the specified field.
        :rtype: set

        :raises ValueError: If the specified field is not present in the documents.
        :raises Exception: If there is an error in retrieving documents from the collection.
        """
        try:
            all_documents = self.collection.get(include=["documents", "metadatas", "embeddings"])
        except Exception as e:
            raise Exception(f"Error retrieving documents from the collection: {e}")

        if not all_documents or 'documents' not in all_documents:
            raise ValueError("No documents retrieved from the collection.")

        unique_values = set()

        for document in all_documents[collection_field]:
            if field_name in document:
                unique_values.add(document[field_name])
            else:
                raise ValueError(f"Field '{field_name}' not found in the document.")

        return [config["app"]["place_hold_all_sources"]] + sorted(list(unique_values))

    def find_matches(self, query: str, similarity_threshold) -> list[dict]:
        matches = self.collection.query(query_texts=[query], n_results=1000)
        matches = self.convert_to_sub_dicts(matches)
        matches = [m for m in matches if 1 - m["distance"] >= similarity_threshold]
        return matches

    def prepare_argument_text(self, query: str, similarity_threshold: float = None):

        matches = self.find_matches(
            query=query, similarity_threshold=similarity_threshold
        )

        results = []

        for doc in matches:
            results.append(
                {
                    "new_sentence": doc["document"],
                    "original_sentence": doc["metadata"]["original_sentence"],
                    "context": doc["metadata"]["context"],
                    "similarity": round(1 - doc["distance"], 2),
                    "business_id": doc["metadata"]["business_id"]
                }
            )

        return results

    def find_arguments(
            self, topic_text: str, rewrite=True, similarity_threshold: float = None,
            allowed_business_ids: list[str] = [config["app"]["place_hold_all_sources"]],
            precheck: bool = False, method: Literal["llm", "finetuned"] = "finetuned",
    ) -> list[dict]:

        if rewrite:
            print("Rewriting topic")
            topic_text_prepared = prepare_documents(
                text=topic_text,
                method="llm",
                client_handler=super(),
            )

        for topic_entry in topic_text_prepared.to_dict(orient="records"):
            semantic_search_results = self.prepare_argument_text(
                query=topic_entry["new_sentence"],
                similarity_threshold=similarity_threshold,
            )

            for argument_entry in semantic_search_results:
                if config["app"]["place_hold_all_sources"] in allowed_business_ids or argument_entry[
                    "business_id"] in allowed_business_ids:

                    if precheck or method == "finetuned":
                        stance_precheck = self.categorize_argument(
                            topic_text=topic_entry["new_sentence"],
                            text_sample=argument_entry["new_sentence"],
                            method="finetuned",
                        )

                    if (precheck and stance_precheck["score"] >= self.stance_class_threshold) or not precheck:

                        if method == "finetuned":
                            result = stance_precheck
                        elif method == "llm":
                            result = self.categorize_argument(
                                topic_text=topic_entry["new_sentence"],
                                text_sample=argument_entry["new_sentence"],
                                method="llm",
                            )
                        else:
                            raise "You can choose only between the following methods: 'llm' or 'finetuned'."

                        stance_label = adjust_labels(
                            label=result["label"],
                            score=result["score"],
                            threshold=self.stance_class_threshold,
                        )

                        evidence = self.extract_evidence(
                            topic=topic_entry["new_sentence"],
                            claim=argument_entry["new_sentence"],
                            stance=stance_label,
                            context=argument_entry["context"])

                        person_information = {"person": "", "party": "", "canton": ""}
                        if stance_label in ["pro", "contra"]:
                            person_information = self.generate(
                                prompt=prompt_dict[config["prompts"]["extract_person"]].format(
                                    # topic=segment,
                                    sentence=argument_entry["original_sentence"],
                                    # stance=stance,
                                    context=argument_entry["context"],
                                )
                            )

                            try:
                                person_information = literal_eval(person_information)
                            except SyntaxError as se:
                                print(f"SyntaxError: {se}")
                                person_information = {"person": "", "party": "", "canton": ""}

                        entry = {
                            "topic_original": topic_entry["original_sentence"],
                            "topic_rewritten": topic_entry["new_sentence"],
                            "argument_rewritten": argument_entry["new_sentence"],
                            "argument_original": argument_entry["original_sentence"],
                            "argument_reason": result["model_generation"],
                            "person": person_information["person"],
                            "party": person_information["party"],
                            "canton": person_information["canton"],
                            "context": argument_entry["context"],
                            "label": stance_label,
                            "score": result["score"],
                            "reasoning": evidence["reasoning"] if "reasoning" in evidence else "",
                            "reasoning_segment": (
                                evidence["reasoning_segment"]
                                if "reasoning_segment" in evidence
                                else ""
                            ),
                            "similarity": argument_entry["similarity"],
                            "model_name": self.model_name_or_path,
                            "num_matches": len(semantic_search_results),
                            "business_id": argument_entry["business_id"]
                        }

                        pprint(entry)

                        yield entry


if __name__ == "__main__":

    from ast import literal_eval

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        default=config["models"]["llm"],
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

    parser.add_argument(
        "-ts",
        "--timestamp",
        default=None,
        help="Specify the timestamp to create a subfolder where the results are stored. Defaults to None. If None, the current timestamp is used.",
    )

    parser.add_argument(
        "-cn",
        "--collection_name",
        type=str,
        default="smart_opinion_default_collection",
        help="Specify the name of the collection.",
    )

    args = parser.parse_args()

    if args.collection_name is not None and args.topic is None:
        raise "If you select a chromadb collection you need to pass the topic as a text."

    args.business_id = str(args.business_id)

    sent_model = SentenceTransformer(config["models"]["sentence_similarity"])

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config["models"]["sentence_similarity"]
    )

    opinion_analyzer = OpinionAnalyzer(args.model_name_or_path)

    # Initialize Chroma
    chroma_client = chromadb.PersistentClient(
        str(config["paths"]["data"] / args.collection_name)
    )

    existing_collections = chroma_client.list_collections()
    if len(existing_collections) > 0 and args.collection_name in [
        ec.name for ec in existing_collections
    ]:
        collection = chroma_client.get_collection(
            args.collection_name, embedding_function=sentence_transformer_ef
        )

    while True:
        args.topic = input("You: ")  # Get input from the user

        if args.topic.lower() == "bye":
            print("Chatbot: Goodbye! Have a great day!")
            break  # Exit the loop if the user types "bye"

        results = []

        if args.topic is not None:
            topic_text = args.topic

        if args.rewrite:
            topic_text_expended_context = prepare_documents(
                text=topic_text,
                method="context",
                client_handler=opinion_analyzer,
            )

            topic_text_segmented_rw = topic_text_expended_context[
                "new_sentence"
            ].tolist()
            topic_text_segmented = topic_text_expended_context[
                "original_sentence"
            ].tolist()

            if args.collection_name is None:
                argument_text_expended_context = prepare_documents(
                    text=config["paths"]["data"]
                         / "referendums"
                         / args.business_id
                         / args.file,
                    method="llm",
                    client_handler=opinion_analyzer,
                )

                argument_text_segmented_rw = argument_text_expended_context[
                    "new_sentence"
                ].tolist()
                argument_text_segmented = argument_text_expended_context[
                    "original_sentence"
                ].tolist()
                argument_text_contexts = argument_text_expended_context[
                    "context"
                ].tolist()

                argument_text_segmented_emb = sent_model.encode(
                    argument_text_segmented_rw
                )
            else:
                all_documents = collection.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=10000,
                    offset=0,
                )

                # Prepare data for JSON output
                output_data = [
                    {
                        "id": doc_id,
                        "text": doc_text,
                        "metadatas": metadata,
                        "embedding": (
                            embedding.tolist() if embedding is not None else None
                        ),  # Convert numpy array to list for JSON compatibility
                    }
                    for doc_id, doc_text, metadata, embedding in zip(
                        all_documents["ids"],
                        all_documents["documents"],
                        all_documents["metadatas"],
                        all_documents["embeddings"],
                    )
                ]

                argument_text_segmented_rw = [doc["text"] for doc in output_data]
                argument_text_segmented = [
                    doc["metadatas"]["original_sentence"] for doc in output_data
                ]
                argument_text_contexts = [
                    doc["metadatas"]["context"] for doc in output_data
                ]
                argument_text_segmented_emb = torch.tensor(
                    [doc["embedding"] for doc in output_data]
                )

        for segment_idx, segment in enumerate(topic_text_segmented_rw):
            segment_emb = sent_model.encode(segment)
            print("Extracting the arguments for the following topic:\n")
            print(segment)
            print("--------")
            results_semantic_search = util.semantic_search(
                segment_emb, argument_text_segmented_emb
            )[0]
            results_semantic_search = [
                x
                for x in results_semantic_search
                if x["score"] > config["thresholds"]["sentence_similarity"]
            ]
            for idx, sent in enumerate(argument_text_segmented_rw):
                if len(sent) > 10:
                    if idx in [x["corpus_id"] for x in results_semantic_search]:
                        debatable = "ja"
                        if args.debatability_check:
                            debatable = (
                                    opinion_analyzer.is_debatable(sent).lower().strip()
                                    == "ja"
                            )
                        if debatable == "ja":
                            result = opinion_analyzer.categorize_argument(
                                topic_text=segment, text_sample=sent, method=args.method
                            )
                            stance = adjust_labels(
                                label=result["label"],
                                score=result["score"],
                                threshold=opinion_analyzer.stance_class_threshold,
                            )

                            # Extracting the reasoning for the argument
                            reason = opinion_analyzer.generate(
                                prompt=prompt_dict[
                                    config["prompts"]["find_reasoning"]
                                ].format(
                                    # topic=segment,
                                    claim=sent,
                                    stance=stance,
                                    context=argument_text_contexts[idx],
                                )
                            )

                            try:
                                reason = literal_eval(reason)
                            except SyntaxError as se:
                                print(f"SyntaxError: {se}")
                                reason = {"reasoning_segment": "", "reasoning": ""}

                            # Extracting the person of the argument
                            person_info = {"person": "", "party": "", "canton": ""}
                            if stance in ["pro", "contra"]:
                                person_info = opinion_analyzer.generate(
                                    prompt=prompt_dict[
                                        config["prompts"]["extract_person"]
                                    ].format(
                                        # topic=segment,
                                        sentence=sent,
                                        # stance=stance,
                                        context=argument_text_contexts[idx],
                                    )
                                )

                                try:
                                    person_info = literal_eval(person_info)
                                except SyntaxError as se:
                                    print(f"SyntaxError: {se}")

                            entry = {
                                "topic_original": topic_text_segmented[segment_idx],
                                "topic_rewritten": segment,
                                "argument_rewritten": sent,
                                "argument_original": argument_text_segmented[idx],
                                "argument_reason": result["model_generation"],
                                "person": person_info["person"],
                                "party": person_info["party"],
                                "canton": person_info["canton"],
                                "context": argument_text_contexts[idx],
                                "label": stance,
                                "score": result["score"],
                                "reasoning": reason["reasoning"],
                                "reasoning_segment": reason["reasoning_segment"],
                                "similarity": [
                                    x
                                    for x in results_semantic_search
                                    if x["corpus_id"] == idx
                                ][0]["score"],
                            }
                            print(entry)
                            results.append(entry)
                            pprint(entry)

        results = pd.DataFrame(results)
        results = results.applymap(escape_xlsx_string)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.timestamp is not None:
            timestamp = args.timestamp
        output_dir = Path("argument_mining_results") / timestamp
        os.makedirs(output_dir, exist_ok=True)
        if args.method == "finetuned":
            model_name = Path(
                opinion_analyzer.stance_classifier.model.name_or_path
            ).name.replace("/", "_")
        else:
            model_name = args.model_name_or_path.replace("/", "_")
        results.to_excel(
            output_dir
            / f"argument_analysis__business_id_{args.business_id}__{args.file}__{model_name}.xlsx",
            index=False,
        )
