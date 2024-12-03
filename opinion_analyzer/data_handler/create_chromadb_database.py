import argparse
import os

import chromadb
from chromadb.utils import embedding_functions

from opinion_analyzer.analyzer import prepare_documents, OpinionAnalyzer
from opinion_analyzer.utils.helper import (
    get_main_config,
)

config = get_main_config()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        default=config["models"]["llm"],
        type=str,
        help="Specify the model name.",
    )

    parser.add_argument(
        "-bi",
        "--business_id",
        type=int,
        default=None,
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
        "-cn",
        "--collection_name",
        type=str,
        default=None,
        help="Specify the name of the collection.",
    )

    args = parser.parse_args()

    if args.business_id is None:
        args.business_id = [
            x for x in os.listdir(config["paths"]["data"] / "referendums")
        ]
    else:
        args.business_id = [str(args.business_id)]

    if args.collection_name is None:
        args.collection_name = config["database"]["chromadb"]

    opinion_analyzer = OpinionAnalyzer(args.model_name_or_path)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config["models"]["sentence_similarity"]
    )

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
    else:
        collection = chroma_client.create_collection(
            args.collection_name,
            embedding_function=sentence_transformer_ef,
            metadata={"hnsw:space": "cosine"},
        )

    for bid in args.business_id:

        argument_text_expended_context = prepare_documents(
            text=config["paths"]["data"] / "referendums" / bid / args.file,
            method="llm",
            client_handler=opinion_analyzer,
        )

        # Embed and add documents to the collection
        collection.add(
            documents=argument_text_expended_context["new_sentence"].tolist(),
            ids=[
                f"{bid}-{n}"
                for n, doc in enumerate(
                    argument_text_expended_context["new_sentence"].tolist()
                )
            ],
            metadatas=[
                {
                    "business_id": bid,
                    "filename": args.file,
                    "original_sentence": doc["original_sentence"],
                    "context": doc["context"],
                    "url": f"https://www.parlament.ch/de/ratsbetrieb/suche-curia-vista/geschaeft?AffairId={bid}",
                }
                for n, doc in enumerate(
                    argument_text_expended_context.to_dict(orient="records")
                )
            ],
        )
