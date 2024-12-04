import argparse
import os.path
from ast import literal_eval
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
from tqdm import tqdm
from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.data_handler.prompt_database import prompt_dict
from opinion_analyzer.utils.helper import get_main_config

if __name__ == "__main__":
    tqdm.pandas()

    config = get_main_config()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        default=config["models"]["llm"],
        type=str,
        help="Specify the model name.",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        default="evaluation_results/evidence_extraction",
        type=str,
        help="Specify the model name.",
    )

    args = parser.parse_args()

    opinion_analyzer = OpinionAnalyzer(model_name_or_path=args.model_name_or_path)

    dataset = pd.read_excel(Path(config["paths"]["datasets"]) / "test_dataset_argument_evidence_extraction.xlsx")

    prompt = prompt_dict[config["prompts"]["find_reasoning"]]

    dataset["prompt"] = config["prompts"]["find_reasoning"]

    dataset["result"] = dataset.progress_apply(
        lambda row: opinion_analyzer.extract_evidence(topic=row["Worum geht es?"], claim=row["Meinung"],
                                                      context=row["Kontext"],
                                                      stance=row["Haltung"]), axis=1)

    dataset["reasoning_segment"] = dataset["result"].apply(lambda x: x["reasoning_segment"])
    dataset["reasoning"] = dataset["result"].apply(lambda x: x["reasoning"])

    dataset["evaluated_model"] = args.model_name_or_path

    os.makedirs(args.output_path, exist_ok=True)

    model_name = re.sub("/", "_", args.model_name_or_path)

    dataset.to_excel(Path(
        args.output_path) / f"evidence_extraction__{config['prompts']['find_reasoning']}__{model_name}__{datetime.now().isoformat()}.xlsx",
                     index=False)
