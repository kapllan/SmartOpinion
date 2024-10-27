# Code evaluate the performance of stance detection models on the IBM Debater dataset.


import os
import datetime
from opinion_analyzer.analyzer import OpinionAnalyzer
from opinion_analyzer.utils.helper import adjust_labels
from opinion_analyzer.data_handler.prompt_database import prompt_dict
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Union

tqdm.pandas()


def convert_label(label: str) -> int:
    """
    Convert a label to an integer representation.

    This function converts a label string to an integer, based on its content:
    - Labels starting with "pro" (case insensitive) are converted to 1.
    - Labels starting with "contra" (case insensitive) are converted to -1.
    - Other labels are converted to 0.

    :param str label: The label to be converted.
    :return: An integer representing the converted label.
    :rtype: int
    """
    if label.lower().strip().startswith("pro"):
        return 1
    if label.lower().strip().startswith("contra"):
        return -1
    return 0


from typing import Union


def round_values(value: Union[int, float, str]) -> Union[int, float, str]:
    """
    Rounds the value if it is a float and less than or equal to 1.0.
    Converts such a float value to a percentage format rounded to two decimal places.

    :param value: The value to be processed. It can be an integer, a float, or a string.
    :type value: Union[int, float, str]
    :return: The processed value. If the input value is a float less than or equal to 1.0,
             it returns the value multiplied by 100 and rounded to two decimal places.
             Otherwise, it returns the input value unchanged.
    :rtype: Union[int, float, str]
    """
    if isinstance(value, float) and value <= 1.0:
        return round(value * 100, 2)
    else:
        return value


def get_target_name(label: int) -> str:
    mapping = {-1: "Contra", 0: "Neutral", 1: "Pro"}

    return mapping[label]


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
        "-p",
        "--prompt",
        type=str,
        default="categorize_argument_zero_shot",
        choices=[
            "categorize_argument_zero_shot",
            "categorize_argument_few_shot",
            "categorize_argument_zero_shot_cot",
        ],
        help="Specify the prompt you want to use for the evaluation. Defaults to categorize_argument_zero_shot.",
    )

    args = parser.parse_args()

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if args.method == "llm":
        thresholds = [None]

    if args.model_name_or_path is not None:
        models_to_evaluate = args.model_name_or_path.split(";")
    else:
        models_to_evaluate = [None]
    print(models_to_evaluate)
    for mnp in models_to_evaluate:
        print(f"Evaluating model {mnp}.")

        opinion_analyzer = OpinionAnalyzer(mnp)

        if args.method == "finetuned":
            model_name = Path(
                opinion_analyzer.stance_classifier.model.name_or_path
            ).name
        else:
            model_name = mnp.replace("/", "_")
        current_time = datetime.datetime.now()

        if args.method == "finetuned":
            subdirectory = (
                f"method_{args.method}/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        else:
            subdirectory = f"method_{args.method}/{args.prompt}/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"

        output_dir = Path(f"results_{model_name}") / subdirectory
        os.makedirs(output_dir, exist_ok=True)

        path_to_dataset = Path(
            "../../data/datasets/IBM_Debater_(R)_XArgMining/Human Authored Arguments/human_authored_arguments_de.csv"
        )
        test_data = pd.read_csv(path_to_dataset)

        test_data = test_data[test_data.stance_conf > 0.5][:20]

        test_data["method"] = args.method
        test_data["prompt"] = args.prompt
        test_data["model_name"] = model_name

        reports_collected = []

        test_data["results"] = test_data.progress_apply(
            lambda row: opinion_analyzer.categorize_argument(
                topic_text=row["topic_DE"],
                text_sample=row["argument_DE"],
                method=args.method,
                prompt=prompt_dict[args.prompt],
            ),
            axis=1,
        )
        test_data["stance_label_pred_str"] = test_data.results.apply(
            lambda x: x["label"]
        )

        test_data["score"] = test_data.results.apply(lambda x: x["score"])

        for threshold in thresholds:
            test_data["stance_label_pred_str"] = test_data.apply(
                lambda item: adjust_labels(
                    label=item["stance_label_pred_str"],
                    score=item["score"],
                    threshold=threshold,
                ),
                axis=1,
            )
            test_data["stance_label_pred"] = test_data.stance_label_pred_str.apply(
                lambda x: convert_label(x)
            )

            y_true = test_data["stance_label"]
            y_pred = test_data["stance_label_pred"]

            target_names = sorted(list(set(list(y_true) + list(y_pred))))
            target_names = [get_target_name(x) for x in target_names]

            report = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            )
            report = pd.DataFrame(report)
            report = report.transpose()
            report = report.applymap(round_values)
            report["threshold"] = threshold

            print(report)

            with pd.ExcelWriter(
                output_dir
                / f"test_results__{model_name}__with_threshold_{threshold}__{current_time.timestamp()}.xlsx"
            ) as writer:
                test_data.to_excel(
                    writer, index=False, sheet_name=f"Prediction_Results"
                )
                report.to_excel(writer, index=True, sheet_name="Overview ClassReports")
            reports_collected.append(report)
        reports_collected = pd.concat(reports_collected)
        reports_collected.to_excel(
            output_dir
            / f"test_results_collected__{model_name}__{current_time.timestamp()}.xlsx",
            index=True,
        )
