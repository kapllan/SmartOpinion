import pandas as pd
from spacy.lang.lij.tokenizer_exceptions import prefix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
from opinion_analyzer.utils.helper import (
    get_main_config,
    compute_metrics_multi_class,
    remove_subfolders,
)
import numpy as np
import wandb
import os
from pathlib import Path
import argparse
import re
import datetime
from transformers import DataCollatorWithPadding
from pprint import pprint

config = get_main_config()

if __name__ == "__main__":

    # Set random seed for reproducibility
    seed = 42
    set_seed(seed)  # Set seed for the Hugging Face Trainer

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mnp",
        "--model_name_or_path",
        type=str,
        help="Specify the model name.",
    )

    args = parser.parse_args()

    MODEL_NAME = args.model_name_or_path
    OUTPUT_DIR = f"./results/{MODEL_NAME}_stance_classifier"

    # Initialize Weights and Biases
    wandb.init(
        project="stance_classification",
        name=f"{MODEL_NAME}__{datetime.datetime.now().isoformat(timespec='seconds')}",
    )

    label2id = {"Contra": 0, "Pro": 1}
    id2label = {0: "Contra", 1: "Pro"}

    # Initialize the tokenizer for the BERT model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Specify the model: BERT for sequence classification with 2 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if MODEL_NAME == "ZurichNLP/swissbert":
        model.model.set_default_language("de_CH")

    # Load your dataset (CSV)
    df = pd.read_csv(
        # config["paths"]["datasets"] / "IBM_Debater_(R)_XArgMining" / "Machine Translations" / "Arguments_6L_MT_own_translations.csv")
        config["paths"]["datasets"]
        / "IBM_Debater_(R)_XArgMining"
        / "Machine Translations"
        / "Arguments_6L_MT.csv"
    )

    # Filter only cases where the confidence was high enough.
    df = df[(df.stance_label_EN.isin([-1, 1]))]
    df = df[(df.quality_score_EN > 0.5)]

    # Combine the argument and topic into one input
    df["input_text"] = (
        # df["topic_DE_own"] + f" {tokenizer.sep_token} " + df["argument_DE_own"]
        df["topic_DE"]
        + f" {tokenizer.sep_token} "
        + df["argument_DE"]
    )
    # -1: "Contra", 0: "Neutral", 1: "Pro"
    df["label"] = df["stance_label_EN"]
    df["label"] = df["label"].apply(lambda x: 0 if x == -1 else 1)
    df = df[df["input_text"].apply(lambda x: isinstance(x, str) and len(x) > 1)]

    # Split the dataset into train and test sets
    train_df = df[df.set == "train"]
    test_df = df[df.set == "test"]
    eval_df = df[df.set == "dev"]

    # Convert pandas DataFrame to Hugging Face Dataset object
    train_dataset = Dataset.from_pandas(train_df[["input_text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["input_text", "label"]])
    eval_dataset = Dataset.from_pandas(eval_df[["input_text", "label"]])

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["input_text"], padding="max_length", truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,  # output directory
        evaluation_strategy="epoch",  # evaluate during training
        save_strategy="epoch",  # evaluate during training
        learning_rate=2e-5,  # learning rate
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=16,  # batch size for evaluation
        num_train_epochs=10,  # number of training epochs
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{config['metrics']['tr_classification_model']}",
        greater_is_better=True,
        fp16=True,
        report_to="wandb",  # report to wandb
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics_multi_class,
        data_collator=data_collator,
    )

    # Train the model
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate the model
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    MODEL_NAME = re.sub("/", "_", MODEL_NAME)

    # Save the trained model
    model.save_pretrained(
        Path(config["paths"]["models"]) / f"{MODEL_NAME}_stance_classifier"
    )
    tokenizer.save_pretrained(
        Path(config["paths"]["models"]) / f"{MODEL_NAME}_stance_classifier"
    )
    """with open(Path(config["paths"]["models"]) / f"{MODEL_NAME}_stance_classifier" / "label2id.json", "w") as file:
        js.dump(label2id, file=file)"""

    os.makedirs(Path("predictions") / f"{MODEL_NAME}_stance_classifier", exist_ok=True)

    # Apply the trained model to the test_dataset
    predictions, labels, metrics = trainer.predict(
        test_dataset, metric_key_prefix="test"
    )
    metrics["test_samples"] = len(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    # Save the results to a DataFrame
    test_df["predicted_label"] = predicted_labels

    # test_metrics = compute_metrics_multi_class(p=None, y_true=test_df['label'], y_pred=test_df['predicted_label'])
    # test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)
    wandb.log(metrics)

    # Save the DataFrame to an Excel file
    test_df.to_excel(
        Path("predictions") / f"{MODEL_NAME}_stance_classifier" / "test_results.xlsx",
        index=False,
    )

    remove_subfolders(OUTPUT_DIR)
