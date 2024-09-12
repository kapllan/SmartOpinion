from dataclasses import field
import pandas as pd

from opinion_analyzer.data_handler.data_handler import *
from bs4 import BeautifulSoup
import os
from pathlib import Path

business_ids = [
    20210063,
    20210067,
    20220075,
    20210047,
    20220043,
    20220054,
    20220036,
    20210501,
    20220046,
]


def make_naive(col):
    """
    Make a datetime column time zone naive.

    :param col: Pandas Series which might be a datetime column.
    :type col: pd.Series
    :return: Series with timezone removed if it's a datetime type.
    :rtype: pd.Series
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.dt.tz_localize(None)
    return col


def remove_tags(content: str):
    """
    Remove HTML tags from a string using BeautifulSoup.

    :param content: The string containing HTML tags.
    :type content: str
    :return: String with HTML tags removed.
    :rtype: str
    """
    if isinstance(content, str):
        content = BeautifulSoup(content).text
    return content


def save_filed(dataframe: pd.DataFrame, field_names: str = None):
    """
    Save specific fields of the dataframe to text files and the whole dataframe to an excel file.

    :param dataframe: The dataframe containing the data.
    :type dataframe: pd.DataFrame
    :param field_names: The specific fields in the dataframe to save separately.
                        If a single field, it should be provided as a string.
    :type field_names: list[str]
    """
    # Apply the function to all datetime columns
    dataframe = dataframe.apply(lambda col: make_naive(col))

    # Remove HTML tags from all columns
    columns = dataframe.columns
    for col in columns:
        dataframe[col] = dataframe[col].apply(remove_tags)

    if isinstance(field_names, str):
        field_names = [field_names]

    for field_name in field_names:
        field_text = dataframe[field_name].tolist()[0]
        field_text = BeautifulSoup(field_text).text
        with open(output_path / Path(f"{field_name}.txt"), "w") as f:
            print(field_text, file=f)

    dataframe.transpose().to_excel(output_path / Path("all_data.xlsx"))


for bid in business_ids:
    output_path = Path(f"../data/referendums/{bid}")
    os.makedirs(output_path, exist_ok=True)
    data = get_data(bid)
    data = data[data.Language == "DE"]
    save_filed(data, ["InitialSituation", "Proceedings"])
