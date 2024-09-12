"""
This module provides functionality to interact with the Swiss Parliament API.

Functions:
    get_data(business_id: int) -> pd.DataFrame:
        Fetches data from the Swiss Parliament API for a given business ID and converts it into a DataFrame.

    get_final_pdf_with_arguments(business_id: int, output_path: str = None):
        Downloads a PDF file from the Swiss Parliament website for the given business ID and saves it to the specified output path.

Usage:
    To use this module from the command line, you can pass the `--business_id` or `-bid` argument to specify the business ID you want to fetch data for.

Example:
    python script_name.py --business_id 12345
"""

import swissparlpy as spp
import argparse
import pandas as pd
import requests


def get_data(business_id: int) -> pd.DataFrame:
    """
    :param business_id: The ID of the business for which data is to be fetched
    :return: A pandas DataFrame containing the fetched data for the specified business
    """
    fetched_data = spp.get_data("Business", ID=business_id)
    return pd.DataFrame(fetched_data)


def get_final_pdf_with_arguments(business_id: int, output_path: str = None):
    """
    Download a PDF file from the Swiss Parliament website for the given business ID and save it to the specified output path.

    :param business_id: The business ID to fetch the PDF for.
    :type business_id: int
    :param output_path: The file path where the PDF should be saved. If not specified, defaults to '<business_id>.pdf'.
    :type output_path: str, optional
    """
    url = f"https://www.parlament.ch/de/ratsbetrieb/suche-curia-vista/geschaeft?AffairId={str(business_id)}"
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    if output_path is None:
        output_path = f"{business_id}.pdf"
    with open(output_path, "wb") as file:
        file.write(response.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch data from Swiss Parliament API")
    parser.add_argument(
        "--business_id", "-bid", type=int, help="The business ID to fetch data for"
    )
    args = parser.parse_args()

    data = get_data(args.business_id)
    print(sorted(data.columns))

    # get_final_pdf_with_arguments(args.business_id)
    print("Tables")
    print(spp.get_tables())

    print("###################")
    fetched_data = spp.get_data("Publication", ID=args.business_id)
    print(pd.DataFrame(fetched_data))
