import os
import pandas as pd
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import torch

from tqdm import tqdm

from aitutor_assessmentkit.helpers import utils
from aitutor_assessmentkit.helpers.constants import COMMON_COLUMNS

class AutoEvaluator:
    def __init__(
        self,
        input_data_dir: str = None,
        output_data_dir: str = None,
        file_names: Union[List[str], str] = None,
        tutor_models: Union[List[str], str] = None,
        num_conv_examples: int = -1,
    ):
        """
        Initialize the AutoEvaluator object.

        Arguments:
            input_data_dir (str): The path to the directory containing the input JSON data files.
            output_data_dir (str): The path to the directory where the output data will be saved as a JSON file.
            file_names (Union[List[str], str]): The names of the input JSON files.
            tutor_models (Union[List[str], str]): The names of the tutor models to consider during evaluation.
            num_conv_examples (int): The number of conversation examples to consider for evaluation. -1 for all examples.
        """
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.file_names = file_names if isinstance(file_names, list) else [file_names] if file_names else []
        self.tutor_models = tutor_models if isinstance(tutor_models, list) else [tutor_models] if tutor_models else []
        self.num_conv_examples = num_conv_examples
        

        # Initialize data storage
        self.data = []

        # Load data from JSON files if input_data_dir is provided
        if self.input_data_dir:
            self.file_names = utils.get_valid_files(self.input_data_dir, file_extension=".json")
            self.data = [utils.load_json_data(fname) for fname in self.file_names]

        # Load JSON files
        if self.file_names is not None:
            if isinstance(self.file_names, str):  # Single file case
                self.data.append(utils.load_json_data(self.file_names))
            elif isinstance(self.file_names, list):  # Multiple files case
                for fname in tqdm(self.file_names, desc="Loading data"):
                    self.data.extend(utils.load_json_data(fname))
        
        # Filter the number of first conversation examples
        if self.num_conv_examples > 0:
            self.data = self.data[:num_conv_examples]
        
        # clean the data and convet into lower case
        utils.clean_data(self.data)

        # Check if the required columns are present in the DataFrames
        required_columns = COMMON_COLUMNS
        utils.sanity_check_columns(self.data, required_columns)

        # Write function to check if all the tutor models are present in each example of the data
        # If not present, raise an error pointing out which tutor is missing in which examples
        if self.tutor_models:
            utils.sanity_check_tutor_models(self.data, self.tutor_models)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def get_data(self) -> pd.DataFrame:
        """
        Get the merged data.
        Returns:
            pd.DataFrame: The merged data.
        """
        return self.data