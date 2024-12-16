import os
import re
import json
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Tuple

from tqdm import tqdm

from aitutor_assessmentkit.helpers.constants import VALID_FILE_EXTENSIONS, LABELS_ENCODING, LLM_LABELS_ENCODING


def get_valid_files(input_data_dir: str) -> List[str]:
    """
    Get a list of valid file names from the input directory.
    Arguments:
        input_data_dir (str): The directory to scan for files.
    Returns:
        List[str]: A list of valid file names with .csv or .json extensions.
    """
    valid_files = []
    for file_name in os.listdir(input_data_dir):
        if validate_file_format(file_name):
            valid_files.append(os.path.join(input_data_dir, file_name))
    return valid_files

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame based on its format.
    Arguments:
        file_path (str): The full path to the file.
    Returns:
        pd.DataFrame: The loaded pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def merge_dataframes(dfs: List[pd.DataFrame], axis: int = 0) -> pd.DataFrame:
    """
    Merge a list of pandas DataFrames along the specified axis.
    
    By default, merge along rows (axis=0) and ensure that all DataFrames have the same columns.

    Arguments:
        dfs (List[pd.DataFrame]): The list of DataFrames to merge.
        axis (int): The axis along which to concatenate. 0 for rows (default), 1 for columns.

    Returns:
        pd.DataFrame: A new DataFrame with all input DataFrames merged.
    
    Raises:
        TypeError: If any item in the list is not a pandas DataFrame.
        ValueError: If concatenating along rows and the DataFrames have different columns.
        ValueError: If concatenating along columns and the DataFrames have different number of rows.
        ValueError: If axis is not 0 or 1.
    """
    if not all(isinstance(df, pd.DataFrame) for df in dfs):
        raise TypeError("All items in the list must be pandas DataFrames.")
    
    if axis == 0:
        # Check for column consistency if concatenating along rows
        columns_set = set(dfs[0].columns)
        if not all(set(df.columns) == columns_set for df in dfs):
            raise ValueError("All DataFrames must have the same columns when concatenating along rows.")
        
        # Concatenate DataFrames along rows (stack them) and retain column names
        merged_df = pd.concat(dfs, axis=axis, ignore_index=True)
    
    elif axis == 1:
        # Check for row consistency if concatenating along columns
        rows_set = set(len(df) for df in dfs)
        if len(rows_set) > 1:
            raise ValueError("All DataFrames must have the same number of rows when concatenating along columns.")
        
        # Concatenate DataFrames along columns (side by side) and retain column names
        merged_df = pd.concat(dfs, axis=axis)
    
    else:
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")

    return merged_df

def validate_file_format(file_name: str) -> bool:
    """
    Validate the file format based on its extension.
    Arguments:
        file_name (str): The name of the file.
    Returns:
        bool: True if the file format is supported, False otherwise.
    """
    return any(file_name.endswith(ext) for ext in VALID_FILE_EXTENSIONS)

def convert_text_to_lowercase(dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all text in the provided DataFrames to lowercase.
    Arguments:
        dataframes (pd.DataFrame): A pandas DataFrames.
    Returns:
        pd.DataFrame: The DataFrames with all text converted to lowercase.
    """
    lowercased_df = dataframes.copy()
    for column in lowercased_df.columns:
        if lowercased_df[column].dtype == object:  # Check if the column is of type object (usually text)
            lowercased_df[column] = lowercased_df[column].str.lower()
    return lowercased_df.astype(str)  # Ensure all columns are of type string

def sanity_check_columns(data: List[Dict[str, Any]], required_columns: List[str]) -> bool:
    """
    Check if all required columns are present in each JSON object.
    
    Arguments:
        data (List[Dict[str, Any]]): A list of dictionaries representing the JSON objects.
        required_columns (List[str]): A list of column names that must be present.
    
    Returns:
        bool: True if all required columns are present in each JSON object, False otherwise.
    """
    # Convert the list of dictionaries to a DataFrame for easier column checking
    df = pd.DataFrame(data)
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in JSON data: {missing_columns}")
    
    return True

def extract_absolute_answer(output):
    pattern = r"""
        (?:                        # Start of non-capturing group
            \[RESULT\]|\[SCORE\]|   # Match [RESULT] or [SCORE]
            Score:?|score:?|        # Match Score: or score:
            Result:?|\[Result\]:?|  # Match Result: or [Result]:
            score\s+of              # Match "score of"
        )                           # End of non-capturing group
        \s*                         # Allow any whitespace
        (?:\(|\[|\s)*               # Allow opening brackets or whitespace
        (\d+)                       # Capture the digit(s)
        (?:                         # Start of non-capturing group
            (?:\)|\]|\s|$)|         # Allow closing brackets, whitespace, or end of string
            (?:/\s*3|               # Allow /3 with optional whitespace
               \s*out\s*of\s*3)     # or "out of 3" with flexible whitespace
        )?                          # End of non-capturing group
        (?:\s*$)                    # Match from the end of string 
    """
    match = re.search(pattern, output, re.IGNORECASE | re.VERBOSE)

    if match:
        result = int(match.group(1))
        if result in [1, 2, 3]:
            return result
    else:
        return None

# def extract_absolute_answer(text):
#     """
#     Extract the answer from the model output.
#     Arguments:
#         text (str): The model output text.
#     Returns:
#         int: The extracted answer if it is a valid number, otherwise None.
#     """

#     match = re.findall(r'\b\d+\b', text)
    
#     if match:
#         number = int(match[-1])
#         if number in [1, 2, 3]:
#             return number
    
#     return None

def batch_completions_with_retries(
    model,
    inputs,
    mode: str,
    max_retries: int = 10,
    params: dict = None, ) -> Tuple[List[float], float]:
    """
    Generate feedback for a batch of inputs using the specified model.
    Arguments:
        model: The language model to use for generating feedback.
        inputs (List[str]): A list of input prompts to generate feedback for.
        mode (str): The mode to use for parsing the model output.
        max_retries (int): The maximum number of retries for failed instances.
        params (dict): Additional parameters to pass to the model.
    Returns:
        Tuple[List[float], float]: A tuple containing the assesment scores and the percentage of failed instances.
    """
    # Override default params
    if params is None or params == {}:
        print("Parameter set is empty, setting to default values") 
        params = {
            "max_tokens": 1024,
            "repetition_penalty": 1.03,
            "best_of": 1,
            "temperature": 0.0,
            "top_p": 0.9,
        }

    batched_outputs = model.completions(inputs, **params, use_tqdm=True)

    scores = []
    for output in tqdm(batched_outputs, desc="Finalizing"):
        answer = extract_absolute_answer(output)
        scores.append(answer)
    
    none_count = scores.count(None)
    return scores, (none_count/len(scores))*100

def load_json_data(file_name: str) -> list:
    """
    Load the JSON data from the specified file.

    Arguments:
        file_name (str): The name of the JSON file to load.
    Returns:
        list: The loaded JSON data, or an empty list if the file doesn't exist.
    """
    json_data = []  # Initialize json_data with a default value
    
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            json_data = json.load(file)
    else:
        print(f"Warning: The file {file_name} does not exist.")
    
    print(f"Loaded {len(json_data)} examples from {file_name}")
    return json_data


def sanity_check_tutor_models(data, input_tutors) -> None:
    """
    Check if all the tutor models are present in each example of the data.
    """
    for example in tqdm(data, desc="Sanity Check for Tutor Models"):
        for tutor_model in input_tutors:
            if tutor_model == 'Novice' and example['Data'] == 'MathDial':
                continue
            if tutor_model not in list(example['anno_llm_responses'].keys()):
                raise ValueError(f"Missing tutor model '{tutor_model}' in example: {example['conversation_id']}")

            
def clean_data(data: List[Dict[str, Any]]) -> None:
    """
    Clean the data by removing any empty or missing values.
    
    Arguments:
        data (List[Dict[str, Any]]): A list of dictionaries representing the JSON objects.
    
    Returns:
        List[Dict[str, Any]]: The cleaned list of dictionaries.
    """
    for example in tqdm(data, desc="Cleaning Data"):
        temp_hist = ' ||| '.join([
            item.encode('ascii', 'ignore').decode('ascii').strip() 
            for item in re.split(r"(?=(student:|tutor:))", example['conversation_history'].lower()) 
            if len(item) != 0 and item != 'tutor:' and item != 'student:'
        ])
        example['conversation_history'] = '||| ' + str(temp_hist) + ' |||'
        example['Ground_Truth_Solution'] = example['Ground_Truth_Solution'].lower().encode('ascii', 'ignore').decode('ascii').strip()
        
        for item in example['anno_llm_responses']:
            example['anno_llm_responses'][item]['response'] = (
                example['anno_llm_responses'][item]['response'].lower()
                .encode('ascii', 'ignore').decode('ascii').strip()
            )

def save_data(output_data_path: str, file_name: str, data: List[Dict[str, Any]]) -> None:
    """
    Save the JSON data to the specified file.

    Arguments:
        output_data_path (str): The path to the output directory.
        file_name (str): The name of the JSON file to save.
        data (list): The JSON data to save.
    """
    if not os.path.exists(output_data_path):
        os.makedirs(output_data_path)
    
    with open(os.path.join(output_data_path, file_name), 'w') as file:
        json.dump(data, file, indent=4)

def extract_scores(
    data: List[Dict[str, Any]], 
    tutor_model: str, 
    dimension: str, 
    eval_flag: str, 
    normalize: bool = False
) -> List[float]:
    """
    Extract evaluation scores for the specified tutor model and dimension.
    
    Args:
        data (List[Dict[str, Any]]): A list of dictionaries representing the JSON objects.
        tutor_model (str): The name of the tutor model.
        dimension (str): The dimension to extract the evaluation scores from.
        eval_flag (str): The type of evaluation ('humaneval', 'autoeval', 'llmeval').
        normalize (bool): Whether to normalize the scores (default: False).
    
    Returns:
        List[float]: A list of scores for the specified tutor model and dimension.
    """
    def get_real_dimension(dim: str) -> str:
        """
        Process and extract the correct dimension key.
        
        Args:
            dim (str): The original dimension string.
        
        Returns:
            str: The processed dimension key.
        """
        parts = dim.strip().split('_')
        if dim.startswith(('Providing_Guidance', 'Tutor_Tone', 'Mistake_Identification', 'Mistake_Location')):
            return '_'.join(parts[:2])
        elif dim.startswith('Revealing_of_the_Answer'):
            return '_'.join(parts[:4])
        return parts[0]

    eval_scores = []
    for example in data:
        if tutor_model not in example.get('anno_llm_responses', {}):
            continue

        response_data = example['anno_llm_responses'][tutor_model]
        
        if eval_flag == 'humaneval':
            real_dim = get_real_dimension(dimension)
            label = response_data['annotation'].get(real_dim)
            eval_scores.append(LABELS_ENCODING[real_dim][label])
        elif eval_flag == 'autoeval':
            auto_score = response_data['auto_annotation'].get(dimension)
            eval_scores.append(auto_score)
        elif eval_flag == 'llmeval':
            real_dim = get_real_dimension(dimension)
            llm_score = response_data['llm_annotation'].get(dimension)
            if normalize and llm_score is not None:
                llm_score = LLM_LABELS_ENCODING[llm_score]
            eval_scores.append(llm_score)

    return eval_scores

def should_skip_example(tutor_model, example) -> bool:
    """Determine if the example should be skipped based on tutor model and data type."""
    return tutor_model == 'Novice' and example.get('Data') == 'MathDial'

def get_response_text(tutor_model, example) -> str:
    """ Get the response text for the specified tutor model from the example."""
    return example["anno_llm_responses"][tutor_model]['response']

def update_auto_annotation(tutor_model, example, hum_score, dimension):
    """Initialize and update auto_annotation with the Humanlikeness score."""
    example["anno_llm_responses"].setdefault(tutor_model, {}).setdefault('auto_annotation', {})
    example["anno_llm_responses"][tutor_model]['auto_annotation'][dimension] = hum_score

def update_llm_annotation(data, tutor_model, scores, convids, metric) -> None:
    """Initialize and update auto_annotation with the Humanlikeness score."""
    
    if tutor_model == 'Novice':
        for id in convids:
            for example in data:
                if example['conversation_id'] == id:
                    example["anno_llm_responses"].setdefault(tutor_model, {}).setdefault('llm_annotation', {})
                    example["anno_llm_responses"][tutor_model]['llm_annotation'][metric] = scores[convids.index(id)]
    else:
        for score, example in zip(scores, data):
            example["anno_llm_responses"].setdefault(tutor_model, {}).setdefault('llm_annotation', {})
            example["anno_llm_responses"][tutor_model]['llm_annotation'][metric] = score

def validate_tutor_model(tutor_model: str, example: dict) -> None:
    """
    Checks if the specified tutor model is a valid key in example["anno_llm_responses"].
    Raises a ValueError if the model is not found.

    Parameters:
        tutor_model (str): The tutor model to validate.
        example (dict): A dictionary containing a key "anno_llm_responses" with available models.

    Raises:
        ValueError: If the specified tutor model is not in example["anno_llm_responses"].
    """
    if tutor_model not in example["anno_llm_responses"]:
        raise ValueError(f"Invalid tutor model. Please choose from {list(example['anno_llm_responses'].keys())}")

def collect_inlist(example: dict, response1: str, score1: float, response2: str = None, score2: float = None) -> list:
    """
    Creates a list with conversation_id, conversation_history, and response(s) with score(s) for annotation.

    Parameters:
        example (dict): A dictionary containing the conversation_id and conversation_history.
        response1 (str): The first response text.
        score1 (float): The score for the first response.
        response2 (str, optional): The second response text (default is None).
        score2 (float, optional): The score for the second response (default is None).

    Returns:
        list: A list containing the conversation_id, conversation_history, response(s), and score(s).
    """
    annotation = [example["conversation_id"], example["conversation_history"], response1, score1]
    if response2 is not None and score2 is not None:
        annotation.extend([response2, score2])
    return annotation

def create_annotations_dataframe(collect_annotations: list, tutor_model1: str, dim: str, tutor_model2: str = None) -> pd.DataFrame:
    """
    Converts a list of collected annotations into a DataFrame with specified columns.

    Parameters:
        collect_annotations (list): A list of annotations, each containing conversation ID, history, response, and score.
        tutor_model (str): The primary tutor model being evaluated, used to name columns for response and score.
        dim (str): The dimension being evaluated (e.g., "Humanlikeness").
        tutor_model2 (str, optional): A second tutor model to include additional response and score columns.

    Returns:
        pd.DataFrame: A DataFrame containing the collected annotations with appropriately named columns.
    """
    columns = ["Conversation ID", "History", f"{tutor_model1} Response", f"{tutor_model1} {dim} Score"]
    if tutor_model2:
        columns.extend([f"{tutor_model2} Response", f"{tutor_model2} {dim} Score"])
    return pd.DataFrame(collect_annotations, columns=columns)

def get_sample_dataframe(data: list, tutor_model1: str, scores1: list, tutor_model2: str = None, scores2: list = None, dim: str = None) -> pd.DataFrame:
    """
    Collects annotations and returns them as a DataFrame.

    Parameters:
        data (list): A list of examples containing conversation data.
        hum_scores1 (list): A list of human evaluation scores for each example for the first tutor model.
        tutor_model1 (str): The primary tutor model being evaluated.
        dim (str): The dimension being evaluated (e.g., "Humanlikeness").
        tutor_model2 (str, optional): A second tutor model to include additional response and score columns.
        hum_scores2 (list, optional): A list of human evaluation scores for the second tutor model, if provided.

    Returns:
        pd.DataFrame: A DataFrame with collected annotations.
    """
    
    collect_annotations = []
    
    for i, (example, score1) in enumerate(zip(data, scores1)):
        # Validate the primary tutor model
        validate_tutor_model(tutor_model1, example)
        response1 = get_response_text(tutor_model1, example)

        # If tutor_model2 and hum_scores2 are provided, process them as well
        if tutor_model2 and scores2:
            validate_tutor_model(tutor_model2, example)
            response2 = get_response_text(tutor_model2, example)
            score2 = scores2[i]  # Use hum_scores2[i] for the second model's score

            # Append results for both models
            collect_annotations.append(collect_inlist(example, response1, score1, response2, score2))
        else:
            # Append results for only the first tutor model
            collect_annotations.append(collect_inlist(example, response1, score1))
    
    # Create and return the DataFrame
    if tutor_model2:
        return create_annotations_dataframe(collect_annotations, tutor_model1, dim, tutor_model2)
    else:
        return create_annotations_dataframe(collect_annotations, tutor_model1, dim)
    
def softmax(logits: List[float]) -> List[float]:
    """
    Compute the softmax of the logits.

    Arguments:
        logits (List[float]): The logits to compute the softmax.
    
    Returns:
        List[float]: The softmax of the logits.

    """
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

def get_coherence_history(data) -> str:
    """
    Clean the history and get the last student utterance.

    Arguments:
        history (str): The history to clean.

    Returns:
        str: The last student utterance. 
    """
    history = data['conversation_history'].lower()
    history = history.replace('student: ', '')  # Remove 'student: '
    history = history.replace('teacher:', '')  # Remove 'teacher:'
    history = history.replace('tutor:', '')  # Remove 'teacher:'
    #history = '[SEP]'.join([item.strip() for item in history.strip().split('|||') if len(item.strip()) != 0][-1]) 
    history = [item.strip() for item in history.strip().split('|||') if len(item.strip()) != 0]
    return history

def get_providing_guidance_history(data) -> str:
    """
    Clean the history and get the last student utterance.

    Arguments:
        history (str): The history to clean.

    Returns:
        str: The last student utterance. 
    """
    history = data['conversation_history'].lower()
    history = history.replace('student: ', '')  # Remove 'student: '
    history = history.replace('teacher:', '')  # Remove 'teacher:'
    history = history.replace('tutor:', '')  # Remove 'teacher:'
    #history = '[SEP]'.join([item.strip() for item in history.strip().split('|||') if len(item.strip()) != 0][-1]) 
    history = [item.strip() for item in history.strip().split('|||') if len(item.strip()) != 0]
    return history

def get_hum_scores(data: List[Dict[str, Any]], tutor_model: str, dimension: str) -> List[float]:
    eval_scores = []
    for example in data:
        if tutor_model in example['anno_llm_responses']:
            eval_scores.append(LABELS_ENCODING[dimension][example['anno_llm_responses'][tutor_model]['annotation'][dimension]])
    return eval_scores        

def extract_average_human_scores(data: List[Dict[str, Any]], tutor_models: List[str], dimension: str) -> Dict[str, float]:
    collect_dict = {}
    for tutor_model in tutor_models:
        if tutor_model not in collect_dict:
            collect_dict[tutor_model] = []
        for dim in dimension:
            raw_scores = get_hum_scores(data, tutor_model, dim)
            collect_dict[tutor_model].append(np.mean(raw_scores))
    return collect_dict

def get_all_history(data) -> str:
    """
    Clean the history and get the last student utterance.

    Arguments:
        history (str): The history to clean.

    Returns:
        str: The last student utterance. 
    """
    history = data['conversation_history'].lower()
    history = '\n'.join([item.strip() for item in history.strip().split('|||') if len(item.strip()) != 0])
    return history
    
def get_gold_response_text(example) -> str:
    """ Get the response text for the specified tutor model from the example."""
    return example["Ground_Truth_Solution"]

def get_topic_text(example) -> str:
    """ Get the response text for the specified tutor model from the example."""
    return example["Topic"]

def get_conv_id(example) -> str:
    """ Get the response text for the specified tutor model from the example."""
    return example["conversation_id"]
    