import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union

from tqdm import tqdm

import torch
from transformers import AutoTokenizer

from aitutor_assessmentkit.helpers import utils
from aitutor_assessmentkit.helpers.utils import batch_completions_with_retries
from aitutor_assessmentkit.helpers.constants import COMMON_COLUMNS

from .prompts import (
    ABS_SYSTEM_PROMPT,
    ABSOLUTE_PROMPT,
    ABSOLUTE_PROMPT_WO_REF,
    REL_SYSTEM_PROMPT,
    RELATIVE_PROMPT,
    RELATIVE_PROMPT_WO_REF,
    DEFINITIONS,
    MISTAKE_IDENTIFICATION_RUBRIC,
    MISTAKE_LOCATION_RUBRIC,
    REVEALING_ANSWER_RUBRIC,
    PROVIDING_GUIDANCE_RUBRIC,
    ACTIONABILITY_RUBRIC,
    COHERENCE_RUBRIC,
    TUTOR_TONE_RUBRIC,
    HUMANLIKENESS_RUBRIC,
)

from .vllm import VLLM

class LLMEvaluator:
    def __init__(
        self,
        llm_model_name: str = None,
        llm_model_parama: Dict[str, Any] = None,
        evaluation_type: str = None,
        prompting_type: str = None,
        input_data_dir: str = None,
        output_data_dir: str = None,
        file_names: Union[List[str], str] = None,
        with_ref: bool = False,
        ngpus: int = 1,
        num_conv_examples: int = -1,
    ):
        """
        Initialize the LLMEvaluator object. This contains methods to evaluate each dimension of the pedagogical conversation.

        Arguments:
            llm_model (str): The name of the language model to use for evaluation.
            llm_model_parama (Dict[str, Any]): Additional parameters for the language model. Refer to the vllm SamplingParmas class.
            evalution_type (str): The type of evaluation to perform (absolute or relative).
            prompting_type (str): The type of prompting to use for evaluation (zero-shot or few-shot).
            input_data_dir (str): The path to the directory containing the input data (should be .csv or .json files).
            output_data_dir (str): The path to the directory where the output data will be saved (should be a .json file).
            file_names (Union[List[str], str]): The names of the input data files.
            with_ref (bool): Whether to use reference answers during evaluation.
            ngpus (int): The number of GPUs to use for evaluation.
            num_conv_examples (int): The number of conversation examples to consider for evaluation. -1 for all examples.

        Note: The runtime will automatically switch to the GPU if a GPU is available.
        - Default is zero-shot prompting, absolute evaluation, without reference answers and utterance level evaluation.
        """
        self.llm_model_name = llm_model_name
        self.llm_model_parama  = llm_model_parama    
        self.evaluation_type = evaluation_type
        self.prompting_type = prompting_type
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.file_names = file_names if isinstance(file_names, list) else [file_names] if file_names else []
        self.with_ref = with_ref
        self.ngpus = ngpus
        self.num_conv_examples = num_conv_examples

        if torch.cuda.device_count() < self.ngpus:
            raise ValueError(f"Requested {self.ngpus} GPUs but only {torch.cuda.device_count()} are available.")

        if self.ngpus > 1:
            print(f"Using {self.ngpus} GPUs for evaluation.")
            print(f"Current allocated GPUs are : {torch.cuda.get_device_capability(torch.cuda.current_device())}")

        self.llm_model = VLLM(self.llm_model_name, self.ngpus)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)

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

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _get_conversation_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Get the conversation prompt from the list of messages.

        Arguments:
            messages (List[Dict[str, str]]): A list of messages in the conversation.

        Returns:
            str: The conversation prompt.
        """
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def _get_data_with_prompt_template(
        self, 
        data: Dict[str, Any],
        tutor_model: str, 
        definition: str, 
        eval_instruction_rubric: str, 
        str = None
    ) -> pd.DataFrame:
        """
        Get the data with the appropriate prompt template based on the evaluation type and prompting type.

        Arguments:
            data (Dict[str, Any]): The data to evaluate.
            tutor_model (str): The name of the tutor model to consider.
            definition (str): The definition of the pedagogical dimension being evaluated.
            eval_instruction_rubric (str): The evaluation instruction rubric for the pedagogical dimension.
            correct_response (str): The correct response to use for evaluation.

        Returns:
            pd.DataFrame: A DataFrame containing the data with the appropriate prompt template.
        """

        system_content = ABS_SYSTEM_PROMPT

        inputs,  covids = [], []
        for example in data:
            if utils.should_skip_example(tutor_model, example):
                continue
            history = utils.get_all_history(example)
            tutresp = utils.get_response_text(tutor_model, example)
            goldresp = utils.get_gold_response_text(example)
            topic = utils.get_topic_text(example)
            covids.append(utils.get_conv_id(example))

            if self.with_ref:
                user_content =  ABSOLUTE_PROMPT.format(
                    previous_conversation = history,
                    response = tutresp,
                    reference_answer = goldresp,
                    rubric = eval_instruction_rubric,
                    definition = definition,
                    topic = topic,
                )
            else:
                user_content = ABSOLUTE_PROMPT_WO_REF.format(
                    previous_conversation = history,
                    response = tutresp,
                    rubric = eval_instruction_rubric,
                    definition = definition,
                    topic = topic,
                )
            
            message = [
                {'role': 'system', 'content': system_content},
                {'role': 'user', 'content': user_content},
            ]
            if hasattr(self.llm_model, "validate_vllm"):
                input_ = self._get_conversation_prompt(message)
            else:
                input_ = message
            inputs.append(input_)
        return inputs, covids

    def _get_eval_rubric(self, dimension: str) -> str:
        """
        Retrieve the evaluation rubric for a given pedagogical dimension.

        Arguments:
            dimension (str): The name of the pedagogical dimension (e.g., 'humanlikeness', 'Coherence').

        Returns:
            str: The evaluation rubric associated with the dimension.
        """
        rubrics = {
            'Mistake_Location': MISTAKE_LOCATION_RUBRIC,
            'Mistake_Identification': MISTAKE_IDENTIFICATION_RUBRIC,
            'Revealing_of_the_Answer': REVEALING_ANSWER_RUBRIC,
            'Providing_Guidance': PROVIDING_GUIDANCE_RUBRIC,
            'Actionability': ACTIONABILITY_RUBRIC,
            'Coherence': COHERENCE_RUBRIC,
            'Tutor_Tone': TUTOR_TONE_RUBRIC,
            'Humanlikeness': HUMANLIKENESS_RUBRIC,
        }

        return rubrics[dimension]
    
    def compute_scores(
        self,
        tutor_models: List[str],
        dimension: str,
        definition: str = None,
        eval_instruction_rubric: str = None,
        correct_response: str = None,
        save: bool = False,
        file_name: str = None,
        num_examples: int = -1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Generalized function to compute scores for a specific pedagogical dimension.

        Arguments:
            dimension (str): The name of the pedagogical dimension (e.g., 'humanlikeness', 'empathy').
            definition (str): The definition of the pedagogical dimension being evaluated.
            eval_instruction_rubric (str): The evaluation instruction rubric for the dimension.
            correct_response (str): The correct response to use for evaluation.
            save (bool): Whether to save the computed results to a file.
            file_name (str): Name of the output file (optional).

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]: A tuple containing the scores, error rates, scores, and the data.
        """
        if num_examples > 0:
            self.data = self.data[:num_examples]

        # Write function to check if all the tutor models are present in each example of the data
        # If not present, raise an error pointing out which tutor is missing in which examples
        if tutor_models:
            utils.sanity_check_tutor_models(self.data, tutor_models)

        if definition is None:
            definition = DEFINITIONS[dimension]

        if eval_instruction_rubric is None:
            eval_instruction_rubric = self._get_eval_rubric(dimension)

        scores_dict, error_dict, raw_scores = {}, {}, {}
        if self.evaluation_type == 'absolute':
            for tutor_model in tqdm(tutor_models, desc=f"Computing {dimension} scores for tutor models"):
                formatted_data, convids = self._get_data_with_prompt_template(
                    self.data, tutor_model, definition, eval_instruction_rubric, correct_response
                )
                scores, nan_percentage = batch_completions_with_retries(
                    self.llm_model, formatted_data, mode=self.evaluation_type, params=self.llm_model_parama
                )
                scores_dict[tutor_model] = round(np.mean([score for score in scores if score is not None]), 3)
                error_dict[tutor_model] = nan_percentage
                raw_scores[tutor_model] = scores

                utils.update_llm_annotation(self.data, tutor_model, scores, convids, f'{dimension}_{self.llm_model_name}')
        else:
            raise ValueError("Invalid evaluation type. Choose 'absolute' or 'relative'.")

        if save:
            if file_name is None:
                file_name = f"llm_{dimension}.json"
            utils.save_data(self.output_data_dir, file_name, self.data)

        return scores_dict, error_dict, raw_scores, self.data
    
    def compute_mistake_identification(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing mistake identification scores.
        """
        return self.compute_scores(tutor_models, dimension='Mistake_Identification', **kwargs)
    
    def compute_mistake_location(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing mistake location scores.
        """
        return self.compute_scores(tutor_models, dimension='Mistake_Location', **kwargs)
    
    def compute_revealing_of_the_answer(self, tutor_models,  **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing revealing of the answer scores.
        """
        return self.compute_scores(tutor_models, dimension='Revealing_of_the_Answer', **kwargs)
    
    def compute_providing_guidance(self,tutor_models,  **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing providing guidance scores.
        """
        return self.compute_scores(tutor_models, dimension='Providing_Guidance', **kwargs)

    def compute_actionability(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing actionability scores.
        """
        return self.compute_scores(tutor_models, dimension='Actionability', **kwargs)

    def compute_coherence(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing empathy scores.
        """
        return self.compute_scores(tutor_models, dimension='Coherence', **kwargs)
    
    def compute_tutor_tone(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing tutor tone scores.
        """
        return self.compute_scores(tutor_models, dimension='Tutor_Tone', **kwargs)
    
    def compute_humanlikeness(self, tutor_models, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], List[float], pd.DataFrame]:
        """
        Wrapper for computing humanlikeness scores.
        """
        return self.compute_scores(tutor_models, dimension='Humanlikeness', **kwargs)
    
    def _get_dim_method(self, dim: str):
        """
        Get the evaluation method based on the metric name.

        Parameters:
            metric (str): Evaluation metric.

        Returns:
            Callable: Evaluation method.
        """
        dim_methods = {
            "Mistake_Identification": self.compute_mistake_identification,
            "Mistake_Location": self.compute_mistake_location,
            "Revealing_of_the_Answer": self.compute_revealing_of_the_answer,
            "Providing_Guidance": self.compute_providing_guidance,
            "Actionability": self.compute_actionability,
            "Coherence": self.compute_coherence,
            "Tutor_Tone": self.compute_tutor_tone,
        }

        if dim not in dim_methods:
            raise ValueError(f"Unsupported metric: {dim}")
        return dim_methods[dim]

    def get_sample_examples_with_scores(
        self,
        dimension: str,
        tutor_model: str = "Expert",
        num_examples: int = 5,
        **kwargs  # Accepting additional keyword arguments
    ) -> pd.DataFrame:
        """
        Get examples with the given dimension scores for a specific tutor model.

        Arguments:
            dimension (str): The pedagogical dimension to consider.
            tutor_model (str): The name of the tutor model to consider.
            num_examples (int): The number of examples to return.

        Returns:
            pd.DataFrame: A DataFrame containing the examples with the pedagogical dimension scores for the given tutor model.
        """

        # Validate that num_examples is less than or equal to the available data
        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        # Calculate scores using the selected dimension and tutor model
        _, _, scores, _ = self.compute_scores([tutor_model], dimension, num_examples=num_examples, **kwargs)

        # Return the results as a DataFrame
        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model, scores[tutor_model], dim=dimension)


    def compare_tutors_scores(
        self,
        dimension: str,
        tutor_model1: str = "Expert",
        tutor_model2: str = "GPT4",
        num_examples: int = 5,
        **kwargs  # Accepting additional keyword arguments
    ) -> pd.DataFrame:
        """
        Compare the scores of two tutor models for a specific pedagogical dimension.

        Arguments:
            dimension (str): The pedagogical dimension to consider.
            tutor_model1 (str): The name of the first tutor model to compare.
            tutor_model2 (str): The name of the second tutor model to compare.
            num_examples (int): The number of examples to return.

        Returns:
            pd.DataFrame: A DataFrame containing the examples with the pedagogical dimension scores for the given tutor models.
        """

        if num_examples > len(self.data):
            raise ValueError(f"Number of examples should be less than or equal to {len(self.data)}")

        _, _, scores1, _ = self.compute_scores([tutor_model1], dimension, num_examples=num_examples, **kwargs)
        _, _, scores2, _ = self.compute_scores([tutor_model2], dimension, num_examples=num_examples, **kwargs)

        print(scores1)
        print(scores2)

        # Return the results as a DataFrame
        return utils.get_sample_dataframe(self.data[:num_examples], tutor_model1, scores1[tutor_model1],
                                          tutor_model2=tutor_model2, scores2=scores2[tutor_model2], dim=dimension)


    def get_llm_evaluation_report(
        self,
        tutor_models: List[str] = [
            "Expert", "Novice", "Llama31405B", "GPT4", 
            "Sonnet", "Phi3", "Llama318B", "Mistral", "Gemini"
        ],
        dimensions: List[str] = [
            'Mistake_Identification', 'Mistake_Location', 'Revealing_of_the_Answer', 
            'Providing_Guidance', 'Actionability', 'Coherence', 'Tutor_Tone', 'Humanlikeness'
        ],
        save_eval:bool = False, 
        save_report:bool = False,
        eval_file_name: str = 'llm_eval_all.json',
        report_file_name: str = 'llm_eval_report_all.csv',
    ) -> pd.DataFrame:
        """
        Generate the LLM evaluation report for specified tutor models and pedagogical dimensions.

        Arguments:
            tutor_models (List[str]): A list of tutor models to evaluate.
            dimensions (List[str]): A list of pedagogical dimensions to assess.

        Returns:
            pd.DataFrame: A DataFrame containing the LLM evaluation report with scores and error rates.
        """

        # Validate inputs
        if not tutor_models or not dimensions:
            raise ValueError("Both 'tutor_models' and 'dimensions' must be non-empty lists.")

        # Initialize an empty DataFrame with tutor models as the index and dimensions as columns
        report = pd.DataFrame(index=tutor_models, columns=dimensions)

        # Iterate through each tutor model and dimension to compute scores
        for tutor_model in tutor_models:
            for dimension in dimensions:
                try:
                    # Compute scores for the given tutor model and dimension
                    if save_eval:
                        scores, _, _, _ = self.compute_scores([tutor_model], dimension, save=True, file_name=eval_file_name)
                    scores, _, _, _ = self.compute_scores([tutor_model], dimension)

                    # Assign the score to the appropriate cell in the DataFrame
                    report.loc[tutor_model, dimension] = scores[tutor_model]
                except Exception as e:
                    print(f"Error evaluating {tutor_model} for {dimension}: {e}")

        # Convert the scores to numeric, handling any potential errors or missing values
        report = report.apply(pd.to_numeric, errors='coerce')

        if save_report:
            report_file = os.path.join(self.output_data_dir, report_file_name)
            report.to_csv(report_file, index=True)

        return report