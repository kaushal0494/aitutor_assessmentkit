import os
import pandas as pd
from typing import List, Dict, Any, Tuple, Union

from aitutor_assessmentkit.autoevaluator import (
    autoeval, 
    AutoMistakeIdentificationEvaluator,
    AutoMistakeLocationEvaluator,
    AutoRevealingOfTheAnswerEvaluator,
    AutoProvidingGuidanceEvaluator,
    AutoActionabilityEvaluator,
    AutoCoherenceEvaluator, 
    AutoTutorToneEvaluator,
    AutoHumanlikenessEvaluator, 
)

from aitutor_assessmentkit.helpers import utils
from aitutor_assessmentkit.helpers.constants import BEST_AUTO_MODELS, ALL_AUTO_MODELS

class AutoEvaluationReport(autoeval.AutoEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the AutoEvaluationReport object.
        Inherits from AutoEvaluator.
        
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results for each metric.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize each evaluator with provided arguments
        self.evaluators = {
            'Mistake_Identification': AutoMistakeIdentificationEvaluator(*args, **kwargs),
            'Mistake_Location': AutoMistakeLocationEvaluator(*args, **kwargs),
            'Providing_Guidance': AutoProvidingGuidanceEvaluator(*args, **kwargs),
            'Revealing_of_the_Answer': AutoRevealingOfTheAnswerEvaluator(*args, **kwargs),
            'Actionability': AutoActionabilityEvaluator(*args, **kwargs),
            'Coherence': AutoCoherenceEvaluator(*args, **kwargs),
            'Tutor_Tone': AutoTutorToneEvaluator(*args, **kwargs),
            'Humanlikeness': AutoHumanlikenessEvaluator(*args, **kwargs),
        }

    def get_automated_evaluation_report_with_best_models(
            self, 
            save_eval:bool = False,
            save_report:bool = False,
            eval_file_name: str = 'auto_eval_best.json',
            report_file_name: str = 'auto_eval_report_best.csv'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate the automated evaluation report for the given input data.

        Arguments:
            save (bool): Whether to save the data to the output directory.
            file_name (str): The name of the file to save the data.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: A tuple containing the evaluation report and the updated.

        """
        # Collect evaluation results from each evaluator
        auto_evaluation_report = {}

        for dim, evaluator in self.evaluators.items():
            metric_scores, _, self.data = evaluator.compute(self.data, save=False, metrics=BEST_AUTO_MODELS[dim])
            for metric, scores in metric_scores.items():
                    auto_evaluation_report[metric] = scores
        if save_eval:
            utils.save_data(self.output_data_dir, eval_file_name, self.data)

        if save_report:
            report_file = os.path.join(self.output_data_dir, report_file_name) 
            pd.DataFrame(auto_evaluation_report).to_csv(report_file, index=True)

        return pd.DataFrame(auto_evaluation_report), self.data
    
    def get_automated_evaluation_report_with_all_models(
            self, 
            save_eval:bool = False, 
            save_report:bool = False,
            eval_file_name: str = 'auto_eval_all.json',
            report_file_name: str = 'auto_eval_report_all.csv',
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate the automated evaluation report for the given input data.

        Arguments:
            save (bool): Whether to save the data to the output directory.
            file_name (str): The name of the file to save the data.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: A tuple containing the evaluation report and the updated.

        """
        # Collect evaluation results from each evaluator
        auto_evaluation_report = {}

        for dim, evaluator in self.evaluators.items():
            metric_scores, _, self.data = evaluator.compute(self.data, save=False, metrics=ALL_AUTO_MODELS[dim])
            for metric, scores in metric_scores.items():
                    auto_evaluation_report[metric] = scores
        if save_eval:
            utils.save_data(self.output_data_dir, eval_file_name, self.data)

        if save_report:
            report_file = os.path.join(self.output_data_dir, report_file_name)
            pd.DataFrame(auto_evaluation_report).to_csv(report_file, index=True)

        return pd.DataFrame(auto_evaluation_report), self.data