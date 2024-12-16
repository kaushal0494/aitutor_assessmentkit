import os
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Tuple, Dict, Any, Union

from tqdm import tqdm

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

from aitutor_assessmentkit.helpers import utils
from aitutor_assessmentkit.helpers.constants import LABELS_ENCODING, BEST_AUTO_MODELS, BEST_LLM_MODELS
from aitutor_assessmentkit.helpers.utils import extract_average_human_scores

class Visualizer:
    def __init__(
        self,
        input_data_dir: str = None,
        output_data_dir: str = None,
        file_names: Union[List[str], str] = None,
        tutor_models: Union[List[str], str] = None,
    ):
        """
        Initialize the Visualizer.
        Arguments:
            input_data_dir (str): The path to the directory containing the input JSON data files.
            output_data_dir (str): The path to the directory where the output data will be saved as a JSON file.
            file_names (Union[List[str], str]): The names of the input JSON files.
            tutor_models (Union[List[str], str]): The names of the tutor models to consider during evaluation.
        """
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.file_names = file_names
        self.tutor_models = tutor_models
     
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
                for fname in tqdm(self.file_names, desc="Loading Annotated data"):
                    self.data.extend(utils.load_json_data(fname))

    from typing import List, Dict, Any

    def get_evaluation_scores(self, normalize=False) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Extract and organize evaluation scores from the data for human, automated, and LLM evaluations.

        Args:
            normalize (bool): Flag to normalize the evaluation scores.
        
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: A dictionary containing evaluation scores categorized 
            into human, automated, and LLM evaluations for each tutor model.
        """
        def extract_tutor_scores(normalize: bool, evaluation_type: str, dimensions: List[str]) -> Dict[str, Dict[str, Any]]:
            """
            Helper function to extract scores for a specific evaluation type and set of dimensions.
            
            Args:
                evaluation_type (str): The type of evaluation ('humaneval', 'autoeval', 'llmeval').
                dimensions (List[str]): List of dimensions to extract scores for.
                normalize (bool): Flag to normalize the evaluation scores.

            Returns:
                Dict[str, Dict[str, Any]]: A dictionary mapping tutor models to their scores by dimension.
            """

            evaluation_scores = {}
            overall_scores = []
            for tutor in self.tutor_models:
                tutor_scores = {}
                overall_tutor = []
                for dim in dimensions:
                    score = utils.extract_scores(self.data, tutor, dim, evaluation_type, normalize)
                    tutor_scores[dim] = score
                    overall_tutor.extend(score)
                tutor_scores[f'Overall_tutors'] = overall_tutor
                evaluation_scores[tutor] = tutor_scores
                overall_scores.extend(overall_tutor)

            evaluation_scores['overall'] = overall_scores
            return evaluation_scores

        # Extract human evaluation scores
        human_evaluation_scores = extract_tutor_scores(
            normalize,
            evaluation_type='humaneval',
            dimensions=list(LABELS_ENCODING.keys()),
        )

        # Extract automated evaluation scores
        auto_evaluation_scores = extract_tutor_scores(
            normalize,
            evaluation_type='autoeval',
            dimensions=[item[0] for item in BEST_AUTO_MODELS.values()],
        )

        # Extract LLM evaluation scores
        llm_evaluation_scores = extract_tutor_scores(
            normalize,
            evaluation_type='llmeval',
            dimensions=[item[0] for item in BEST_LLM_MODELS.values()],
        )

        return {
            "human_evaluation_scores": human_evaluation_scores,
            "auto_evaluation_scores": auto_evaluation_scores,
            "llm_evaluation_scores": llm_evaluation_scores
        }
    
    def _clean_scores(self, scores: List[float]) -> List[float]:
        """
        Clean the scores by removing any None values.
        
        Parameters:
        scores (List[float]): A list of scores.
        
        Returns:
        List[float]: A cleaned list of scores.
        """
        return [score for score in scores if score is not None]

    def get_average_evaluation_scores(self,  normalize=False) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Extract and calculate average evaluation scores from the data for human, automated, and LLM evaluations.

        Args:
            normalize (bool): Flag to normalize the evaluation scores.
        
        Returns:
            Dict[str, Dict[str, Dict[str, float]]]: A dictionary containing average evaluation scores categorized 
            into human, automated, and LLM evaluations for each tutor model.
        """
        def extract_tutor_average_scores(normalize: bool, evaluation_type: str, dimensions: List[str]) -> Dict[str, Dict[str, float]]:
            """
            Helper function to calculate average scores for a specific evaluation type and set of dimensions.
            
            Args:
                evaluation_type (str): The type of evaluation ('humaneval', 'autoeval', 'llmeval').
                dimensions (List[str]): List of dimensions to extract scores for.
                normalize (bool): Flag to normalize the evaluation scores.

            Returns:
                Dict[str, Dict[str, float]]: A dictionary mapping tutor models to their average scores by dimension.
            """
            evaluation_scores = {}
            overall_scores = []
            
            for tutor in self.tutor_models:
                tutor_scores = {}
                tutor_overall_sum = 0
                tutor_count = 0
                
                for dim in dimensions:
                    scores_temp = utils.extract_scores(self.data, tutor, dim, evaluation_type, normalize)
                    scores = self._clean_scores(scores_temp)
                    average_score = sum(scores) / len(scores) if scores else 0.0
                    tutor_scores[dim] = round(average_score, 3)
                    tutor_overall_sum += sum(scores)
                    tutor_count += len(scores)
                
                # Add the overall average score for the tutor
                tutor_scores[f'Overall_tutors'] = round(tutor_overall_sum / tutor_count, 3) if tutor_count > 0 else 0.0
                evaluation_scores[tutor] = tutor_scores
                overall_scores.append(tutor_overall_sum / tutor_count if tutor_count > 0 else 0.0)
            
            # Add the global overall average score
            evaluation_scores['overall'] = round(sum(overall_scores) / len(overall_scores), 3) if overall_scores else 0.0
            return evaluation_scores

        # Extract human evaluation average scores
        human_evaluation_scores = extract_tutor_average_scores(
            normalize,
            evaluation_type='humaneval',
            dimensions=list(LABELS_ENCODING.keys()),
        )

        # Extract automated evaluation average scores
        auto_evaluation_scores = extract_tutor_average_scores(
            normalize,
            evaluation_type='autoeval',
            dimensions=[item[0] for item in BEST_AUTO_MODELS.values()],
        )

        # Extract LLM evaluation average scores
        llm_evaluation_scores = extract_tutor_average_scores(
            normalize,
            evaluation_type='llmeval',
            dimensions=[item[0] for item in BEST_LLM_MODELS.values()],
        )

        return {
            "human_avg_scores": human_evaluation_scores,
            "auto_avg_scores": auto_evaluation_scores,
            "llm_avg_scores": llm_evaluation_scores
        }

    def get_evaluation_reprot(self, evaluation_type: str, normalize: bool = False) -> pd.DataFrame:
        """
        Get the evaluation report for the specified evaluation type.

        Parameters:
        - evaluation_type (str): Type of evaluation ('autoeval', 'llmeval', or 'both').
        - normalize (bool): Flag to normalize the evaluation scores.

        Returns:
        - pd.DataFrame: Evaluation report containing average scores for each tutor model and dimension.
        """
        if evaluation_type not in ['autoeval', 'llmeval', 'humeval']:
            raise ValueError("Invalid evaluation type specified. Choose 'autoeval', 'llmeval', or 'humeval'.")

        all_evaluation_scroes = self.get_average_evaluation_scores(normalize)

        if evaluation_type == 'autoeval':
            del all_evaluation_scroes['auto_avg_scores']['overall']
            evaluation_scores = all_evaluation_scroes['auto_avg_scores']
        elif evaluation_type == 'llmeval':
            del all_evaluation_scroes['llm_avg_scores']['overall']
            evaluation_scores = all_evaluation_scroes['llm_avg_scores']
        else:
            del all_evaluation_scroes['human_avg_scores']['overall']
            evaluation_scores = all_evaluation_scroes['human_avg_scores']

        return pd.DataFrame(evaluation_scores).T


    def label_distribution_plot(self, dimension: str, normalize: bool = False, plot_name: str = 'label_distribution.png', plot_type: str = 'box') -> str:
        """
        Generate a distribution plot (boxplot or violin plot) comparing human, automated, and LLM evaluation scores.

        Parameters:
        - dimension (str): The evaluation dimension to compare (prefix-based matching).
        - normalize (bool): Flag to normalize the evaluation scores.
        - plot_name (str): Name of the plot file to save.
        - plot_type (str): Type of plot to generate ('box' or 'violin').

        Returns:
        - str: Path to the saved plot image.
        """
        # Fetch evaluation scores
        score_dict = self.get_evaluation_scores(normalize)

        # Function to collect and clean scores
        def collect_scores(source_scores: dict) -> list:
            return [
                score
                for tutor in self.tutor_models
                for item in source_scores[tutor].keys()
                if item.startswith(dimension)
                for score in self._clean_scores(source_scores[tutor][item])
            ]

        # Collect scores for human, auto, and LLM evaluations
        human_scores = collect_scores(score_dict['human_evaluation_scores'])
        auto_scores = collect_scores(score_dict['auto_evaluation_scores'])
        llm_scores = collect_scores(score_dict['llm_evaluation_scores'])

        # Validate data
        if not any([human_scores, auto_scores, llm_scores]):
            raise ValueError(f"No scores found for the dimension '{dimension}'. Please verify the data or the dimension name.")

        # Organize scores into a dictionary for plotting
        data = {
            "Human": human_scores,
            "Automated": auto_scores,
            "LLM": llm_scores,
        }

        # Flatten the data into a list of tuples for DataFrame creation
        plot_data = [(source, score) for source, scores in data.items() for score in scores]

        # Create a DataFrame for seaborn plotting
        plot_df = pd.DataFrame(plot_data, columns=["Source", "Score"])

        # Generate the specified plot
        plt.figure(figsize=(10, 6))  # Set figure size
        if plot_type == 'box':
            sns.boxplot(x="Source", y="Score", data=plot_df, palette="Set2")  # Create the boxplot
            plt.title(f"Box Plot for Dimension: {dimension}", fontsize=16, fontweight='bold')
        elif plot_type == 'violin':
            sns.violinplot(x="Source", y="Score", data=plot_df, palette="Set3", inner="quartile")  # Create the violin plot
            plt.title(f"Violin Plot for Dimension: {dimension}", fontsize=16, fontweight='bold')
        elif plot_type == 'swarm':
            sns.swarmplot(x="Source", y="Score", data=plot_df, palette="Set1", size=6)
            plt.title(f"Swarm Plot for Dimension: {dimension}", fontsize=16, fontweight='bold')
        else:
            raise ValueError(f"Invalid plot type '{plot_type}'. Supported types are 'box' and 'violin'.")

        # Add axis labels and grid
        plt.xlabel("Evaluation Source", fontsize=12)
        plt.ylabel("Scores", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
 
        # Display the plot
        plt.show()

        # Define the output path and save the plot
        output_path = os.path.join(self.output_data_dir, plot_name)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()  # Close the plot to free up memory

        # Log the success of plot generation
        print(f"{plot_type.capitalize()} plot successfully saved at: {output_path}")

    def _create_spider_plot(self, categories, values_list, labels, title="Spider Plot", save_path=None):
        """
        Create a professional-looking spider plot with multiple datasets and a legend, using contrastive colors.

        Parameters:
        categories (list of str): Labels for the different axes.
        values_list (list of lists of float): List of datasets to plot. Each dataset is a list of values for each category.
        labels (list of str): Labels for each dataset.
        title (str): Title of the plot.
        save_path (str): Path to save the plot image. If None, the plot will be displayed.
        """
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop to close the plot

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Use the 'tab20' colormap to get a set of contrastive colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(values_list)))

        # Plot each dataset
        for values, label, color in zip(values_list, labels, colors):
            values += values[:1]  # Complete the loop
            ax.plot(angles, values, color=color, linewidth=2, linestyle='solid', label=label)
            ax.fill(angles, values, color=color, alpha=0.25)

        # Customize the plot appearance
        ax.set_yticklabels([])  # Hide radial axis labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, color='navy')

        ax.set_title(title, size=20, color='navy', y=1.1, fontweight='bold')

        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=10)

        ax.grid(color='gray', linestyle='dashed')  # Dashed grid for better clarity
        ax.spines['polar'].set_visible(False)  # Hide the polar spine

        # Adjust radial grid lines to a strip size of 0.2
        ax.set_rlabel_position(0)  # Set position of radial labels
        ax.set_ylim(0, 1)  # Set limit for radial axis (normalized values)
        ax.set_rticks(np.arange(0, 1.2, 0.2))  # Set radial ticks with 0.2 spacing

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, format='png', dpi=300)
            print(f"Spider plot saved at: {save_path}")
        else:
            plt.show()

    def spider_plot(self, normalize: bool, evaluation_type: str = 'autoeval', plot_type: str = 'tutor', plot_name: str = "spider_plot.png"):
        """
        Generate and save a spider graph comparing evaluation scores based on the evaluation type and plot type.

        Args:
            normalize (bool): Flag to normalize the evaluation scores.
            evaluation_type (str): Type of evaluation ('autoeval', 'llmeval', or 'humeval').
            plot_type (str): Type of plot ('tutor' or 'dimension').
            plot_name (str): Name of the output plot file.
        """
        # Read evaluation scores
        all_evaluation_scores = self.get_average_evaluation_scores(normalize)

        # Select evaluation scores based on the evaluation_type
        if evaluation_type == 'autoeval':
            evaluation_scores = all_evaluation_scores['auto_avg_scores']
        elif evaluation_type == 'llmeval':
            evaluation_scores = all_evaluation_scores['llm_avg_scores']
        else:
            evaluation_scores = all_evaluation_scores['human_avg_scores']

        # Initialize categories and labels
        if plot_type == 'tutor':
            # Extract labels and categories for tutor plot
            labels = list(evaluation_scores.keys())[:-1]
            categories = list(all_evaluation_scores['human_avg_scores'][labels[0]].keys())[:-1]

            # Prepare values for tutor plot
            values_list = []
            for tutor in labels:
                values_list.append(list(evaluation_scores[tutor].values())[:-1])

            title = f"Spider Plots (Tutors): {evaluation_type.capitalize()}"
        elif plot_type == 'dimension':
            # Extract categories and labels for dimension plot
            categories = list(evaluation_scores.keys())[:-1]
            labels = list(all_evaluation_scores['human_avg_scores'][categories[0]].keys())[:-1]

            # Prepare values for dimension plot
            values_list = []
            for dim in labels:
                temp_dim_score = []
                for tutor in categories:
                    for item_temp in evaluation_scores[tutor].keys():
                        if item_temp.startswith(dim):
                            temp_dim_score.append(evaluation_scores[tutor][item_temp])
                values_list.append(temp_dim_score)

            title = f"Spider Plots (Dimension): {evaluation_type.capitalize()}"

        # Define the path to save the plot
        save_path = os.path.join(self.output_data_dir, plot_name)

        # Generate the spider plot
        self._create_spider_plot(categories, values_list, labels, title=title, save_path=save_path)


    def compare_spider_plot(self, normalize: bool, evaluation_dim: str = 'Mistake_Identification', plot_name: str = "compare_spider.png"):
        """
        Generate a comparison spider plot of LLM evaluation scores across different dimensions.

        Args:
            normalize (bool): Flag to normalize the evaluation scores.
            evaluation_dim (str): The dimension of evaluation to compare (e.g., 'Mistake_Identification').
            plot_name (str): Name of the output plot file (defaults to 'compare_spider.png').
        """
        
        # Read evaluation scores
        all_evaluation_scores = self.get_average_evaluation_scores(normalize)

        # Extract scores for Human, Auto, and LLM
        hum_scores = all_evaluation_scores['human_avg_scores']
        auto_scores = all_evaluation_scores['auto_avg_scores']
        llm_scores = all_evaluation_scores['llm_avg_scores']

        # Labels for the plot
        labels = ["Auto", "Human", "LLM"]
        
        # Categories for comparison (excluding the last element for symmetry in the plot)
        categories = list(hum_scores.keys())[:-1]

        # Function to extract values for a given evaluation dimension across categories
        def extract_values(scores, evaluation_dim):
            values = []
            for tutor in categories:
                for item_dim in scores[tutor].keys():
                    if item_dim.startswith(evaluation_dim):
                        values.append(scores[tutor][item_dim])
            return values

        # Extract values for each dimension (Human, Auto, LLM)
        val1 = extract_values(hum_scores, evaluation_dim)
        val2 = extract_values(auto_scores, evaluation_dim)
        val3 = extract_values(llm_scores, evaluation_dim)

        # Prepare the values list for the plot
        values_list = [val2, val1, val3]

        # Title for the plot
        title = f"Comparison Spider Plots: {evaluation_dim}"

        # Define the path to save the plot
        save_path = os.path.join(self.output_data_dir, plot_name)

        # Generate the spider plot
        self._create_spider_plot(categories, values_list, labels, title=title, save_path=save_path)

    #TODO: Implement the spider plot for the comparison of the evaluation scores for multiple metrics for each dimensions

    def _spearman_bootstrap_ci(self, model_scores, human_labels, n_iterations=1000, ci=95):
        """
        Compute Spearman's rank-order correlation with bootstrapped confidence intervals.
        
        Parameters:
        - model_scores (list or array): Scores from the model.
        - human_labels (list or array): Human-labeled scores.
        - n_iterations (int): Number of bootstrap samples (default is 1000).
        - ci (float): Confidence interval level (default is 95).
        
        Returns:
        - spearman_corr (float): Spearman's correlation for the original data.
        - conf_interval (tuple): Lower and upper bounds of the confidence interval.
        """
        # Clean inputs: Remove None or NaN values and ensure alignment
        model_scores, human_labels = self._clean_scores_jointly(model_scores, human_labels)
        
        if len(model_scores) <= 1 or len(human_labels) <= 1:
            # Handle cases where not enough data remains after cleaning
            return float('nan'), (float('nan'), float('nan'))

        # Check for constant input in the original data
        if len(set(model_scores)) == 1 or len(set(human_labels)) == 1:
            # Spearman correlation is undefined for constant inputs
            return float('nan'), (float('nan'), float('nan'))

        # Original Spearman correlation
        spearman_corr, _ = spearmanr(model_scores, human_labels)

        # Bootstrapping
        bootstrapped_corrs = []
        n = len(model_scores)
        for _ in range(n_iterations):
            # Generate a random sample with replacement
            indices = np.random.choice(range(n), size=n, replace=True)
            sample_model_scores = np.array(model_scores)[indices]
            sample_human_labels = np.array(human_labels)[indices]

            # Check for constant inputs in bootstrap samples
            if len(set(sample_model_scores)) == 1 or len(set(sample_human_labels)) == 1:
                # Skip iteration if correlation is undefined
                continue

            # Calculate Spearman correlation for the sample
            boot_corr, _ = spearmanr(sample_model_scores, sample_human_labels)
            bootstrapped_corrs.append(boot_corr)

        # Ensure we have valid bootstrap correlations
        if not bootstrapped_corrs:
            return spearman_corr, (float('nan'), float('nan'))

        # Calculate confidence interval
        lower_bound = np.percentile(bootstrapped_corrs, (100 - ci) / 2)
        upper_bound = np.percentile(bootstrapped_corrs, 100 - (100 - ci) / 2)
        conf_interval = (lower_bound, upper_bound)

        return spearman_corr, conf_interval

    
    def _compute_accuracy(self, ground_truth: list, predictions: list) -> float:
        """
        Compute the accuracy score between two lists of inputs.

        This function calculates the proportion of matching values between
        the ground truth and the predictions.

        Args:
            ground_truth (list): The list of true values.
            predictions (list): The list of predicted values.

        Returns:
            float: The accuracy score, a value between 0 and 1.

        Raises:
            ValueError: If the input lists have different lengths.
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("The input lists must have the same length.")

        # Count matching values
        matches = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == pred)
        
        # Compute accuracy as the proportion of matches
        accuracy = matches / len(ground_truth) if ground_truth else 0.0

        return accuracy

    def _clean_scores_jointly(self, hscores: List[float], ascores: List[float]) -> Tuple[List[float], List[float]]:
        """
        Clean the scores by removing any None values and ensuring equal length.
        
        Parameters:
        - hscores (List[float]): A list of human-labeled scores.
        - ascores (List[float]): A list of automated model scores.
        
        Returns:
        - Tuple[List[float], List[float]]: Cleaned lists of scores for both inputs.
        """
        if len(hscores) != len(ascores):
            raise ValueError("Human scores and automated scores must have the same length.")
        
        # Filter out None values while maintaining alignment between the two lists
        cleaned_hscores, cleaned_ascores = zip(
            *[(h, a) for h, a in zip(hscores, ascores) if h is not None and a is not None]
        )

        return list(cleaned_hscores), list(cleaned_ascores)

    def peformance_report(self, normalize: bool, metric: str = "correlation", correlation_with: str = "autoeval") -> pd.DataFrame:
        """
        Generate an evaluation report comparing human and automated evaluation scores for tutor models.

        This function computes either the Spearman's rank correlation or accuracy between human evaluation 
        scores and automated evaluation scores (either 'autoeval' or 'llmeval') for each tutor model and 
        dimension. Optionally, it can also plot the results.

        Args:
            normalize (bool): Flag to indicate whether to normalize the evaluation scores.
            metric (str): The evaluation metric to compute. Options are 'correlation' or 'accuracy'.
            correlation_with (str): The evaluation type to compute the metric with. 
                Options are 'autoeval' (auto evaluation) or 'llmeval' (LLM evaluation).
            plot (bool): Flag to indicate whether to plot the results.

        Returns:
            pd.DataFrame: A DataFrame containing the computed scores (correlation or accuracy) for each tutor model and dimension.

        Raises:
            ValueError: If the `metric` or `correlation_with` argument is invalid.
            KeyError: If required keys are missing in the evaluation scores.
        """
        # Validate arguments
        if metric not in ['correlation', 'accuracy']:
            raise ValueError("Invalid metric specified. Choose 'correlation' or 'accuracy'.")
        if correlation_with not in ['autoeval', 'llmeval']:
            raise ValueError("Invalid evaluation type specified. Choose 'autoeval' or 'llmeval'.")

        # Retrieve all evaluation scores
        all_evaluation_scores = self.get_evaluation_scores(normalize)
        hum_scores = all_evaluation_scores.get('human_evaluation_scores', {})

        # Extract human and automated scores based on the evaluation type
        if correlation_with == 'autoeval':
            auto_scores = all_evaluation_scores.get('auto_evaluation_scores', {})
        else:
            auto_scores = all_evaluation_scores.get('llm_evaluation_scores', {})

        # Check if both human and auto scores are available
        if not hum_scores or not auto_scores:
            raise KeyError(f"Missing evaluation scores for {correlation_with} or human evaluation.")

        collect_all_scores = []

        # Compute scores for each tutor and dimension
        for tutor in list(hum_scores.keys())[:-1]:
            tutor_scores = []
            for dim in list(hum_scores[tutor].keys()):
                hum_score_values = hum_scores[tutor][dim]
                for temp_dim in auto_scores[tutor].keys():
                    if temp_dim.startswith(dim):
                        auto_score_values = auto_scores[tutor][temp_dim]
                _hum_score_values, _auto_score_values = self._clean_scores_jointly(hum_score_values, auto_score_values)
                
                # Compute correlation or accuracy
                if metric == "correlation":
                    score = round(self._spearman_bootstrap_ci(_hum_score_values, _auto_score_values)[0], 3)
                else:  # Accuracy
                    score = round(self._compute_accuracy(_hum_score_values, _auto_score_values), 3)
                
                tutor_scores.append(score)
            collect_all_scores.append([tutor] + tutor_scores)

        # Compute overall scores across all tutors for each dimension
        overall_scores = []
        for dim in list(hum_scores[next(iter(hum_scores))].keys()):  # Assuming all tutors have the same dimensions
            hum_score_values_all = []
            auto_score_values_all = []
            for tutor in hum_scores.keys():
                if dim in hum_scores[tutor]:
                    hum_score_values = hum_scores[tutor][dim]
                    for temp_dim in auto_scores[tutor].keys():
                        if temp_dim.startswith(dim):
                            auto_score_values = auto_scores[tutor][temp_dim]
                    _hum_score_values, _auto_score_values = self._clean_scores_jointly(hum_score_values, auto_score_values)
                    hum_score_values_all.extend(_hum_score_values)
                    auto_score_values_all.extend(_auto_score_values)
            
            if metric == "correlation":
                score = round(self._spearman_bootstrap_ci(hum_score_values_all, auto_score_values_all)[0], 3)
            else:  # Accuracy
                score = round(self._compute_accuracy(hum_score_values_all, auto_score_values_all), 3)

            overall_scores.append(score)

        collect_all_scores.append(["Overall_dim"] + overall_scores)

        # Convert results into a DataFrame
        result_df = pd.DataFrame(
            collect_all_scores,
            columns=['Tutor'] + list(hum_scores[next(iter(hum_scores))].keys())
        )
       
        if metric == "correlation":
            overall = round(self._spearman_bootstrap_ci(hum_scores['overall'], auto_scores['overall'])[0], 3)
        else:  # Accuracy
            overall = round(self._compute_accuracy(hum_scores['overall'], auto_scores['overall']), 3)

        return overall, result_df

    def plot_correlation(self, normalize: bool, correlation_with: str = "autoeval", dimension: str = "Mistake_Identification", plot_name: str = "correlation_plot.png"):
        """
        Plots and saves a correlation plot between scores from two models.
        
        Args:
        - model_1_scores (list or np.array): Scores from the first model.
        - model_2_scores (list or np.array): Scores from the second model.
        - save_path (str): The file path to save the plot.
        """

        if correlation_with not in ['autoeval', 'llmeval']:
            raise ValueError("Invalid evaluation type specified. Choose 'autoeval' or 'llmeval'.")

        # Retrieve all evaluation scores
        all_evaluation_scores = self.get_evaluation_scores(normalize)
        hum_scores = all_evaluation_scores.get('human_evaluation_scores', {})

        # Extract human and automated scores based on the evaluation type
        if correlation_with == 'autoeval':
            auto_scores = all_evaluation_scores.get('auto_evaluation_scores', {})
        else:
            auto_scores = all_evaluation_scores.get('llm_evaluation_scores', {})

        # Check if both human and auto scores are available
        if not hum_scores or not auto_scores:
            raise KeyError(f"Missing evaluation scores for {correlation_with} or human evaluation.")
        
        hum_score_values_all = []
        auto_score_values_all = []
        for tutor in hum_scores.keys():
            if dimension in hum_scores[tutor]:
                hum_score_values = hum_scores[tutor][dimension]
                for temp_dim in auto_scores[tutor].keys():
                    if temp_dim.startswith(dimension):
                        auto_score_values = auto_scores[tutor][temp_dim]
                _hum_score_values, _auto_score_values = self._clean_scores_jointly(hum_score_values, auto_score_values)
                hum_score_values_all.extend(_hum_score_values)
                auto_score_values_all.extend(_auto_score_values)
        
        # # Ensure the scores are numpy arrays for consistency
        hscore_arr = np.array(hum_score_values_all)
        ascore_arr = np.array(auto_score_values_all)
        
        # # Create a DataFrame for better handling of the data
        data = pd.DataFrame({
            'Human Score': hscore_arr,
            'Model Score': hscore_arr, 
        })

        sns.set(style="whitegrid", palette="muted")
        
        # Create a figure
        plt.figure(figsize=(8, 6))
        
        # Create scatter plot with different colors for points
        scatter = plt.scatter(
           hscore_arr  , ascore_arr, 
            c=hscore_arr , cmap='viridis',  # Color by Model 1 Scores
            s=100, edgecolors="w", alpha=0.7  # Customize scatter point size and transparency
        )
        
        # Add color bar to show the color mapping
        plt.colorbar(scatter, label='Human Score')

        # Plot the regression line
        sns.regplot(x='Human Score', y='Model Score', data=data, 
                    scatter=False, line_kws={'color': 'red', 'lw': 2})
        
        # Calculate the correlation coefficient
        correlation_coefficient = round(self._spearman_bootstrap_ci(hscore_arr, ascore_arr)[0], 3)
        
        # Display the correlation coefficient in the title
        plt.title(f"Correlation Plot\nCorrelation Coefficient: {correlation_coefficient:.2f}", fontsize=16)
        
        # Labels for axes
        plt.xlabel("Human Score", fontsize=14)
        plt.ylabel("Model Score", fontsize=14)
        
        # Display the plot
        plt.show()
        save_path = os.path.join(self.output_data_dir, plot_name)
        
        # Save the plot as a high-quality image
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        print(f"Correlation plot saved at: {save_path}")

    def _plot_binned_heatmap(self, discrete_values, continuous_values, save_path, bins, plot_title="Heatmap for Binned Continuous Values"):
        """
        Plots a heatmap to show the binned distribution of continuous values across discrete categories.
        
        Args:
            discrete_values (list): List of discrete categories (e.g., different groups or classes).
            continuous_values (list): List of continuous values associated with each category.
            bins (int): Number of bins for continuous values.
            plot_title (str): Title of the plot.
        """
        if isinstance(bins, str):
            raise TypeError(f"Expected 'bins' to be an integer, but got {type(bins)}.")
        
        # Bin the continuous values and generate the labels showing range of values in each bin
        bin_labels = [f"{round(interval.left, 2)} - {round(interval.right, 2)}" for interval in pd.cut(continuous_values, bins=bins).categories]
        binned_values = pd.cut(continuous_values, bins=bins, labels=bin_labels)

        # Create a DataFrame to organize the data
        data = pd.DataFrame({
            'Discrete Values': discrete_values,
            'Binned Values': binned_values
        })
        
        # Count the occurrences of each binned value per discrete category
        heatmap_data = pd.crosstab(data['Discrete Values'], data['Binned Values'])
        
        # Create the heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="d", cbar_kws={'label': 'Frequency'})
        
        # Customize the plot
        plt.title(plot_title, fontsize=16)
        plt.xlabel('Binned Uptake Values', fontsize=12)
        plt.ylabel('Human Annotation', fontsize=12)

        plt.show()

        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the plot to free memory
        print(f"Plot saved at {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def interpretability_plot(self, normalize: bool, bins: int = 5, interpwith: str = 'autoeval', evaluation_dim: str = 'Providing_Guidance', plot_name: str = "interpretability_plot.png"):
        """
        Generate an interpretability plot comparing human and automated evaluation scores for a given dimension.

        Args:
            normalize (bool): Flag to normalize the evaluation scores.
            interpwith (str): The evaluation type to compute the interpretability with ('autoeval' or 'llmeval').
            evaluation_dim (str): The dimension of evaluation to compare (e.g., 'Mistake_Identification').
            plot_name (str): Name of the output plot file (defaults to 'interpretability_plot.png').
        """
        # Retrieve all evaluation scores
        all_evaluation_scores = self.get_evaluation_scores(normalize)

        # Extract scores for human and automated evaluations
        hum_scores = all_evaluation_scores.get('human_evaluation_scores', {})
        if interpwith == 'autoeval':
            auto_scores = all_evaluation_scores.get('auto_evaluation_scores', {})
        else:
            auto_scores = all_evaluation_scores.get('llm_evaluation_scores', {})

        # Extract relevant scores based on the evaluation dimension
        extract_human_score = [
            score for tutor in self.tutor_models
            for score in hum_scores.get(tutor, {}).get(evaluation_dim, [])
        ]

        extract_auto_score = [
            score for tutor in self.tutor_models
            for dim, scores in auto_scores.get(tutor, {}).items()
            if dim.startswith(evaluation_dim)
            for score in scores
        ]

        # Ensure the lists of scores are of equal length
        assert len(extract_human_score) == len(extract_auto_score), "Mismatch in number of scores."

        # Prepare plot title and save path
        title = f"Interpretability Plot: HumEval vs. {interpwith.capitalize()} ({evaluation_dim})"
        save_path = os.path.join(self.output_data_dir, plot_name)

        # Generate and save the interpretability plot
        self._plot_binned_heatmap(extract_human_score, extract_auto_score, save_path, bins, plot_title=title)

    def _get_itractive_scores(self, all_lables, eval_type):
        """
        TODO: Get the interactive scores for the evaluation data.
        """
        collect_labels = []
        if eval_type == 'autoeval':
            for key, val in BEST_AUTO_MODELS.items():
                for item in all_lables:
                    if item[0] == val[0]:
                        collect_labels.append(item[1])
        else:
            for key, val in BEST_LLM_MODELS.items():
                for item in all_lables:
                    if item[0] == val[0]:
                        collect_labels.append(item[1])
        return collect_labels
    
    def user_interaction(self, normalize: bool = False):
        """
        Display a user interaction interface to explore the evaluation data interactively.

        Args:
        - select_example (int): Number of examples to display in the interface.
        """

        # Step 1: Select an example number
        select_example = int(input(f"Please enter a dialogue index number in the range [0 - {len(self.data) - 1}] to explore: "))
        if select_example < 0 or select_example >= len(self.data):
            print("Invalid dialogue index selected.")
            return
        
        print(f"Conversation Topic: {self.data[select_example]['Topic']}")
        conv_history = self.data[select_example]['conversation_history'].replace('||| ||| ||| ', '\n')[9:].replace(' ||| ||| |||', '\n').replace('tutor', 'Tutor').replace('student', 'Student')
        print(f"Conversation History:\n{conv_history.capitalize()}")

        # Step 2: Select a tutor
        print("Available Tutors:", self.tutor_models)
        selected_tutor = input(f"Select a tutor from the list {self.tutor_models}: ")

        if selected_tutor not in self.tutor_models:
            print("Invalid tutor selected.")
            return

        print(f"Showing the next tutor response with {selected_tutor}")
        print(f"Next Tutor Response: {self.data[select_example]['anno_llm_responses'][selected_tutor]['response']}")
        
        # Step 3: Select evaluation type (human or automated)
        eval_type = input("Select evaluation type (human/auto/llm): ").lower()

        if eval_type not in ['human', 'auto', 'llm']:
            print("Invalid evaluation type selected.")
            return

        if eval_type == 'human':
            print("Human Evaluation:")
            evaluation = list(self.data[select_example]['anno_llm_responses'][selected_tutor]['annotation'].values())
        elif eval_type == 'auto':
            print("Automated Evaluation:")
            evaluation = self._get_itractive_scores(self.data[select_example]['anno_llm_responses'][selected_tutor]['auto_annotation'].items(), 'autoeval')
        else:
            print("LLM Evaluation:")
            evaluation = self._get_itractive_scores(self.data[select_example]['anno_llm_responses'][selected_tutor]['llm_annotation'].items(), 'llmeval')

        return pd.DataFrame([evaluation], columns=list(self.data[select_example]['anno_llm_responses'][selected_tutor]['annotation'].keys()))