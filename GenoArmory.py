import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, NamedTuple
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertConfig, AutoModelForSequenceClassification
import argparse
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
import datetime
import pandas as pd
import subprocess
import os
from collections import Counter
from matplotlib.ticker import MaxNLocator


@dataclass
class AdversarialAttack:
    index: int
    datasets: str
    total_number_of_sequences: int
    asr: float
    average_queries: Union[float, str]
    origin_acc: float
    after_attack_acc: float

@dataclass
class AttackParameters:
    method: str
    model: str
    attack_type: str
    defense: Optional[str]
    attack_success_rate: float
    total_number_of_sequences: int
    evaluation_date: str
    method_parameters: Dict[str, Any]

@dataclass
class AttackMetadata:
    parameters: AttackParameters
    adversarial_attacks: List[AdversarialAttack] = field(default_factory=list)

    @staticmethod
    def from_json(data: dict) -> "AttackMetadata":


        parameters = AttackParameters(
            method=data["parameters"]["method"],
            model=data["parameters"]["model"],
            attack_type=data["parameters"]["attack_type"],
            defense=data["parameters"]["defense"],
            attack_success_rate=data["parameters"]["attack_success_rate"],
            total_number_of_sequences=data["parameters"]["total_number_of_sequences"],
            evaluation_date=data["parameters"]["evaluation_date"],
            method_parameters=data["parameters"]["method_parameters"]
        )

        attacks = [
            AdversarialAttack(**entry)
            for entry in data["adversarial_attacks"]
        ]
        return AttackMetadata(parameters=parameters, adversarial_attacks=attacks)


@dataclass
class DefenseDatasetResult:
    index: int
    dataset: str
    total_sequences: int
    average_queries: float
    original_accuracy: float
    accuracy_after_attack: float
    defense_success_rate: float


@dataclass
class DefenseMetadata:
    method: str
    model: str
    attack_type: str
    defense: str
    total_sequences: int
    evaluation_date: str
    method_parameters: Dict[str, Any]
    overall_defense_success_rate: float
    results_by_dataset: List[DefenseDatasetResult]

    @staticmethod
    def from_json(data: Dict) -> "DefenseMetadata":
        params = data["parameters"]
        dataset_results = [
            DefenseDatasetResult(
                index=entry["index"],
                dataset=entry["datasets"],
                total_sequences=entry["total_number_of_sequences"],
                average_queries=entry["average_queries"],
                original_accuracy=entry["origin_acc"],
                accuracy_after_attack=entry["after_attack_acc"],
                defense_success_rate=entry["dsr"]
            )
            for entry in data["adversarial_defense"]
        ]
        return DefenseMetadata(
            method=params["method"],
            model=params["model"],
            attack_type=params["attack_type"],
            defense=params["defense"],
            total_sequences=params["total_number_of_sequences"],
            evaluation_date=params["evaluation_date"],
            method_parameters=params["method_parameters"],
            overall_defense_success_rate=params["defense_success_rate"],
            results_by_dataset=dataset_results,
        )


@dataclass
class AttackInfo:
    """Information about an attack attempt"""

    index: int
    method: str
    target_sequence: str
    modified_sequence: str
    target_label: Optional[int]
    success: bool
    number_of_queries: int
    queries_to_success: int
    confidence_original: float
    confidence_modified: float


@dataclass
class DefenseInfo:
    """Information about a defense attempt"""

    index: int
    method: str
    original_sequence: str
    protected_sequence: str
    protection_score: float
    computational_cost: float
    robustness_score: float


class AttackArtifact:
    """Container for attack artifacts"""

    def __init__(self, method: str, model_name: str, description: str = ""):
        self.metadata = AttackMetadata(
            method=method, model_name=model_name, description=description
        )
        self.attacks: List[AttackInfo] = []

    def add_attack(self, attack: AttackInfo):
        self.attacks.append(attack)
        self._update_metadata()

    def _update_metadata(self):
        """Update metadata statistics"""
        self.metadata.total_sequences = len(self.attacks)
        if self.attacks:
            self.metadata.success_rate = sum(a.success for a in self.attacks) / len(
                self.attacks
            )
            self.metadata.average_queries = sum(
                a.number_of_queries for a in self.attacks
            ) / len(self.attacks)

    def __getitem__(self, idx: int) -> AttackInfo:
        return self.attacks[idx]

    def __len__(self) -> int:
        return len(self.attacks)


class DefenseArtifact:
    """Container for defense artifacts"""

    def __init__(self, method: str, model_name: str, description: str = ""):
        self.metadata = DefenseMetadata(
            method=method, model_name=model_name, description=description
        )
        self.defenses: List[DefenseInfo] = []

    def add_defense(self, defense: DefenseInfo):
        self.defenses.append(defense)
        self._update_metadata()

    def _update_metadata(self):
        """Update metadata statistics"""
        self.metadata.total_sequences = len(self.defenses)
        if self.defenses:
            self.metadata.average_protection = sum(
                d.protection_score for d in self.defenses
            ) / len(self.defenses)
            self.metadata.average_robustness = sum(
                d.robustness_score for d in self.defenses
            ) / len(self.defenses)

    def __getitem__(self, idx: int) -> DefenseInfo:
        return self.defenses[idx]

    def __len__(self) -> int:
        return len(self.defenses)



class GenoArmory:
    def __init__(
        self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GenoArmory with a model and tokenizer

        Args:
            model: The GFM to be used
            tokenizer: The tokenizer for the model
            device: Device to run the model on (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        if self.model is not None:
            self.model.to(device)

        # Store artifacts
        self._attack_artifacts: Dict[str, AttackArtifact] = {}
        self._defense_artifacts: Dict[str, DefenseArtifact] = {}

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None, num_labels: int = 2):
        """Load GenoArmory from a pretrained model"""
        if 'bert' in model_path:
            config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
        else:
            config = AutoConfig.from_pretrained(model_path, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            model,
            tokenizer,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def read_attack_artifact(
        self, method: str, model_name: str, description: str = ""
    ) -> AttackArtifact:
        """Read attack artifacts for a specific method and model"""
        key = f"{method}_{model_name}"
        if key not in self._attack_artifacts:
            # Load artifact from storage or create new
            self._attack_artifacts[key] = AttackArtifact(
                method, model_name, description
            )
        return self._attack_artifacts[key]

    def read_defense_artifact(
        self, method: str, model_name: str, description: str = ""
    ) -> DefenseArtifact:
        """Read defense artifacts for a specific method and model"""
        key = f"{method}_{model_name}"
        if key not in self._defense_artifacts:
            # Load artifact from storage or create new
            self._defense_artifacts[key] = DefenseArtifact(
                method, model_name, description
            )
        return self._defense_artifacts[key]

    def get_attack_metadata(
        self, method: str, model_name: str
    ) -> Optional[AttackMetadata]:
        """Get metadata for a specific attack method and model, loading from disk if available."""
        metadata_path = f"./metadata/{method}/{model_name}.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                data = json.load(f)
            # Convert loaded dict to AttackMetadata (handles missing fields with defaults)
            return AttackMetadata.from_json(data)
        else:
            return None


    def get_defense_metadata(
        self, method: str, model_name: str, attack_method: str
    ) -> Optional[DefenseMetadata]:
        """Get metadata for a specific defense method and model"""
        metadata_path = f"./metadata/{method}/{model_name}-{attack_method}.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                data = json.load(f)
            return DefenseMetadata.from_json(data)
        else:
            return None

    def visualization(self, **kwargs) -> None:
        """
        Visualize the frequency distribution of changes from a folder of JSON files.
        Args (in kwargs):
            folder_path: Path to the folder containing JSON files. (required)
            output_pdf_path: Path to save the PDF (optional, defaults to frequency.pdf in folder_path)
        """
        folder_path = kwargs["folder_path"]
        output_pdf_path = kwargs.get("output_pdf_path", os.path.join(folder_path, "frequency.pdf"))
        person_name = os.path.basename(os.path.normpath(folder_path))

        # Initialize a Counter to count positions
        position_counter = Counter()

        # Iterate through all JSON files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    # Check if the root is a list
                    if isinstance(data, list):
                        for entry in data:
                            if isinstance(entry, dict) and entry.get("success") == 4:
                                for change in entry.get("changes", []):
                                    position_counter[change[0] + 1] += 1
                    elif isinstance(data, dict) and data.get("success") == 4:
                        for change in data.get("changes", []):
                            position_counter[change[0] + 1] += 1

        # Set font to Times New Roman
        plt.rcParams['font.family'] = 'DeJavu Serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        # Plot the frequency distribution as a bar chart
        positions = sorted(position_counter.keys())
        frequencies = [position_counter[pos] for pos in positions]

        plt.bar(positions, frequencies, color='#ff7f0e')
        plt.xlabel('Position', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.title(f'{person_name}', fontsize=18)
        plt.xticks(positions, fontsize=16)
        plt.yticks(fontsize=16)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_pdf_path)
        plt.close()
        print(f"Processing complete for folder: {folder_path}")

    def attack(
        self,
        attack_method: str,
        **kwargs,
    ) -> AttackInfo:
        """
        Perform attacks on GFM

        Args:
            attack_method: One of ['bertattack', 'textfooler', 'pgd', 'fimba']
            **kwargs: Additional arguments for specific attack methods

        Returns:
            AttackInfo object containing attack results
        """
        attack_methods = {
            "bertattack": self._bert_attack,
            "textfooler": self._textfooler_attack,
            "pgd": self._pgd_attack,
            "fimba": self._fimba_attack,
        }

        if attack_method not in attack_methods:
            raise ValueError(
                f"Attack method must be one of {list(attack_methods.keys())}"
            )

        attack_methods[attack_method](**kwargs)

        # # Create attack info
        # attack_info = AttackInfo(
        #     index=len(
        #         self._attack_artifacts.get(
        #             f"{attack_method}_{self.model.name_or_path}", []
        #         )
        #     ),
        #     method=attack_method,
        #     target_sequence=sequence,
        #     modified_sequence=result["modified_sequence"],
        #     target_label=target_label,
        #     success=result["success"],
        #     number_of_queries=result["num_queries"],
        #     queries_to_success=result["queries_to_success"],
        #     confidence_original=result["confidence_original"],
        #     confidence_modified=result["confidence_modified"],
        # )

        # # Add to artifacts
        # artifact = self.read_attack_artifact(attack_method, self.model.name_or_path)
        # artifact.add_attack(attack_info)

        # return attack_info

    def defense(self, defense_method: str, **kwargs) -> DefenseInfo:
        """
        Apply defense methods to DNA sequences

        Args:
            sequence: Input DNA sequence
            defense_method: One of ['adfar', 'freelb']
            **kwargs: Additional arguments for specific defense methods

        Returns:
            DefenseInfo object containing defense results
        """
        defense_methods = {"adfar": self._adfar_defense, "freelb": self._freelb_defense, "at": self._at_defense}

        if defense_method not in defense_methods:
            raise ValueError(
                f"Defense method must be one of {list(defense_methods.keys())}"
            )

        defense_methods[defense_method](**kwargs)

        # # Create defense info
        # defense_info = DefenseInfo(
        #     index=len(
        #         self._defense_artifacts.get(
        #             f"{defense_method}_{self.model.name_or_path}", []
        #         )
        #     ),
        #     method=defense_method,
        #     original_sequence=sequence,
        #     protected_sequence=result["protected_sequence"],
        #     protection_score=result["protection_score"],
        #     computational_cost=result["computational_cost"],
        #     robustness_score=result["robustness_score"],
        # )

        # # Add to artifacts
        # artifact = self.read_defense_artifact(defense_method, self.model.name_or_path)
        # artifact.add_defense(defense_info)

        # return defense_info

    # Attack method implementations
    def _bert_attack(
        self, **kwargs
    ) -> Dict[str, Any]:
        """BERT-Attack implementation"""
        run_bertattack_attack_script(
            data_path=kwargs["data_path"],
            mlm_path=kwargs["mlm_path"],
            tgt_path=kwargs["tgt_path"],
            model=kwargs["model"],
            output_dir=kwargs["output_dir"],
            num_label=kwargs["num_label"],
            k=kwargs["k"],
            threshold_pred_score=kwargs["threshold_pred_score"],
            start=kwargs["start"],
            end=kwargs["end"],
            use_bpe=kwargs["use_bpe"]
        )

    def _textfooler_attack(
        self, **kwargs
    ) -> Dict[str, Any]:
        """TextFooler attack implementation"""
        # Call the subprocess runner for TextFooler
        run_textfooler_attack_script(
            base_dir=kwargs["base_dir"],
            tasks=kwargs["tasks"],
            target_model=kwargs["model"],
            target_model_path=kwargs["target_model_path"],
            num_label=kwargs["num_label"],
            output_dir_base=kwargs["output_dir_base"],
            attack_script_path=kwargs.get("attack_script_path", "TextFooler/attack_classification_general.py")
        )
       
    def _pgd_attack(self, **kwargs) -> Dict[str, Any]:
        """PGD attack implementation"""
        run_pgd_attack_script(
            tasks=kwargs["tasks"],
            model=kwargs["model"],
            data_base_dir=kwargs["data_base_dir"],
            model_base_dir=kwargs["model_base_dir"],
            output_base_dir=kwargs["output_base_dir"],
            tokenizer_base_dir=kwargs["tokenizer_base_dir"],
            model_type=kwargs["model_type"],
            test_script_path=kwargs.get("test_script_path", "PGD/test.py"),
            task_name=kwargs.get("task_name", "0"),
            num_label=kwargs.get("num_label", "2"),
            n_gpu=kwargs.get("n_gpu", "1"),
            max_seq_length=kwargs.get("max_seq_length", "256"),
            batch_size=kwargs.get("batch_size", "16"),
        )
        

    def _fimba_attack(self, **kwargs) -> Dict[str, Any]:
        """
        FIMBA attack implementation (dynamic, like _pgd_attack).
        Runs shap_dl_analysis2.py and runatk_standalone.py for each task in tasks.
        kwargs should include:
            - tasks: list of task names
            - model: model name (e.g., 'nt1', 'nt2', 'og')
            - script_dir: directory containing FIMBA scripts (optional, default: 'fimba-attack' relative to this file)
            - max_seq_length, batch_size, etc. (optional, with sensible defaults)
        """


        tasks = kwargs["tasks"]
        model = kwargs["model"]
        script_dir = kwargs.get("script_dir", os.path.join(os.path.dirname(__file__), "fimba-attack"))
        max_seq_length = str(kwargs.get("max_seq_length", 128))
        batch_size_shap = str(kwargs.get("batch_size_shap", 1))
        batch_size_atk = str(kwargs.get("batch_size_atk", 64))
        num_label = str(kwargs.get("num_label", 2))
        target_model_path = kwargs.get("target_model_path", "")

        for task in tasks:
            print(f"Running FIMBA attack for model: {model}, task: {task}")

            shap_output_file = os.path.join(script_dir, "shap_dicts", f"shap_{model}_fimba_{task}.pkl")
            data_dir = f"./GUE/{task}"
            model_name_or_path = target_model_path
            output_dir = f"./fimba-attack/results/{model}"

            # 1. Run shap_dl_analysis2.py
            subprocess.run([
                "python", os.path.join(script_dir, "shap_dl_analysis2.py"),
                "--data_dir", data_dir,
                "--model_name_or_path", model_name_or_path,
                "--task_name", task,
                "--num_label", num_label,
                "--max_seq_length", max_seq_length,
                "--batch_size", batch_size_shap,
                "--dataset_name", task,
                "--model_type", model,
                "--overwrite_cache",
                "--shap_output_file", shap_output_file
            ], check=True)

            # 2. Run runatk_standalone.py
            subprocess.run([
                "python", os.path.join(script_dir, "runatk_standalone.py"),
                "--data_dir", data_dir,
                "--model_name_or_path", model_name_or_path,
                "--task_name", task,
                "--num_label", num_label,
                "--max_seq_length", max_seq_length,
                "--shap_file", shap_output_file,
                "--increase_fn",
                "--batch_size", batch_size_atk,
                "--model_type", model,
                "--output_dir", output_dir,
                "--overwrite_cache",
                "--overwrite_output_dir"
            ], check=True)


    # Defense method implementations
    def _adfar_defense(self, **kwargs) -> Dict[str, Any]:
        """ADFAR defense implementation: runs the adversarial training pipeline as in run_nt1.py."""

        # Extract parameters from kwargs or set defaults
        tasks = kwargs.get('tasks')
        if not tasks:
            raise ValueError("ADFAR defense requires a 'tasks' argument in kwargs.")
        
        base_dir = kwargs.get('base_dir', './GUE')
        model_type = kwargs.get('model_type', 'bert')
        textfooler_dir = kwargs.get('textfooler_dir', './TextFooler')
        adv_results_dir = kwargs.get('adv_results_dir', 'adv_results')
        adfar_src_dir = kwargs.get('adfar_src_dir', './ADFAR/src')
        experiments_dir = kwargs.get('experiments_dir', f'{adfar_src_dir}/experiments/GUE')
        datasize = kwargs.get('datasize', 9662)
        num_classes = kwargs.get('num_classes', 2)
        batch_size = kwargs.get('batch_size', 32)
        max_seq_length = kwargs.get('max_seq_length', 256)
        train_batch_size = kwargs.get('train_batch_size', 2)
        eval_batch_size = kwargs.get('eval_batch_size', 2)
        num_train_epochs = kwargs.get('num_train_epochs', 5)
        learning_rate = kwargs.get('learning_rate', 3e-5)
        attack_methods = kwargs.get('attack_methods', ['bertattack', 'textfooler', 'pgd'])
        output_base_dir = kwargs.get('output_base_dir', 'test/PGD')
        output_dir = kwargs.get('output_dir', 'test/ADFAR')

        results = {}
        for task in tasks:
            dataset_path = os.path.join(base_dir, task, 'cat.csv')
            target_model_path = kwargs["target_model_path"]
            tokenizer_base_dir = kwargs["tokenizer_path"]

            # 1.2 Use TextFooler as the attack method to produce adversarial examples:
            command3 = [
                'python', adfar_src_dir + '/attack_classification_simplified.py',
                '--dataset_path', dataset_path,
                '--target_model', model_type,
                '--target_model_path', target_model_path,
                '--max_seq_length', str(max_seq_length),
                '--batch_size', str(batch_size),
                '--counter_fitting_embeddings_path', f'{textfooler_dir}/embeddings/subword_{model_type}_embeddings.txt',
                '--counter_fitting_cos_sim_path', f'{textfooler_dir}/cos_sim_counter_fitting/cos_sim_counter_fitting_{model_type}.npy',
                '--USE_cache_path', f'{textfooler_dir}/tf_cache',
                '--nclasses', str(num_classes),
                '--output_dir', f'{adv_results_dir}/{model_type}/{task}'
            ]

            command4 = [
                'python', adfar_src_dir + '/get_pure_adversaries.py',
                '--adversaries_path', f'{adv_results_dir}/{model_type}/{task}/adversaries.txt',
                '--output_path', f'{base_dir}/{task}/{model_type}/attacked_data',
                '--times', '1',
                '--change', '0',
                '--txtortsv', 'tsv',
                '--datasize', str(datasize)
            ]

            command5 = [
                'python', adfar_src_dir + '/combine_data.py',
                '--add_file', f'{base_dir}/{task}/{model_type}/attacked_data/pure_adversaries.tsv',
                '--change_label', '2',
                '--original_dataset', f'{base_dir}/{task}',
                '--output_path', f'{base_dir}/{task}/{model_type}/combined_data/2times_adv_0-3/',
                '--isMR', '0'
            ]

            command6 = [
                'python', adfar_src_dir + '/run_simplification.py',
                '--complex_threshold', '3000',
                '--ratio', '0.25',
                '--syn_num', '20',
                '--most_freq_num', '10',
                '--simplify_version', 'random_freq_v1',
                '--cos_sim_file', f'{textfooler_dir}/cos_sim_counter_fitting/cos_sim_counter_fitting_{task}.npy',
                '--counterfitted_vectors', f'{textfooler_dir}/embeddings/subword_{task}_embeddings.txt',
                '--file_to_simplify', f'{base_dir}/{task}/{model_type}/combined_data/2times_adv_0-3/train.tsv',
                '--output_path', f'{base_dir}/{task}/{model_type}/simplified_data/2times_adv_0-3/',
                '--freq_file', f'{base_dir}/{task}/subword_frequencies.json'
            ]

            command7 = [
                'python', adfar_src_dir + '/combine_data.py',
                '--add_file', f'{base_dir}/{task}/{model_type}/simplified_data/2times_adv_0-3/train.tsv',
                '--change_label', '4',
                '--original_dataset', f'{base_dir}/{task}/{model_type}/combined_data/2times_adv_0-3/',
                '--output_path', f'{base_dir}/{task}/{model_type}/combined_data/4times_adv_0-7/',
                '--isMR', '0'
            ]

            command8 = [
                'python', adfar_src_dir + '/run_classification_adv.py',
                '--task_name', task,
                '--max_seq_len', '128',
                '--do_train',
                '--do_eval',
                '--attention', '2',
                '--data_dir', f'{base_dir}/{task}/{model_type}/combined_data/4times_adv_0-7/',
                '--output_dir', f'{experiments_dir}/{task}/{model_type}/4times_adv_double_0-7',
                '--model_name_or_path', target_model_path,
                '--per_device_train_batch_size', str(train_batch_size),
                '--per_device_eval_batch_size', str(eval_batch_size),
                '--save_total_limit', '2',
                '--learning_rate', str(learning_rate),
                '--num_train_epochs', str(num_train_epochs),
                '--svd_reserve_size', '0',
                '--evaluation_strategy', 'epoch',
                '--overwrite_output_dir',
                '--model_type', task,
                '--overwrite_cache'
            ]

            # Run all commands in order, in the ADFAR/src directory
            for idx, cmd in enumerate([command3, command4, command5, command6, command7, command8], start=3):
                print(f"[ADFAR] Running command{idx}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            
            # After training, run only the specified attacks for this task/model
            if "bertattack" in attack_methods:
                print(f"Running bertattack on FreeLB-trained model for task: {task}")
                run_bertattack_attack_script(
                    data_path=kwargs["data_path"],
                    mlm_path=f'{experiments_dir}/{task}/{model_type}/4times_adv_double_0-7',
                    tgt_path=kwargs["tgt_path"],
                    model=kwargs["attack_model_type"],
                    output_dir=kwargs["output_dir"],
                    num_label=kwargs["num_label"],
                    k=kwargs["k"],
                    threshold_pred_score=kwargs["threshold_pred_score"],
                    start=kwargs["start"],
                    end=kwargs["end"],
                    use_bpe=kwargs["use_bpe"]
                )
            if "textfooler" in attack_methods:
                print(f"Running textfooler on FreeLB-trained model for task: {task}")
                run_textfooler_attack_script(
                    base_dir=kwargs["base_dir"],
                    tasks=[task],
                    target_model=kwargs["attack_model_type"],
                    target_model_path=f'{experiments_dir}/{task}/{model_type}/4times_adv_double_0-7',
                    num_label=kwargs["num_label"],
                    output_dir_base=kwargs["output_dir_base"],
                    attack_script_path=kwargs.get("attack_script_path", "TextFooler/attack_classification_general.py")
                )
            if "pgd" in attack_methods:
                print(f"Running pgd on FreeLB-trained model for task: {task}")
                model_type = None
                    
                if any(key in model_type for key in ['dnabert', 'dnabert-2']):
                    model_type = 'bert'
                elif any(key in model_type for key in ['hyenadna']):
                    model_type = 'hyena'
                elif any(key in model_type for key in ['genomeocean', 'og']):
                    model_type = 'og'
                elif any(key in model_type for key in ['nt1']):
                    model_type = 'nt1'
                elif any(key in model_type for key in ['nt2']):
                    model_type = 'nt2'

                run_pgd_attack_script(
                    tasks=[task],
                    model=kwargs["model"],
                    data_base_dir=kwargs["base_dir"],
                    model_base_dir=f'{experiments_dir}/{task}/{model_type}/4times_adv_double_0-7',
                    output_base_dir=os.path.join(output_base_dir, "PGD"),
                    tokenizer_base_dir=tokenizer_base_dir,
                    model_type=model_type,
                    test_script_path=kwargs.get("test_script_path", "./PGD/pgd.py"),
                    task_name=kwargs.get("task_name", "0"),
                    num_label=kwargs.get("num_label", "2"),
                    n_gpu=kwargs.get("n_gpu", "1"),
                    max_seq_length=kwargs.get("max_seq_length", "256"),
                    batch_size=kwargs.get("batch_size", "16"),
                )




    def _freelb_defense(self, **kwargs) -> Dict[str, Any]:
        """
        FreeLB defense implementation:
        - Runs FreeLB adversarial training for each task
        - After training, runs the specified attacks (bertattack, textfooler, pgd) for each task/model, as given in attack_methods
        - Returns a dummy result for now
        kwargs should include:
            - tasks: list of task names
            - model: model name
            - attack_methods: list of attack names to run (e.g., ['bertattack', 'pgd'])
            - project_root: path to FreeLB project root (optional)
            - log_dir, ckpt_dir: optional, will be set relative to project_root if not provided
            - Other parameters as needed
        """


        tasks = kwargs["tasks"]
        model = kwargs["model"]
        attack_model_type = kwargs["attack_model_type"]
        attack_methods = kwargs.get("attack_methods", ["bertattack", "textfooler", "pgd"])
        project_root = kwargs.get("project_root", "./FreeLB/huggingface-transformers")
        output_dir_base = kwargs.get("output_dir_base", "./test/FreeLB")
        log_dir = kwargs.get("log_dir", os.path.join(output_dir_base, "logs"))
        ckpt_dir = kwargs.get("ckpt_dir", os.path.join(output_dir_base, "checkpoints"))
        gue_dir = kwargs.get("gue_dir", "./GUE")
        script_glue_freelb2 = os.path.join(project_root, "examples", "run_glue_freelb2.py")
        script_glue_freelb3 = os.path.join(project_root, "examples", "run_glue_freelb3.py")

        # FreeLB hyperparameters (can be overridden via kwargs)
        alr = str(kwargs.get("adv_lr", 1e-1))
        amag = str(kwargs.get("adv_init_mag", 6e-1))
        anorm = str(kwargs.get("adv_max_norm", 0))
        asteps = str(kwargs.get("adv_steps", 2))
        lr = str(kwargs.get("lr", 1e-5))
        bsize = str(kwargs.get("batch_size", 32))
        gas = str(kwargs.get("gradient_accumulation_steps", 1))
        seqlen = str(kwargs.get("max_seq_length", 256))
        hdp = str(kwargs.get("hidden_dropout_prob", 0.1))
        adp = str(kwargs.get("attention_probs_dropout_prob", 0))
        ts = str(kwargs.get("max_steps", 2000))
        ws = str(kwargs.get("warmup_steps", 100))
        seed = str(kwargs.get("seed", 42))
        wd = str(kwargs.get("weight_decay", 1e-2))
        num_label = str(kwargs.get("num_label", 2))
        gpu = str(kwargs.get("gpu", 0))

        for task in tasks:
            mname = kwargs["target_model_path"]
            model_type = model
            expname = f"{model_type}_{task}"
            if model_type == "hyena":
                script = "./FreeLB/huggingface-transformers/examples/run_glue_freelb3.py"
            else:
                script = "./FreeLB/huggingface-transformers/examples/run_glue_freelb2.py"
            output_dir = os.path.join(ckpt_dir, expname)
            data_dir = os.path.join(gue_dir, task)
            log_file = os.path.join(log_dir, f"{expname}.log")
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            # Run FreeLB adversarial training
            command = [
                "python", script,
                "--model_type", model_type,
                "--model_name_or_path", mname,
                "--task_name", task,
                "--do_train",
                "--do_eval",
                "--do_lower_case",
                "--data_dir", data_dir,
                "--max_seq_length", seqlen,
                "--per_gpu_train_batch_size", bsize,
                "--gradient_accumulation_steps", gas,
                "--learning_rate", lr,
                "--weight_decay", wd,
                "--gpu", gpu,
                "--output_dir", output_dir,
                "--hidden_dropout_prob", hdp,
                "--attention_probs_dropout_prob", adp,
                "--adv-lr", alr,
                "--adv-init-mag", amag,
                "--adv-max-norm", anorm,
                "--adv-steps", asteps,
                "--expname", expname,
                "--evaluate_during_training",
                "--max_steps", ts,
                "--warmup_steps", ws,
                "--seed", seed,
                "--logging_steps", "100",
                "--save_steps", "100",
                "--num_label", num_label,
                "--overwrite_output_dir",
                "--overwrite_cache"
            ]
            print(f"Running FreeLB for task: {task}")
            with open(log_file, "w+") as lf:
                try:
                    subprocess.run(command, check=True, stdout=lf, stderr=lf)
                except subprocess.CalledProcessError as e:
                    lf.seek(0)
                    print("=== Error log ===")
                    print(lf.read())
                    raise
            print(f"---\n{task} FreeLB finish\n---")

            # After training, run only the specified attacks for this task/model
            if "bertattack" in attack_methods:
                print(f"Running bertattack on FreeLB-trained model for task: {task}")
                run_bertattack_attack_script(
                    data_path=kwargs["data_path"],
                    mlm_path=output_dir,
                    tgt_path=kwargs["tgt_path"],
                    model=kwargs["attack_model_type"],
                    output_dir=output_dir,
                    num_label=kwargs["num_label"],
                    k=kwargs["k"],
                    threshold_pred_score=kwargs["threshold_pred_score"],
                    start=kwargs["start"],
                    end=kwargs["end"],
                    use_bpe=kwargs["use_bpe"]
                )
            if "textfooler" in attack_methods:
                print(f"Running textfooler on FreeLB-trained model for task: {task}")
                run_textfooler_attack_script(
                    base_dir=kwargs["base_dir"],
                    tasks=[task],
                    target_model=kwargs["attack_model_type"],
                    target_model_path=output_dir,
                    num_label=kwargs["num_label"],
                    output_dir_base=kwargs["output_dir_base"],
                    attack_script_path=kwargs.get("attack_script_path", "TextFooler/attack_classification_general.py")
                )
            if "pgd" in attack_methods:
                print(f"Running pgd on FreeLB-trained model for task: {task}")
                model_type = None
                    
                if any(key in model for key in ['dnabert', 'dnabert-2']):
                    model_type = 'bert'
                elif any(key in model for key in ['hyenadna']):
                    model_type = 'hyena'
                elif any(key in model for key in ['genomeocean', 'og']):
                    model_type = 'og'
                elif any(key in model for key in ['nt1']):
                    model_type = 'nt1'
                elif any(key in model for key in ['nt2']):
                    model_type = 'nt2'

                run_pgd_attack_script(
                    tasks=[task],
                    model=kwargs["model"],
                    data_base_dir=kwargs["base_dir"],
                    model_base_dir=output_dir,
                    output_base_dir=os.path.join(output_dir_base, "PGD"),
                    tokenizer_base_dir=kwargs["tokenizer_path"],
                    model_type=kwargs.get("attack_model_type", "dnabert"),
                    test_script_path=kwargs.get("test_script_path", "PGD/test.py"),
                    task_name=kwargs.get("task_name", "0"),
                    num_label=kwargs.get("num_label", "2"),
                    n_gpu=kwargs.get("n_gpu", "1"),
                    max_seq_length=kwargs.get("max_seq_length", "256"),
                    batch_size=kwargs.get("batch_size", "16"),
                )

    def _at_defense(self, **kwargs) -> Dict[str, Any]:
        """
        AT defense implementation:
        - Runs AT adversarial training for each task
        - After training, runs the specified attacks (bertattack, textfooler, pgd) for each task/model, as given in attack_methods
        - Returns a dummy result for now

        kwargs should include:
            - tasks: list of task names
            - model: model name or huggingface path
            - attack_methods: list of attack names to run (e.g., ['bertattack', 'pgd'])
            - data_dir_base: base path to the GUE dataset
            - output_dir_base: base path to save trained models
            - other training parameters: learning_rate, batch_size, epochs, etc.
        """

        tasks = kwargs["tasks"]
        attack_methods = kwargs["attack_methods"]
        model = kwargs.get("model", "zhihan1996/DNABERT-2-117M")
        data_dir_base = kwargs.get("data_dir_base", "./GUE")
        output_dir_base = kwargs.get("output_dir_base", "./test/AT")

        # Hyperparameters
        kmer = str(kwargs.get("kmer", -1))
        model_max_length = str(kwargs.get("model_max_length", 256))
        train_batch_size = str(kwargs.get("train_batch_size", 64))
        eval_batch_size = str(kwargs.get("eval_batch_size", 64))
        grad_accum = str(kwargs.get("gradient_accumulation_steps", 1))
        learning_rate = str(kwargs.get("learning_rate", 3e-5))
        epochs = str(kwargs.get("num_train_epochs", 4))
        save_steps = str(kwargs.get("save_steps", 200))
        eval_steps = str(kwargs.get("eval_steps", 200))
        warmup_ratio = str(kwargs.get("warmup_ratio", 0.05))
        logging_steps = str(kwargs.get("logging_steps", 100))
        seed = str(kwargs.get("seed", 42))

        # Set environment variables
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["DATA_CACHE_DIR"] = ".hf_data"
        os.environ["MODEL_CACHE_DIR"] = ".hf_cache"

        # Change to project directory
        os.makedirs(output_dir_base, exist_ok=True)

        for task in tasks:
            print(f"Running AT training for task: {task}")

            data_path = os.path.join(data_dir_base, task)
            output_dir = os.path.join(output_dir_base, task)
            run_name = f"AT_{model}_{task}"

            command = [
                "python", "./AT/train.py",
                "--model_name_or_path", model,
                "--data_path", data_path,
                "--kmer", kmer,
                "--run_name", run_name,
                "--model_max_length", model_max_length,
                "--per_device_train_batch_size", train_batch_size,
                "--per_device_eval_batch_size", eval_batch_size,
                "--gradient_accumulation_steps", grad_accum,
                "--learning_rate", learning_rate,
                "--num_train_epochs", epochs,
                "--fp16",
                "--save_strategy", "steps",
                "--save_steps", save_steps,
                "--output_dir", output_dir,
                "--evaluation_strategy", "steps",
                "--eval_steps", eval_steps,
                "--warmup_ratio", warmup_ratio,
                "--logging_steps", logging_steps,
                "--overwrite_output_dir", "True",
                "--log_level", "info",
                "--find_unused_parameters", "False",
                "--save_model", "True"
            ]

            # Execute and log output
            log_file = os.path.join(output_dir, "train.log")
            os.makedirs(output_dir, exist_ok=True)
            with open(log_file, "w") as lf:
                subprocess.run(command, check=True, stdout=lf, stderr=lf)

            print(f"---\n{task} AT training finished\n---")

            # After training, run only the specified attacks for this task/model
            if "bertattack" in attack_methods:
                print(f"Running bertattack on FreeLB-trained model for task: {task}")
                run_bertattack_attack_script(
                    data_path=kwargs["data_path"],
                    mlm_path=kwargs["mlm_path"],
                    tgt_path=kwargs["tgt_path"],
                    model=kwargs["attack_model_type"],
                    output_dir=kwargs["output_dir"],
                    num_label=kwargs["num_label"],
                    k=kwargs["k"],
                    threshold_pred_score=kwargs["threshold_pred_score"],
                    start=kwargs["start"],
                    end=kwargs["end"],
                    use_bpe=kwargs["use_bpe"]
                )
            if "textfooler" in attack_methods:
                print(f"Running textfooler on FreeLB-trained model for task: {task}")
                run_textfooler_attack_script(
                    base_dir=kwargs["gue_dir"],
                    tasks=kwargs["tasks"],
                    target_model=kwargs["attack_model_type"],
                    target_model_path=kwargs["target_model_path"],
                    num_label=kwargs["num_label"],
                    output_dir_base=kwargs["output_dir_base"],
                    attack_script_path=kwargs.get("attack_script_path", "TextFooler/attack_classification_general.py")
                )
            if "pgd" in attack_methods:
                print(f"Running pgd on FreeLB-trained model for task: {task}")
                model_type = None
                    
                if any(key in model for key in ['dnabert', 'dnabert-2']):
                    model_type = 'bert'
                elif any(key in model for key in ['hyenadna']):
                    model_type = 'hyena'
                elif any(key in model for key in ['genomeocean', 'og']):
                    model_type = 'og'
                elif any(key in model for key in ['nt1']):
                    model_type = 'nt1'
                elif any(key in model for key in ['nt2']):
                    model_type = 'nt2'

                run_pgd_attack_script(
                    tasks=kwargs["tasks"],
                    model=kwargs["model"],
                    data_base_dir=kwargs["gue_dir"],
                    model_base_dir=output_dir,
                    output_base_dir=os.path.join(output_dir_base, "PGD"),
                    tokenizer_base_dir=output_dir,
                    model_type=model_type,
                    test_script_path=kwargs.get("test_script_path", "PGD/test.py"),
                    task_name=kwargs.get("task_name", "0"),
                    num_label=kwargs.get("num_label", "2"),
                    n_gpu=kwargs.get("n_gpu", "1"),
                    max_seq_length=kwargs.get("max_seq_length", "256"),
                    batch_size=kwargs.get("batch_size", "16"),
                )

def run_bertattack_attack_script(
    data_path,
    mlm_path,
    tgt_path,
    model,
    output_dir,
    num_label,
    k,
    threshold_pred_score,
    start,
    end,
    use_bpe=0
): 
    model_type = None

    if any(key in model for key in ['dnabert', 'dnabert-2']):
        model_type = 'dnabert'
    elif any(key in model for key in ['hyenadna']):
        model_type = 'hyena'
    elif any(key in model for key in ['genomeocean', 'og']):
        model_type = 'og'
    elif any(key in model for key in ['nt1', 'nt2']):
        model_type = 'nt'

    if model_type is None:
        raise ValueError(f"Unsupported model type: {model}")

    print(f"Model Type Detected: {model_type}")

    script_name = {
        'dnabert': 'BERT-Attack/bertattack.py',
        'hyena': 'BERT-Attack/hyenaattack.py',
        'og': 'BERT-Attack/ogattack.py',
        'nt': 'BERT-Attack/ntattack.py'
    }[model_type]

    command = [
        'python', script_name,
        '--data_path', data_path,
        '--mlm_path', mlm_path,
        '--tgt_path', tgt_path,
        '--output_dir', output_dir,
        '--num_label', str(num_label),
        '--k', str(k),
        '--threshold_pred_score', str(threshold_pred_score),
        '--start', str(start),
        '--end', str(end),
        '--use_bpe', str(use_bpe),
    ]

    print(f"Running command: {' '.join(command)}")

    subprocess.run(command, check=True)



def run_textfooler_attack_script(
    base_dir,
    tasks,
    target_model_path,
    output_dir_base,
    target_model='bert',
    max_seq_length=256,
    batch_size=128,
    num_label=2,
    attack_script_path="TextFooler/attack_classification_general.py",
    
):
    
    for dataset_dir in tasks:
        dataset_path = os.path.join(base_dir, dataset_dir, 'five_percent/cat.csv')
        output_dir = os.path.join(output_dir_base, target_model, dataset_dir)
        counter_fitting_embeddings_path=f"./TextFooler/embeddings/subword_{target_model}_embeddings.txt"
        counter_fitting_cos_sim_path=f"./TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting_{target_model}.npy"
        USE_cache_path="./TextFooler/tf_cache"
        tokenizer_path=f"/scratch/anonymous/Attack_Benchmark/models/{target_model}/{dataset_dir}/origin"
        
        command_template = (
            f'python {attack_script_path} --dataset_path {dataset_path} '
            '--target_model {target_model} '
            '--target_model_path {target_model_path} '
            '--output_dir {output_dir} '
            '--max_seq_length {max_seq_length} --batch_size {batch_size} '
            '--counter_fitting_embeddings_path {counter_fitting_embeddings_path} '
            '--counter_fitting_cos_sim_path {counter_fitting_cos_sim_path} '
            '--USE_cache_path {USE_cache_path} '
            '--nclasses {num_label} --tokenizer_path {tokenizer_path}'
        )
        
        if os.path.exists(dataset_path):
            print(f"Dataset file found: {dataset_path}")
            command = command_template.format(
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                target_model_path=target_model_path,
                target_model=target_model,
                output_dir=output_dir,
                counter_fitting_embeddings_path=counter_fitting_embeddings_path,
                counter_fitting_cos_sim_path=counter_fitting_cos_sim_path,
                USE_cache_path=USE_cache_path,
                tokenizer_path=tokenizer_path,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                num_label=num_label,
            )
            print(command)
            subprocess.run(command, shell=True, check=True)
        else:
            print(f"Dataset file not found: {dataset_path}")


def run_pgd_attack_script(
    tasks,
    model,
    data_base_dir,
    model_base_dir,
    output_base_dir,
    tokenizer_base_dir,
    model_type,
    test_script_path="PGD/test.py",
    task_name="0",
    num_label=2,
    n_gpu=1,
    max_seq_length=256,
    batch_size=16,
):
    for task in tasks:
        data_dir = os.path.join(data_base_dir, task)
        model_name_or_path = os.path.join(model_base_dir)
        output_dir = os.path.join(output_base_dir, model, task)
        tokenizer_name = os.path.join(tokenizer_base_dir)
        command = [
            "python", test_script_path,
            "--data_dir", data_dir,
            "--model_name_or_path", model_name_or_path,
            "--task_name", str(task_name),
            "--num_label", str(num_label),  # 转换为字符串
            "--n_gpu", str(n_gpu),           # 转换为字符串
            "--max_seq_length", str(max_seq_length),  # 转换为字符串
            "--batch_size", str(batch_size),  # 转换为字符串
            "--output_dir", output_dir,
            "--model_type", str(model_type),
            "--overwrite_cache",
            "--tokenizer_name", tokenizer_name
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        print(f"{task} finished")



def main():
    parser = argparse.ArgumentParser(
        description="GenoArmory: DNA Sequence Attack and Defense Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Visualization command
    vis_parser = subparsers.add_parser("visualize", help="Visualize frequency distribution from JSON files")
    vis_parser.add_argument("--folder_path", type=str, help="Path to the folder containing JSON files")
    vis_parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the PDF")

    # Read artifacts command
    read_parser = subparsers.add_parser("read", help="Read attack or defense artifacts")
    read_parser.add_argument("--type", choices=["attack", "defense"], required=True)
    read_parser.add_argument("--method", required=True, help="Method name")
    read_parser.add_argument("--model_name", required=True, help="Model name")
    read_parser.add_argument("--attack_method", type=str,)
    read_parser.add_argument(
        "--metadata", action="store_true", help="Show metadata only"
    )

    # Query command for batch attacks
    query_parser = subparsers.add_parser(
        "query", help="Query model with multiple sequences"
    )
    query_parser.add_argument(
        "--sequences", nargs="+", required=True, help="List of DNA sequences"
    )
    query_parser.add_argument(
        "--method", required=True, choices=["bertattack", "textfooler", "pgd", "fimba"]
    )
    query_parser.add_argument(
        "--target-labels", nargs="+", type=int, help="Target labels (optional)"
    )
    query_parser.add_argument(
        "--model-name", required=True, help="Name or path of the model"
    )
    query_parser.add_argument("--api-key", help="API key for the model (if needed)")
    query_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Perform an attack")
    attack_parser.add_argument(
        "--method", required=True, choices=["bertattack", "textfooler", "pgd", "fimba"]
    )
    attack_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )
    attack_parser.add_argument(
        "--params_file", type=str, help="Path to a JSON file containing parameters"
    )

    # Defense command
    defense_parser = subparsers.add_parser("defense", help="Apply a defense")
    defense_parser.add_argument("--method", required=True, choices=["adfar", "freelb", "at"])
    defense_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )
    defense_parser.add_argument(
        "--params_file", type=str, help="Path to a JSON file containing parameters"
    )

    # Defense query command
    defense_query_parser = subparsers.add_parser(
        "defense-query", help="Query model with defense"
    )
    defense_query_parser.add_argument(
        "--sequence", required=True, help="Input DNA sequence"
    )
    defense_query_parser.add_argument(
        "--model-name", required=True, help="Name or path of the model"
    )
    defense_query_parser.add_argument(
        "--defense-method", required=True, choices=["adfar", "freelb"]
    )
    defense_query_parser.add_argument(
        "--api-key", help="API key for the model (if needed)"
    )
    defense_query_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Model loading arguments
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Initialize GenoArmory
    armory = GenoArmory.from_pretrained(args.model_path, device=args.device)

    # Execute commands
    if args.command == "visualize":
        armory.visualization(
            folder_path=args.folder_path,
            save_path=args.save_path,
        )

    elif args.command == "attack":
        if args.params_file:
            try:
                with open(args.params_file, "r") as f:
                    kwargs = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in params file '{args.params_file}': {e}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Params file '{args.params_file}' not found.")
        elif args.params:
            try:
                kwargs = json.loads(args.params)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --params: {e}")
        else:
            kwargs = {}

        print("Loaded parameters:", kwargs)

        result = armory.attack(
            attack_method=args.method,
            **kwargs,
        )
        

    elif args.command == "defense":
        if args.params_file:
            try:
                with open(args.params_file, "r") as f:
                    kwargs = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in params file '{args.params_file}': {e}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Params file '{args.params_file}' not found.")
        elif args.params:
            try:
                kwargs = json.loads(args.params)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --params: {e}")
        else:
            kwargs = {}

        print("Loaded parameters:", kwargs)

        
        
        result = armory.defense(
            defense_method=args.method, **kwargs
        )
    elif args.command == "read":
        if args.type=='attack':
            result = armory.get_attack_metadata(
                method=args.method, model_name=args.model_name
            )
            print(result)
        elif args.type == "defense":
            result = armory.get_defense_metadata(
                method=args.method, attack_method=args.attack_method, model_name=args.model_name
            )
            print(result)
    else:
        raise RuntimeError("The commend is not support")


if __name__ == "__main__":
    main()


__all__ = ["GenoArmory"]