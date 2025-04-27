# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified, VILAGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # NOTE(Zhouenshen): Need to modify this function to use VILA
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # if os.getenv("DEBUG_MODE") == "true":
        # log_path = os.getenv("LOG_PATH")
        log_path = "./debug_log_8b.txt"
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        with open(log_path, "a") as f:
            f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
            f.write(f"Content: {content}\n")
            f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# NOTE(Zhouenshen): Neet to register the our own reward function.
reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward
}

# NOTE(Zhouenshen): load our own spatial dataset.
import json
from datasets import Dataset, DatasetDict

def load_spatial_datasets(dataset_configs):
    """
    Load and merge multiple JSON files, each containing conversation data along with image/depth references.
    
    Args:
        dataset_configs (List[dict]): A list of dataset configuration. Each dict should contain:
            - 'json_path'  : Path to the JSON file
            - 'image_root' : Path to the image directory
            - 'depth_root' : Path to the depth image directory
    
    Returns:
        DatasetDict: A Hugging Face DatasetDict object with a 'train' split that includes:
            - 'images': A list of lists of absolute image paths
            - 'depths': A list of lists of absolute depth image paths
            - 'problem': user question (human)
            - 'solution': GPT answer
    """
    
    # A dictionary to store combined results from all JSON files
    combined_data = {
        'image': [],
        'depth': [],
        'problem': [],
        'solution': []
    }
    
    # Iterate over every dataset config in the provided list
    for config in dataset_configs:
        json_path   = config.data_path
        image_root  = config.image_path
        depth_root  = config.depth_path
        
        # Read JSON file
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Traverse each entry in the JSON content
        for entry in data:
            # Map all images to absolute paths
            if 'image' in entry:
                abs_images = [os.path.join(image_root, img_name) for img_name in entry['image']]
            else:
                abs_images = []

            # Map all depth images to absolute paths
            if 'depth' in entry:
                abs_depths = [os.path.join(depth_root, depth_name) for depth_name in entry['depth']]
            else:
                abs_depths = []

            # Parse conversation turns in order to preserve multiple user->gpt pairs
            conversations = entry.get('conversations', [])
            
            i = 0
            while i < len(conversations):
                # Check if the current turn is from user
                if conversations[i]['from'] == 'human':
                    user_text = conversations[i]['value']
                    
                    # If the next turn is GPT's response, pair them up
                    if i + 1 < len(conversations) and conversations[i + 1]['from'] == 'gpt':
                        gpt_text = conversations[i + 1]['value']
                        
                        combined_data['image'].append(abs_images)
                        combined_data['depth'].append(abs_depths)
                        combined_data['problem'].append(user_text)
                        combined_data['solution'].append(gpt_text)
                        
                        # Move the index by 2 (skip the next turn processed)
                        i += 2
                    else:
                        # If there's no GPT turn following the user, just move to the next turn
                        i += 1
                else:
                    # If it's GPT or some other role that doesn't strictly pair, skip to next
                    i += 1


    assert (len(combined_data["depth"]) == len(combined_data["image"])) or (len(combined_data["depth"]) == 0), "The dataset should either have depth for all samples or none, but not mixed"

    # Convert merged data to a Hugging Face Dataset and then wrap in DatasetDict
    hf_dataset = Dataset.from_dict(combined_data)
    dataset_dict = DatasetDict({'train': hf_dataset})
    
    return dataset_dict


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # if '+' not in script_args.dataset_name:
    #     # Load the dataset
    #     dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # else:
    # NOTE(Zhouenshen): Support loading our own dataset (llava/NVILA format)
    # Register mixed datasets (this may include legacy dataset definitions)
    import llava.data.datasets_mixture as datasets_mixture
    datasets_mixture.register_datasets_mixtures()

    # Split the dataset name string on '+' to get individual dataset identifiers
    dataset_name_parts = script_args.dataset_name.split("+")

    # Build dataset configuration list using the dataset name parts and a legacy mapping dictionary
    dataset_configs = []
    for name in dataset_name_parts:
        if name not in datasets_mixture.DATASETS_LEGACY:
            raise ValueError(f"Dataset name '{name}' not found in DATASETS_LEGACY.")
        dataset_configs.append(datasets_mixture.DATASETS_LEGACY[name])

    # Load spatial dataset using the constructed configuration list
    dataset = load_spatial_datasets(dataset_configs)
    
    
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer (option) in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    # NOTE(Zhouenshen): This function needs to be modified to handle datasets containing either images alone or both images and depth information.
    def make_conversation_rgbd(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": QUESTION_TEMPLATE.format(Question=example["problem"]),
                },
            ],
        }

    if "image" and "depth" in dataset[script_args.dataset_train_split].features:
        print("has image and depth in dataset")
        dataset = dataset.map(make_conversation_rgbd)
    elif "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    # NOTE(Zhouenshen): Need to use VILAGRPOTrainer for training.
    trainer_cls = VILAGRPOTrainer
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
