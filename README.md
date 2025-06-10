# SSA: Sample Set Aggregator

Following are the codebase for the SSA project.
For Quick Start, please refer to the `QuickStart.ipynb` file.

You can also run a quick demo of the 0.5B SSA model for some initial steps on Colab (Not support full training due to VRAM limitation). [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user074/ssa/blob/main/QuickStart.ipynb)


# Environment

Please refer to the `environment.yml` file to create the environment.
```
conda env create -f environment.yml
conda activate SSA
```

Then please install prm800k package for grading purpose at the root directory of the project.
```
git clone https://github.com/openai/prm800k
```

After creation please manually install the `torchtune` package. There are many updates recently, and we changed the codebase to adapt to the grading and reward functions.

```
cd torchtune
pip install -e .
cd ..
```



## Data

Following files are used to generate the data for the SSA:

- `generateAnswers.py`: generate answers using a language model
- `createSFTData.py`: create SFT data


### Create answers for RL training and evaluation
Follow are the instructions in `generateAnswers.py` to generate answers for RL training. num_processes is the number of GPUs to use. model_name is the model path to use. dataset_name is the dataset to use. There are few options to choose from:
- `gsm8k`: gsm8k dataset
- `MATH`: MATH dataset
- `aime24`: aime24 dataset
- `aime25`: aime25 dataset
- `amc23`: amc23 dataset
- `olympiad`: olympiad dataset
- `mmlu-pro`: mmlu-pro dataset
- `arc`: ai2_arc dataset
- `truthfulqa`: truthfulqa dataset


For training data, we use split `train` and for testing data we use split `test`. total_num_samples is the number of samples to generate so they can be used for concatenation. Generated answers are saved in `/answers` folder.

Before you run the script, please make sure you relevant models. We suggest you to download the models to the `./model` folder.

To launch the script, please use the following command:
```
accelerate launch --num_processes=4 generateAnswers.py --model_name "model/Qwen2.5-7B-Instruct" --dataset_name "gsm8k" --split "train" --total_num_samples 20
```

You can use the same script to generate answers for evaluation. Please set the `split` to `test` and `total_num_samples` to the largest number of samples you want to put in the context.

### Create SFT data

To create SFT data, please use the following command:
```
python createSFTData.py --api_key "your_openai_api_key" --dataset_name "Your k concatenated dataset name of the gsm8k "
```

This script will create SFT data

# Training

## Construct training data
You can skip this step if you use the dataset we constructed which is `user074/concat_cleaned_gsm8k_math_5` on hugging face.

### Details of constructing training data
After you construct the answers from the answers, you can use the `constructTrainData.py` script to construct the training data. It concatenates the answers and creates a training dataset for the SSA. In addition it also filter out the none answers and the answers that exceed the max length (4000 tokens). You can set the different max length but in our experience more than 4000 tokens would cause the OOM error, and the results of more than 4000 tokens are not ususally beneficial to arrive at the correct answer.

To launch the script, please use the following command. `output_dataset` is the name of the dataset to push to the hub. `push_to_hub` is a boolean flag to push the dataset to the hub. You can also save the dataset to the local directory by setting `push_to_hub` to `False`. `plot_distribution` is a boolean flag to plot the distribution of the token length. `output_path` is the path to save the distribution plot. `gsm8k_path` is the path to the gsm8k answers. `math_path` is the path to the math answers.

```
python constructTrainData.py --gsm8k_path "answers/train/Qwen2.5-7B-Instruct_gsm8k_8" --math_path "answers/train/Qwen2.5-7B-Instruct_MATH_8" --output_dataset 'your/output/dataset/name' --push_to_hub --plot_distribution --output_path 'your/output/path'  
```
Please use the constructed training data to train the SSA. For anonymity requirements, please specify your own `gsm8k_dataset` and `math_dataset` to the dataset name on hugging face.


We provide an example of the training script for Qwen2.5-3B based model. For 0.5B and 1.5B based models, please change the relevant model name and parameters in the training script. Details can be found in the torchtune documentation.
The training script is `3B_rl_SSA_qwen.yaml` and `3B_sft_SSA_qwen.yaml`.


To launch the script, please use the following command:

## RL training
Please specify the model path and dataset path in the training script. Dataset path should be the path to the constructed training data above.
```
tune run --nproc_per_node 4 dev/grpo_full_finetune_distributed --config ./3B_rl_SSA_qwen.yaml
```

## SFT training
Please specify the model path and dataset path in the training script. Dataset path should be the path to the constructed SFT data from oracle responses.
```
tune run --nproc_per_node 4 full_finetune_distributed --config ./3B_sft_SSA_qwen.yaml
```

## RL training with SFT data
Use the checkpoints from SFT training as the initial model, then follow the RL training instructions above.


It will output models in the `./model/checkpoints` folder.


# Evaluation

Most of the commands have the same parameters to define the inference k for the parallel input. `num_answers` is the number of answers to concatenate or the number of files to evaluate for prm and majority vote cases.

To evaluate the model, please use the following command:
```
python evaluation.py --model_path "path/to/your/model" --answer_path "path/to/your/answers" --num_answers 5
```
model_path is the path to the model checkpoint. answer_path is the path to the generated answers which should be in the `./answers` folder. num_answers is the number of answers to concatenate. The evaluation results will be saved in the `./evaluation_results` folder. 

An example command for aime24 is:
```
python evaluation.py --model_path "model/checkpoints/rl_SSA_qwen3b" --answer_path "answers/Qwen2.5-7B-Instruct_aime24_20" --output_dir "evaluation_results/aime24"
```

For the USC results, please just change the model_path to the instruct model checkpoints, any checkpoint can work as long as there is `Instruct` in the model name path.

For the PRM results, please use the following command. It supports both ORM and PRM models evaluations.
```
python evaluation_prm.py --reward_model_path "PRM/ORM/model/path" --answer_path "path/to/your/answers" --num_files 20
```



In order to display the results, please use the following command:
```
python display_accuracy.py --results_dir 'path/to/your/evaluation_results/directory/right/above/json/file'
```
You can also check the generated answer's majority vote and pass@k accuracy
For anonymity requirements, please specify the dataset path and answer key.
```
python calculate_accuracies.py --answer_path "path/to/your/answers/directory/right/above/json/file" --number_of_files 20 --dataset_path "path/to/your/dataset" --answer_key "answer_key_name"
```

