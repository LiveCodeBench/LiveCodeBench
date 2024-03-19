# LiveCodeBench
Official repository for the paper "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"

<p align="center">
    <a href="https://livecodebench.github.io/">🏠 Home Page</a> •
    <a href="https://huggingface.co/datasets/livecodebench/">💻 Data </a> •
    <a href="https://livecodebench.github.io/leaderboard.html">🏆 Leaderboard</a> •
</p>

## Introduction
LiveCodeBench is a comprehensive and contamination-free evaluation of LLMs for code, which continuously collects new problems over time from contests across three competition platforms, namely LeetCode, AtCoder, and CodeForces.  
Distinctly, LiveCodeBench also focuses on a broader range of code-related capabilities, such as self-repair, code execution, and test output prediction, beyond just code generation. 
Currently, LiveCodeBench hosts four hundred high-quality coding problems that were published between May 2023 and February 2024.


## Installation
You can install the dependencies using pip:
```bash
pip install -U pebble datasets pyext # primary dependencies
pip install -U vllm google-generativeai anthropic openai mistralai # optional dependencies depending on models
```

Or alternatively you can use the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Data
We provide a benchmark for different code capability scenarios
- [Code Generation](https://huggingface.co/datasets/livecodebench/code_generation)
- [Code Execution](https://huggingface.co/datasets/livecodebench/execution)
- [Test Output Prediction](https://huggingface.co/datasets/livecodebench/test_generation)

## Inference and Evaluation

### Code Generation

We use `vllm` for inference using local models. We use  `tensor_parallel_size=${num_gpus}` for running the inference on multiple GPUs and can be configured using the `--tensor_parallel_size` flag.
For running the inference, please use the following command. 

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codegeneration
```

We compute `pass@1` and `pass@5` metrics for each model. 
We use a modified version of the checker released with the `apps` benchmark to compute the metrics. 
Particularly, we identified some unhandled edge cases in the original checker and fixed them and additionally simplified the checker based on our collected dataset.

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario codegeneration --evaluate
```

Note that time limits can cause a slight ( `< 0.3`) points of variation in the computation of the `pass@1` and `pass@5` metrics.
If you observe a significant variation in performance, adjust the `--num_process_evaluate` flag to a lower value or increase the `--timeout` flag. Please report particular issues caused by improper timeouts here. 

The Leaderboard uses [ExecEval](https://github.com/ntunlp/ExecEval) which is a more comprehensive evaluation script using docker for evaluation of the generated programs. We will release an alternative evaluation script soon. 

### Test Output Prediction
For running the test output prediction scenario you can simply run

```bash
python -m lcb_runner.runner.main --model {model_name} --scenario testoutputprediction --evaluate
```


## Adding Support for New Models

To add support for new models, we have implemented an extensible framework to add new models and customize prompts appropirately. 

Step 1: Add a new model to the [./lcb_runner/lm_styles.py](./lcb_runner/lm_styles.py) file. Particularly, extend the `LMStyle` class to add a new model family and extend the model to the `LanguageModelList` array.

Step 2: Since we use instruction tuned models, we allow configuring the instruction for each model. Modify the [./lcb_runner/prompts/generation.py](./lcb_runner/prompts/generation.py) file to add a new prompt for the model in the `format_prompt_generation` function. 
For example, the prompt for `DeepSeekCodeInstruct` family of models looks as follows

```python
if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
    prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n"
    prompt += f"{get_deepseekcode_question_template_answer(question)}"
    return prompt
```

## Results
LiveCodeBench can be used to evaluate performance of LLMs on different time-windows (using problem release date to filter the models). 
Thus we can detect and prevent potential contamination in the evaluation process and evaluate LLMs on _new_ problems.

<div style="text-align: center;">
    <img src="./assets/images/contamination1.png" alt="Code Generation Live Evaluation" class="teaser-image"
    width="40%" />
    <img src="./assets/images/contamination2.png" alt="Test Output Prediction Live Evaluation" class="teaser-image"
    width="40%" />
</div>

Next, we evaluate models on different code capabilities and find that relative performances of models do change over tasks (left). 
Thus, it highlights the need for holistic evaluation of LLMs for code.

<div style="text-align: center;">
    <img src="./assets/images/tasks_radar.png" alt="Holistic Tasks Evaluation" class="teaser-image"
    width="36.1%" />
    <img src="./assets/images/lcb_vs_he.png" alt="Comparing LCB vs HumanEval" class="teaser-image"
    width="46%" />
</div>

We also find evidence of possible overfitting on HumanEval (right). 
Particularly, models that perform well on HumanEval do not necessarily perform well on LiveCodeBench. 
In the scatterplot above, we find the models get clustered into two groups, shaded in red and green. 
The red group contains models that perform well on HumanEval but poorly on LiveCodeBench, while the green group contains models that perform well on both.

For more details, please refer to our paper at this [url](https://livecodebench.github.io/pdfs/paper.pdf).

## Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
