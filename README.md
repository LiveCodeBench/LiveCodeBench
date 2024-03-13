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

<img src="./images/LCB_holistic_tasks.png" alt="Teaser" class="teaser-image center" width="80%" />


## Data
We provide a benchmark for different code capability scenarios
- [Code Generation](https://huggingface.co/datasets/livecodebench/code_generation)
- [Code Execution](https://huggingface.co/datasets/livecodebench/execution)
- [Test Output Prediction](https://huggingface.co/datasets/livecodebench/test_generation)

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
    <img src="./assets/images/tasks_radar.png" alt="Code Generation Live Evaluation" class="teaser-image"
    width="36%" width="40%"/>
    <img src="./assets/images/lcb_vs_he.png" alt="Test Output Prediction Live Evaluation" class="teaser-image"
    width="46%" width="40%" />
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