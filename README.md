In this study, the researchers will identify the connections between the research variables and the study outcomes.<|human|>Reproduction Study: COVID-Fact

In this research, the scholars will determine the relationships between research variables and the outcomes of the study.

This is the code and documentation of a reproduction study of the paper [COVID-Fact: Fact Extraction and Verification of Real-World Claims on COVID-19 Pandemic] (https://aclanthology.org/2021.acl-long.165) by Saakyan et al. (ACL 2021).

Claim of Original Paper: Two stage pipeline with Dense Passage Retrieval (DPR) and a RoBERTa-based Natural Language Inference (NLI) model is useful to establish the validity of COVID-19 assertions.

**Our Reproduction Focus: We managed to replicate the claim verification (Stage 2) in an oracle environment (based on gold evidence) and demonstrated a high level of performance, which confirmed the main approach in the methodology.

**Our Findings: We have trained a RoBERTa-base model on the COVID-Fact dataset and a high accuracy on the validation set confirming the validity of the initial strategy.


Sometimes, one requires a step-by-step guide to reproduction.

These steps are used to reproduce our results on a conventional machine (CPU or GPU).

### Prerequisites

*   **Python Version:** 3.8+
*  **Operating System: Windows, Linux or MacOS. The replication has been done through the Hugging Face ecosystem to be fully cross-platform compatible.
*   **Hardware: A computer with a minimum of 8GB of memory. Training will go much faster with a GPU, but will not be required.

### 1. Clone and initialise Repository.

```bash
# Clone the repository
git clone git: https://github.com/aidev0p6/Maamari-Manar_covid fact.git.
cd Maamari-Manar_covidfact

1. Develop and start a virtual environment (recommended)
python -m venv facts
Source facts/bin/activate  # (On Windows) venv/Scripts/activate.
2. Install Dependencies
Install any necessary Python packages. The primary libraries include transformers, datasets, evaluate and pandas.

<>
bash


pip install transformers datasets test pandas.
Or, in case you have a requirements.txt:
pip install requirements.txt
3. Acquire and Process the Dataset.
Obtaining the COVID-Fact dataset: The dataset should be taken out of the original COVID-Fact repository. You have to download the RTE-covidfact folder.
Install data: After downloading it, put the entire RTE-covidfact folder in the root of this project. The framework will resemble such:
<>
Plain Text


Maamari-Manar_covidfact/
├── RTE-covidfact/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── train.py
└── ...
The train.py script is set to take the data loaded in RTE-covidfact/split.tsv.
4. Conduct the Training and Evaluation.
Execute the main script. This will:

Preprocess and load (automatically deal with column name inconsistency).
Get the roberta-base model.
Train the model 3 epochs (configurable) on the data given.
Assess the validation (dev.tsv) set and report results.
Keep the best model and a result file.
<>
bash


python train.py
Expected Output & Behavior:

The script will print the columns of all the TSV files and verify any renaming (e.g., entailment to label).
Each 50 steps, logging output will appear with the training loss and learning rate.
The script will also test on the dev set (because the test set is not label-ed) and report the final validation accuracy and loss on the dev set.
The optimal model checkpoint will be stored in ./ final covidfact model/.
An overview of the results will be stored to the file ./final validation results.txt.
Training Time Note: The batch size of the script is 32. On a CPU, this will be slow. In a modern GPU, it will be much faster. In train.py, you can set the size of the per-device-train-batch size depending on your VRAM size.

5. Understanding the Results
The main measure that this script provides is the accuracy of validation. The outcome to have should be high, meaning that the RoBERTa model can successfully learn the task of veracity prediction using the COVID-Fact dataset, which will prove the statements of the original paper.

Detailed results are saved in the script with the name final validation results.txt.

 It has been noted that variations existed among the sources because of the scope and adaptations.
We reproduced in the oracle setting (gold evidence) in the stage of veracity prediction (NLI). This separates the transformer model performance of the retrieval component where one can be able to fairly validate the main claim.

Some of the modifications made in the original paper are:

Framework: Hugging Face transformers library should be used, rather than fairseq, because it is more compatible and maintainable.
Model Size: This setting uses roberta-base (125M parameters) as a more computationally efficient alternative to roberta-large (355M parameters). Its performance is also good, which proves the soundness of the methodology.
Data Handling: The code consists of an excellent preprocessing to deal with differences in the column names of the original dataset between splits.

 License and Citation
The project will be designed as academic research. Refer to the original paper:

<>
bibtex


@inproceedings{saakyan-2021-covid-fact,
    title = " {COVID} -Fact: Fact Extraction and Verification of Real-World Claims on {COVID} -19 Pandemic,
    author = "Saakyan, Arkadiy  and
      Chakrabarty, Tuhin  and
      Muresan, Smaranda",
    title of book = Proceedings of the 59 th Annual Meeting of the Association For Computational Linguistics and the 11 th International Joint Conference on Natural Language Processing (Volume 1: Long Papers),
    year = "2021",
    publisher = "Association for computational linguistics,
}
