# SOMD: Software Mention Detection in Scholarly Publications Task

This repository contains scripts for identifying and categorizing software mentions in text data. The project is the source code of ABCD team participating in NLSP workshop of [SOMD task](https://nfdi4ds.github.io/nslp2024/docs/somd_shared_task.html) using LLMs for NER task

## Files for all 3 subtasks:

1. **prompt.py**: Contains a function to format input sentences for processing.

2. **model.py**: Initializes the model for software mention identification and categorization. It also prints the number of trainable parameters in the model.

3. **training.py**: Includes functions for training the model on labeled data.

4. **utilis.py**: Contains utility functions used in preprocessing and evaluation.

5. **train_inference.py**: Main script that preprocesses data, trains the model, and makes predictions on test data.

## Requirements:

- Python 3.10
- torch
- transformers
- pandas
- tqdm
- datasets


## Usage:

1. Clone the repository:
To clone this repository, use the following command:

```bash
git clone https://github.com/Xphi310302/ABCD-team-NLSP.git
cd ABCD-team-NLSP
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Go to the folder of the subtask I, II or III
```bash
cd Subtask_1
```
3. Execute the desired script:
```bash
python train_inference.py
```


