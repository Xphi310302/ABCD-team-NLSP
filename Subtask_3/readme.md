# SoMeSci - Software Mentions in Science

SoMeSci is a project aimed at detecting and categorizing software mentions in scientific text data. The project utilizes natural language processing (NLP) techniques to extract and identify software-related entities from scientific documents.

## Overview

This repository contains scripts and resources for training and evaluating a model for software mention detection in scientific text. It includes:

- `prompt.py`: Contains functions for formatting input prompts and converting between text formats.
- `model.py`: Defines the model architecture and provides functions for loading the pre-trained model.
- `training.py`: Includes functions for preparing data, training the model, and performing inference.
- `data/`: Directory for storing the training and evaluation datasets.
- `models/`: Directory for storing pre-trained models.
- `predictions.txt`: Output file containing the model's predictions on the evaluation dataset.

## Usage

1. **Setup Environment**: Ensure you have the necessary dependencies installed. You can install them using `pip install -r requirements.txt`.

2. **Prepare Data**: Organize your training and evaluation datasets in CSV format. Each dataset should have columns for the text data and corresponding labels.

3. **Train Model**: Run the training script `training.py` to train the model on your training dataset. Adjust the configurations in the script as needed.

4. **Evaluate Model**: After training, the model will automatically evaluate its performance on the evaluation dataset. The predictions will be saved in `predictions.txt`.

5. **Inference**: You can perform inference on new data by using the model's `generate` method. See `training.py` for an example.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
