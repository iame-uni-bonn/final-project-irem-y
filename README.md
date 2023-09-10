# Fake News Classification

## Description

The Fake News Classification project consists of two models trained to recognize fake news using the LIAR dataset. This dataset contains 12.8K human-labeled short statements obtained from PolitiFact.com, with each statement rated for truthfulness by a PolitiFact.com editor. You can access the LIAR dataset [here](https://huggingface.co/datasets/liar).

Two models have been trained for this project:

1. **BERT Model**: One model is trained using BERT.
2. **Random Forest Model**: Another model is trained using the Random Forest algorithm.

Hyperparameters used for training have been optimized, and the training results can be found in the `docs/training_results` folder. After training, the models are compared with each other and a model from Hugging Face: [Jawaher/LIAR-fake-news-roberta-base](https://huggingface.co/Jawaher/LIAR-fake-news-roberta-base).

## Installation

To get started with the Fake News Classification project, follow these steps:

1. Clone the repository:
```console
$ git clone https://github.com/iame-uni-bonn/final-project-irem-y
```

2. Navigate to the project folder in your terminal.

3. Create a Conda environment using the provided `environment.yml` file:
```console
$ conda env create -f environment.yml
```

4. Activate the Conda environment:
```console
$ conda activate final-project-irem-y
```

## Usage

Disclaimer: Depending on your hardware, the runtime may vary as training the model is time-intensive and can take multiple hours.

To use the project, follow these steps:

1. Navigate to the project folder in your terminal.

2. Run the following command in the command line:
```console
$ python ./src/training_and_evaluate_functions/run_program.py
```

3. To adjust the training parameters, check the `src/training_and_evaluate_functions/run_program.py` file.

4. For general training examples, refer to the `src/training_and_evaluate_functions/training_examples.py` file. Here, you can customize and train the model according to your needs by adjusting the training parameters and commenting out the code that you don't need.

## Results and Evaluation

Due to the long training time, the hyperparameter optimisation of the models was difficult and the full potential couldn't be reached.
In comparison with the Hugginface model, both models achieved better results in all metrics (accuracy, precision, recall and F1 score).
A more stable environment could help to train the BERT model more efficiently.
