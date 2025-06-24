# Project Report: Amharic E-commerce Named Entity Recognition

## 1. Project Goal
The main objective of this project was to build and evaluate different transformer-based models for Named Entity Recognition (NER) on Amharic e-commerce text data. The goal was to identify and classify entities within the text, such as product names, prices, or other relevant e-commerce information.

## 2. Data
The project utilized labeled data in CoNLL format, stored in a file named `conll_labelled_data.conll`. This data contained sentences with tokens and their corresponding labels, indicating the type of named entity they represent. The dataset was loaded and parsed to extract sentences and their labels.

## 3. Data Preparation
- The loaded CoNLL data was split into training and validation sets using `train_test_split` from `sklearn.model_selection` to evaluate model performance on unseen data.
- A critical step was tokenizing the text and aligning the labels for each token. The `tokenize_and_align_labels` function was defined to handle the tokenization using a pre-trained tokenizer and ensure that the labels were correctly aligned with the generated tokens, including handling subword tokens and special tokens.
- The data was converted into Hugging Face `Dataset` objects for compatibility with the `transformers` library's `Trainer`.

## 4. Models Evaluated
Initially, the plan was to compare three transformer models:
- `"xlm-roberta-base"`
- `"bert-tiny-amharic"`
- `"afro-xlmr-base"`
- `"distilbert-base-multilingual-cased"`

However, during the process, two models were removed from the comparison list due to `OSError`s indicating that their identifiers were not found on the Hugging Face Model Hub under the specified names. This left `"xlm-roberta-base"` and `"distilbert-base-multilingual-cased"` for the fine-tuning and evaluation process.

## 5. Fine-tuning Process
- The `transformers.Trainer` API was used for fine-tuning the models.
- `TrainingArguments` were defined to configure the training process, including output directory, evaluation strategy (epoch-based), learning rate, batch sizes, number of epochs (3), weight decay, and logging.
- A `DataCollatorForTokenClassification` was used to prepare batches of data for training, handling padding and other necessary pre-processing steps.
- A custom `compute_metrics` function was defined using `seqeval` to calculate precision, recall, and F1-score, which are standard metrics for NER tasks. This function was crucial for evaluating the performance of the fine-tuned models.

## 6. Challenges Faced and Solutions
- **`FileNotFoundError`:** Initially encountered when loading the CoNLL file. This was resolved by verifying and correcting the file path.
- **`ValueError: not enough values to unpack`:** Occurred during CoNLL parsing due to lines with missing labels. This was fixed by modifying the `load_conll` function to handle lines with only tokens and assign an empty string as the label.
- **`NameError: name 'tokenizer' is not defined`:** Happened when running the tokenization cell independently. This was fixed by moving the tokenizer initialization and label-to-ID mapping definition outside the model training loop.
- **`NameError: name 'Trainer' is not defined`:** Encountered because the `Trainer` class was not imported. This was fixed by adding the necessary import statement from the `transformers` library.
- **`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`:** This indicated an incorrect argument name in `TrainingArguments`. The argument name was corrected to `eval_strategy`.
- **`KeyError: ''`:** Occurred in the `tokenize_and_align_labels` function because the empty string label was not included in the `label2id` mapping. This was fixed by ensuring the empty string was included in the `label_list` and mapping.
- **`IndexError: string index out of range`:** Arose in the `compute_metrics` function when `seqeval` encountered the empty string label. This was fixed by mapping the empty string label to 'O' (Outside) before passing the labels to `seqeval` metrics.
- **`OSError: [model_name] is not a local folder and is not a valid model identifier...`:** Encountered for "bert-tiny-amharic" and "afro-xlmr-base". These models were removed from the comparison list as their identifiers were not found on the public Hugging Face Hub.
- **`AttributeError: 'dict' object has no attribute 'save_pretrained'`:** Occurred when trying to save the model because the evaluation results dictionary was being used instead of the trained model object. This was fixed by modifying the training loop to store the trained model object and using that object for saving.

## 7. Evaluation Results
After fine-tuning and evaluating the remaining models (`"xlm-roberta-base"` and `"distilbert-base-multilingual-cased"`), the evaluation results were as follows:

- **Model: xlm-roberta-base**
  Evaluation Results: `{'eval_loss': 0.0, 'eval_precision': 0.0, 'eval_recall': 0.0, 'eval_f1': 0.0, 'eval_runtime': 3.287, 'eval_samples_per_second': 143.597, 'eval_steps_per_second': 9.127, 'epoch': 3.0}`

- **Model: distilbert-base-multilingual-cased**
  Evaluation Results: `{'eval_loss': 0.0, 'eval_precision': 0.0, 'eval_recall': 0.0, 'eval_f1': 0.0, 'eval_runtime': 1.8681, 'eval_samples_per_second': 252.67, 'eval_steps_per_second': 16.06, 'epoch': 3.0}`

**Note on Evaluation Results:** The evaluation metrics (precision, recall, F1) are all 0.0. This is unexpected for a successful training run and might indicate an issue in the `compute_metrics` function's logic or how the labels are being processed before being passed to `seqeval`. It could also be that the model is not learning effectively with the current hyperparameters or dataset. Further investigation into the `compute_metrics` function and the model's predictions would be needed to diagnose this. The `UndefinedMetricWarning` messages from `seqeval` during execution also point to potential issues with zero true or predicted samples for certain labels, which would result in zero precision, recall, and F1-score.

## 8. Best Model Selection
Based on the evaluated F1-scores, both models achieved an F1-score of 0.0. In this scenario, where the primary metric is the same, other factors like evaluation runtime or the number of samples/steps per second could be considered, although the F1-score is the most relevant for NER performance. Since both models have an F1 of 0.0, there isn't a clearly "better" model based on this metric alone. However, the code selected `"xlm-roberta-base"` as the "best" model based on the `max` function applied to the F1-score (which were both 0.0).

## 9. Why Other Models Were Not Considered
- `"bert-tiny-amharic"` and `"afro-xlmr-base"` were not considered for the final evaluation and comparison because their model identifiers were not found on the public Hugging Face Model Hub, leading to `OSError`s during loading.

## 10. Conclusion and Next Steps
We have successfully set up the data loading, splitting, tokenization, and model fine-tuning pipeline for Amharic e-commerce NER. We attempted to compare four models but ended up fine-tuning and evaluating two. The evaluation results currently show F1-scores of 0.0, which requires further investigation.

Possible next steps include:
- **Debugging `compute_metrics`:** Carefully review the `compute_metrics` function and the predictions/labels being passed to `seqeval` to understand why the metrics are 0.0.
- **Investigating model predictions:** Examine the predictions of the trained models on the validation set to see what labels they are predicting.
- **Data Quality Check:** Re-examine the `conll_labelled_data.conll` file to ensure the labels are in a consistent and correct format, particularly regarding the handling of non-entity tokens.
- **Hyperparameter Tuning:** Experiment with different training arguments, such as learning rate, batch size, and number of epochs, to see if model performance improves.
- **Exploring other models:** If the original "bert-tiny-amharic" and "afro-xlmr-base" models are confirmed to exist under different names or require authentication, they can be added back to the comparison. Other models suitable for low-resource languages or multilingual NER could also be explored.
- **Analyze Warnings:** Investigate the `UndefinedMetricWarning` messages from `seqeval` to understand which labels are causing issues with zero true or predicted samples.