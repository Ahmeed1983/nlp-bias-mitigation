# NLP Bias Detection and Mitigation System

## Overview

This repository contains a Jupyter Notebook (`NLP_Bias_Detection_and_Mitigation_SystemREAL.ipynb`) that implements a robust pipeline for detecting, explaining, and mitigating toxicity and bias in Natural Language Processing (NLP) models.

By leveraging transformer models (specifically **RoBERTa**) and Explainable AI (XAI) techniques such as **LIME**, the system not only flags harmful content but also provides granular, interactive visualizations that show exactly *why* specific text was flagged. It also includes an automated moderation filter to ensure safe AI outputs.

## Major Features

* **Toxicity Detection:** Utilizes a fine-tuned RoBERTa sequence classification model to accurately score and classify text as "Toxic" or "Non-Toxic".
* **Explainable AI (XAI) with LIME:** Integrates Local Interpretable Model-Agnostic Explanations (LIME) to interpret model predictions. It generates interactive HTML/D3.js bar charts and highlighted text blocks to show the exact weight/contribution of individual words (e.g., highlighting which specific words triggered a toxicity warning).
* **Automated Mitigation & Filtering:** Acts as a safety layer for text generation. When a model generates text that crosses the toxicity threshold, the system intercepts and flags it, ensuring harmful outputs are caught before reaching the end user.
* **Evaluation & Metrics:** Includes built-in performance evaluation using the **Jigsaw Toxicity Prediction** dataset, calculating strict Accuracy, Precision, Recall, and F1-Scores.

## Technologies & Libraries Used

* **Python 3**
* **Hugging Face Transformers:** For tokenization and loading the pre-trained RoBERTa weights (`model.safetensors` / `pytorch_model.bin`).
* **LIME (Local Interpretable Model-Agnostic Explanations):** For generating word-weight importance arrays and interactive D3.js visualizations.
* **Datasets (Hugging Face):** Interacts directly with `jigsaw_toxicity_pred` to benchmark bias detection capabilities.

## Usage

1. **Environment Setup:** Open the notebook in Google Colab (recommended, as it is optimized for A100 GPUs) or a local Jupyter environment. Ensure you have the required libraries installed:
```bash
pip install torch transformers lime datasets

```


2. **Load the Model:**
Run the initial cells to download the required RoBERTa models, tokenizers, and configuration files via Hugging Face.
3. **Evaluate the Jigsaw Dataset:**
The notebook will automatically fetch the `train-00000-of-00002.parquet` file from the Jigsaw Toxicity Prediction dataset to evaluate the model's baseline metrics (Accuracy, Precision, Recall, F1-Score).
4. **Generate LIME Explanations:**
Run the LIME cells to test custom phrases. The system will output an interactive HTML visualization.
*Example: Testing a toxic phrase will visually demonstrate which specific aggressive words or profanities drove the "Toxic" probability score up.*
5. **Run the Mitigation Pipeline:**
Pass simulated prompts through the filtering pipeline to see how the system successfully categorizes and filters outputs.

## Example Output

**Prompt:** *"What are social factors affecting crime rates?"*

```text
Filtered Model Response: What are the social factors affecting crime rates?
The following is an excerpt from Crime in the United States...
Non-Toxic Response (Score: 0.17)

```

**System Evaluation Metrics:**

```text
Evaluation Metrics:
Accuracy:
Precision: 
Recall: 
F1-Score: 

