# FLoRIST: Singular Value Thresholding for Efficient and Accurate Federated Fine-Tuning of Large Language Models

This repository contains the official implementation of **FLoRIST**, a framework for communication-efficient and performance-preserving federated fine-tuning of large language models (LLMs) using low-rank adapters and singular value thresholding. FLoRIST has been submitted to NeurIPS 2025.

In federated learning settings, where training data remains decentralized across clients (e.g., institutions or devices), fine-tuning large models becomes challenging due to communication and computational constraints. Parameter-efficient fine-tuning (PEFT) methods such as LoRA allow clients to train compact low-rank adapters locally, but aggregating these adapters efficiently and effectively remains an open problem—especially under heterogeneous client configurations.

FLoRIST addresses this by:
- **Aggregating directly in the low-rank latent space**, avoiding the construction of full dense update matrices.
- **Applying singular value thresholding** to retain only the most informative components, enabling compact and performant global adapters.
- **Supporting heterogeneous local ranks** across clients without requiring full-rank communication or complex coordination.
- **Introducing two variants**: `FLoRIST-O` for optimal performance, and `FLoRIST-E` for maximum communication efficiency.

FLoRIST outperforms state-of-the-art baselines such as FedIT, FLoRA, FlexLoRA, and FFA-LoRA across multiple datasets (Dolly, Alpaca, WizardLM) and model scales (TinyLlama, Llama-3.2-1B, Llama-7B), achieving  **lower communication** while matching or exceeding their accuracy.

<p align="center">
  <img src="./figures/workflow-8.pdf" width="75%" alt="FLoRIST Workflow">
</p>

## Requirements

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Datasets

We follow the same data format and directory structure as used in the original [FLoRA repository](https://github.com/ATP-1010/FederatedLLM). All datasets are stored in JSON format and should be placed in the appropriate folders.

### Available Datasets

- **Wizard**  
  - Source: [WizardLM/WizardLM_evol_instruct_70k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_70k)  
  - Path:  Downloaded and pre-split in `./data_wiz/`  

- **Alpaca**  
  - Source: [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)  
  - Path:  Downloaded and pre-split in `./data_alpaca/`  

- **Dolly**  
  - Source: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)  
  - Path:  Downloaded and pre-split in `./data/`  

If you want to use a custom dataset, make sure it follows the same instruction-response JSON format as these folders. Each sample should contain fields such as `"instruction"`, `"input"` (optional), and `"output"`.


## Training

To train a model in a **homogeneous** federated setting:

**FLoRIST**
```bash
python3 main.py --global_model 'tinyllama' \
  --data_path "./data" \
  --output_dir './tinyllama-dolly-homo-1-3-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --florist True \
  --num_clients 8 \
  --threshold 0.9
```

**FLoRA**
```bash
python3 main.py --global_model 'tinyllama' \
  --data_path "./data" \
  --output_dir './tinyllama-dolly-homo-1-3-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --stacking True \
  --num_clients 8
```

**FedIT**
```bash
python3 main.py --global_model 'tinyllama' \
  --data_path "./data" \
  --output_dir './tinyllama-dolly-homo-1-3-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --num_clients 8
```

**FlexLoRA**
```bash
python3 main.py --global_model 'tinyllama' \
  --data_path "./data" \
  --output_dir './tinyllama-dolly-homo-1-3-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --flex True \
  --num_clients 8
```

**FFA-LoRA**
```bash
python3 main.py --global_model 'tinyllama' \
  --data_path "./data" \
  --output_dir './tinyllama-dolly-homo-1-3-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --ffa True \
  --num_clients 8
```

To train in a **heterogeneous** client rank setup, add `--heter True`. For methods that do not support heterogeneity (e.g., FedIT and FFA-LoRA), add `--zero_padding True`.

Example:

```bash
python3 main.py --global_model 'huggyllama/llama-7b' \
  --data_path "./data_wiz" \
  --output_dir './llama7b-wiz-heter-1-1-8/' \
  --num_communication_rounds 1 \
  --local_num_epochs 3 \
  --florist True \
  --num_clients 8 \
  --threshold 0.80 \
  --heter True
```

## Evaluation

All training runs automatically evaluate the final global model on the [MMLU benchmark](https://huggingface.co/datasets/mmlu) and report accuracy at the end.


## Results

FLoRIST achieves state-of-the-art trade-offs between accuracy and communication efficiency across all model sizes, datasets, and client types.

### Results on Homogeneous Setting (MMLU Benchmark)

| Model           | Method         | Dolly Acc (%) | Dolly Eff. (×10⁻⁴) | Alpaca Acc (%) | Alpaca Eff. (×10⁻⁴) | Wizard Acc (%) | Wizard Eff. (×10⁻⁴) |
|-----------------|----------------|----------------|--------------------|----------------|---------------------|----------------|----------------------|
| **TinyLlama**   | FedIT          | 28.88          | 14.20              | **31.99**      | 14.20               | 41.42          | 14.20                |
|                 | FLoRA          | 27.48          | 1.78               | 29.09          | 1.78                | 41.99          | 1.78                 |
|                 | FlexLoRA       | 28.03          | 14.20              | 29.00          | 14.20               | _42.53_        | 14.20                |
|                 | FFA-LoRA       | 24.74          | 28.40              | 25.57          | 28.40               | 26.31          | 28.40                |
|                 | FLoRIST-O      | **30.42** (τ=0.87) | 45.40          | _29.81_ (τ=0.93) | 34.36               | **43.63** (τ=0.99) | 16.92           |
|                 | FLoRIST-E      | _29.25_ (τ=0.80) | **76.30**       | 29.43 (τ=0.84) | **63.30**           | 42.39 (τ=0.82) | **73.50**            |
| **Llama-7b**     | FedIT          | _34.75_        | 9.77               | 27.38          | 9.77                | 28.50          | 9.77                 |
|                 | FLoRA          | 34.38          | 1.22               | 26.34          | 1.22                | 28.50          | 1.22                 |
|                 | FlexLoRA       | 33.88          | 9.77               | 26.27          | 9.77                | 28.69          | 9.77                 |
|                 | FFA-LoRA       | 31.52          | 19.50              | 22.69          | 19.50               | 28.34          | 19.50                |
|                 | FLoRIST-O      | **35.58** (τ=0.95) | 21.40          | **29.05** (τ=0.85) | 57.47          | **29.25** (τ=0.95) | 29.41           |
|                 | FLoRIST-E      | 34.45 (τ=0.85) | **51.02**          | _28.30_ (τ=0.80) | **70.90**        | _29.14_ (τ=0.87) | **52.90**            |
| **Llama-3.2-1B** | FedIT          | 19.07          | 19.50              | 25.99          | 19.50               | 27.27          | 19.50                |
|                 | FLoRA          | 18.97          | 2.44               | **30.34**      | 2.44                | 27.48          | 2.44                 |
|                 | FlexLoRA       | 19.45          | 19.50              | 30.16          | 19.50               | 27.01          | 19.50                |
|                 | FFA-LoRA       | 19.59          | 39.06              | 18.68          | 39.06               | _28.01_        | 39.06                |
|                 | FLoRIST-O      | **20.68** (τ=0.95) | 37.59          | _30.29_ (τ=0.99) | 18.10          | **28.29** (τ=0.95) | 38.80           |
|                 | FLoRIST-E      | _19.95_ (τ=0.82) | **64.93**       | 29.66 (τ=0.80) | **94.30**           | 27.18 (τ=0.82) | **87.70**            |

_Bold = Highest, Italic = Second-Highest_

See the full table in the [paper] for all datasets and baseline comparisons.

## License

This repository is released under the Apache License 2.0. See `LICENSE` for details.

## Contributing

We welcome pull requests for reproducibility enhancements, dataset loaders, and benchmarking scripts. For major changes, please open an issue first to discuss what you would like to change.
