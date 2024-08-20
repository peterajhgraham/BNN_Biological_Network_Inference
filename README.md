# Bayesian Neural Network Biological Network Inference

This BNN BNI Analysis Tool can be used for predictive insight in Biological Networks through the use of Bayesian Neural Networks. This repository can be used for any biological datasets, however it is tailored toward individuals who are interested in computational biology, computational systems biology, computational genomics, computational pharmacology, and computational neuroscience. It also made just as available to anyone who would like to use it for their own learning endeavours!

## Directory Structure

```
BNN_Biological_Network_Inference/
├── data/
│   └── example_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── utils.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Example Databases

* **[Protein Data](https://www.wwpdb.org/)**

* **[Genomic Data](https://www.genomicsengland.co.uk/)**

* **[Drug Interaction Data](https://go.drugbank.com/)**

* **[Metabolite Data](https://www.metabolomicsworkbench.org/databases/metabolitedatabase.php)**

* **[Microbiome Data](https://portal.hmpdacc.org/)**

### Data Structure

Your `.csv` files should follow this structure:

1. **Features**: Columns representing various features.
2. **Target Variable**: A column for the target variable that represents the interaction score or outcome you want to predict.

**Example Structure**:

feature1,feature2,feature3,target

value1,value2,value3,target_value

## Instructions

### 1. Prepare Your Data

* **Copy The Repository**:

```
git clone https://github.com/peterajhgraham/BNN_Biological_Network_Inference.git
cd BNN_Biological_Network_Inference
```
* **Format Your Data**: Ensure your `.csv` files follow the structure outlined above.

* **Place Data Files**: Save your `.csv` files in the `data/` directory of the repository.

### 2. Install Required Packages

Install the required Python packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```

### 3. Preprocess Data

Run the data preprocessing script to load and prepare your data for modeling:

```python
from src.data_preprocessing import load_data, preprocess_data

file_paths = {
    'example': 'data/example_data.csv'
}

data = load_data(file_paths)
processed_data = preprocess_data(data)
```

### 4. Train the Model

Use the training script to build and train your Bayesian Neural Network model with torchbnn:

```python
from src.train import train_model

input_dim = processed_data['example'][0].shape[1]  # Example for one dataset
model = train_model(*processed_data['example'], input_dim)
```

### 5. Make Predictions

After training, use the inference script to make predictions with your model:

```python
from src.inference import infer

predictions = infer(model, processed_data['example'][1])
```

### 6. Visualize Results

You can use the utility functions to visualize your results:

```python
from src.utils import plot_predictions

plot_predictions(processed_data['example'][2], predictions)
```
