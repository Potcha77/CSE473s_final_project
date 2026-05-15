# CSE473s – Build Your Own Neural Network Library

## Structure
```
.
├── lib/                  # Pure-NumPy neural network library
│   ├── __init__.py
│   ├── layers.py         # Layer, Dense, Dropout
│   ├── activations.py    # ReLU, Sigmoid, Tanh, Softmax
│   ├── losses.py         # MSE, BinaryCrossEntropy
│   ├── optimizer.py      # SGD with Momentum
│   └── network.py        # Sequential model
├── notebooks/
│   └── project_demo.ipynb
├── report/
│   └── project_report.pdf
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebooks/project_demo.ipynb
```

## Milestones
| Part | Description |
|------|-------------|
| 1    | Library implementation + XOR validation + gradient checking |
| 2    | Denoising autoencoder on Fashion-MNIST + SVM classifier + TF comparison |
