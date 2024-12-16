# Natural Attention and Optimization in Transformers

This repository implements novel attention mechanisms and optimization techniques for transformer architectures, focusing on natural gradient approximation through attention patterns and critical phenomena in transformer networks.


## Project Structure

```
.
├── natural_attention.py      # Core implementation of Natural Attention mechanism
├── analysis/                 # to be done
│   ├── metrics.py           
│   └── visualization.py     
├── papers/
│   ├── attention-informed-optimization.pdf
│   ├── hierarchical-fim-preprint.pdf
│   └── transformer-criticality-paper.pdf
└── notebooks/
    └── Atención.ipynb      # Example notebook with training code for gpt2
```

## Key Components

### Natural Attention Implementation

The `NaturalAttention` class in `natural_attention.py` implements an attention mechanism that:
- Computes and stores raw attention energies
- Provides natural gradient information through attention patterns
- Integrates with standard transformer architectures

Key classes:
- `NaturalAttention`: Core attention mechanism
- `GPT2NaturalAttentionBlock`: GPT-2 compatible attention block
- `AttentionInformedOptimizer`: Custom optimizer leveraging attention patterns

### Training and Optimization

The training pipeline includes:
- Custom dataset handling for WikiText
- Parallel training of standard and natural attention models
- Integration with Weights & Biases for experiment tracking
- Attention-informed optimization techniques

## Installation

```bash
pip install wandb transformers datasets torch tqdm
```

## Usage

### Basic Training Example

```python
from transformers import GPT2Config
from natural_attention import GPT2NaturalAttentionBlock, AttentionInformedOptimizer

# Configuration
config_dict = {
    'max_length': 32,
    'batch_size': 4,
    'n_embd': 64,
    'n_layer': 2,
    'n_head': 2,
    'learning_rate': 1e-3,
    'epochs': 10,
    'save_every': 2
}

# Initialize models and train
standard_model, natural_model = train_models(config_dict)
```

### Analysis Tools

The repository includes tools for analyzing:
- Attention pattern dynamics
- Training convergence metrics
- Model performance comparisons
- Critical phenomena in transformer behavior

## Theoretical Background

This implementation is based on three key papers:

1. "Attention-Informed Optimization": Introduces the concept of using attention energies for optimization
2. "Attention as Natural Gradient": Establishes theoretical connections between attention and Fisher Information
3. "Criticality and Phase Transitions": Explores critical phenomena in transformer networks

## Experimental Results

Our implementation shows:
- Improved convergence rates with attention-informed optimization
- More stable attention patterns
- Better perplexity scores on language modeling tasks
- Evidence of critical behavior in transformer training

## Contributing

Contributions are welcome! Areas of particular interest:
- Additional analysis tools
- Performance optimizations
- New attention mechanisms
- Extended theoretical analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@article{aranda2024natural,
  title={Attention-Informed Optimization: Leveraging Attention Energies for Neural Network Training},
  author={Aranda Barois, Jeronimo},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and feedback:
- Open an issue in this repository
- Contact the authors through the paper correspondence
