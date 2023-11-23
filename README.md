# MetaSSA
Exploring Frequencies via Feature Mixing and Meta-Learning for Improving Adversarial Transferability
## Requirements

- python 3.9
- torch 1.8
- pretrainedmodels 0.7
- numpy 1.19
- pandas 1.2


## Implementation
- **Prepare models**

  Download pretrained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./models/`

- **Generate adversarial examples under inception-v3 ** -
