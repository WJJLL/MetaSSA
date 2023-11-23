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

  Download pre-trained PyTorch models [here](https://github.com/ylhz/tf_to_pytorch_model), which are converted from widely used Tensorflow models. Then put these models into `./models/`

- **Generate adversarial examples under inception-v3** -

  ```bash
  CUDA_VISIBLE_DEVICES=gpuid  python main.py --model_type inceptionv3
  ```
- **Evaluations on normally trained and AT models**
  ```bash
  python verify.py
  ```
- **Evaluations on other defenses**

    To evaluate the attack success rates on six more advanced models (HGD, R&P, NIPS-r3, RS, JPEG, NRP).

    - [Inc-v3<sub>*ens3*</sub>,Inc-v3<sub>*ens4*</sub>,IncRes-v2<sub>*ens*</sub>](https://github.com/ylhz/tf_to_pytorch_model):  You can directly run `verify.py` to test these models.
    - [HGD](https://github.com/lfz/Guided-Denoise), [R&P](https://github.com/cihangxie/NIPS2017_adv_challenge_defense), [NIPS-r3](https://github.com/anlthms/nips-2017/tree/master/mmd): We directly run the code from the corresponding official repo.
    - [RS](https://github.com/locuslab/smoothing): noise=0.25, N=100, skip=100. Download it from the corresponding official repo.
    - [JPEG](https://github.com/thu-ml/ares/blob/main/ares/defense/jpeg_compression.py): No extra parameters.
    - [NRP](https://github.com/Muzammal-Naseer/NRP): purifier=NRP, dynamic=True, base_model=Inc-v3<sub>*ens3*</sub>. Download it from the corresponding official repo.
 
      More details in [third_party](https://github.com/JHL-HUST/VT/tree/main/third_party)https://github.com/JHL-HUST/VT/tree/main/third_party)
