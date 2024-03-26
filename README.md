### Introduction
This repository contains the data and code for the paper **[Exploiting Contrastive Learning and Numerical Evidence for Confusing Legal Judgment Prediction](https://aclanthology.org/2023.findings-emnlp.814/)**.
<br>Leilei Gan, Baokui Li, Kun Kuang, Yating Zhang, Lei Wang, Anh Luu, Yi Yang, Fei Wu</br>

If you find this repository helpful, please cite the following:
```tex
@inproceedings{gan2023exploiting,
  title={Exploiting Contrastive Learning and Numerical Evidence for Confusing Legal Judgment Prediction},
  author={Gan, Leilei and Li, Baokui and Kuang, Kun and Zhang, Yating and Wang, Lei and Luu, Anh and Yang, Yi and Wu, Fei},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  pages={12174--12185},
  year={2023}
}
```

### Data Preparation
---
When you get the CAIL datasets, run 'data/tongji3.py' to get '_{}cs.json' first. Then choose the corresponding code among 'data/data_pickle.py', and 'data/make_Legal_basis_data.py' to generate the data structure according to model you want to run.

You can get the word embeddin file "cail_thulac.npy" at the following address: https://drive.google.com/file/d/1_j1yYuG1VSblMuMCZrqrL0AtxadFUaXC/view?usp=drivesdk+ the dataset file "CAIL2018.zip" at the address: https://drive.google.com/file/d/12QOsAumyzmdsqgqhqcNOhOaGcp--ECnm/view?usp=sharing https://drive.google.com/file/d/1JsP6co2GiCodv4oqyjaa6Q8TY5FWOQDM/view?usp=sharing and https://drive.google.com/file/d/1VX8YhqF6ZX2tVLeOJV4VxdnNkWEDrbyS/view?usp=sharing

---
### Train HARNN model

```shell
CUDA_VISIBLE_DEVICES=0 python train_harnn.py
```

### Train LADAN model

```shell
CUDA_VISIBLE_DEVICES=0 python train_ladan_mtl.py
```
