### Data Preparation
---
When you get the CAIL datasets, run 'data_and_config/data/tongji3.py' to get '_{}cs.json' first. Then choose the corresponding code among 'data_processed/data_pickle.py', 'data_processed_big/data_pickle_big.py', and 'data/make_Legal_basis_data.py' to generate the data structure according to model you want to run.

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
