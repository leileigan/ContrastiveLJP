## Exploiting Contrastive Learning and Numerical Evidence for Confusing Legal Judgment Prediction
### Introduce
This repository contains the data and code for the paper [Exploiting Contrastive Learning and Numerical Evidence for Confusing Legal Judgment Prediction.](https://arxiv.org/abs/2211.08238) 
Leilei Gan, Baokui Li, Kun Kuang, Yating Zhang, Lei Wang, Luu Anh Tuan, Yi Yang, Fei Wu
If you find this repository helpful, please cite the following:
```latex
@article{gan2022exploiting,
  title={Exploiting Contrastive Learning and Numerical Evidence for Improving Confusing Legal Judgment Prediction},
  author={Gan, Leilei and Li, Baokui and Kuang, Kun and Yang, Yi and Wu, Fei},
  journal={arXiv e-prints},
  pages={arXiv--2211},
  year={2022}
}
```
### Requirements

- Python == 3.7
- torch==1.8.1
- transformers==4.24.0
### Dataset and Trained crime amount encoding Model
You can download the initial dataset and processed dataset, as well as other resources through the link. Please put the downloaded data in the law directory.
And the following is the process of obtaining evidence with criminal amounts through the initial dataset and trained crime amount encoding Model.

1. First, run "data/tongji3.py" to get "_{}cs.json". 
2. Second, use the sequence labeling method to extract the crime amount. Please refer to the [link](https://github.com/hackerxiaobai/fyb-withdrawal-crime-amount) for the method.
> train_cs.json -> train_cs_with_number_process.json

3. Third, select the corresponding code in "data/data_pickle.py" and "data/make_Legal_basis_data.py" to generate the data structure according to the model you want to run.(**We put all the processed .pkl files in the link at the end of the article. If you do not want to process the data, you can download it directly from the link.**） 
4. Last, train the digital encoding model through DICE loss.
```
\begin{align}
    \mathbf{x}_i &= \text{NumEncoder}(x_i) \\
    \mathbf{y}_i &= \text{NumEncoder}(y_i) \\
    \ell_{num} &= \bigg\lVert \dfrac{2|x_i - y_i|}{|x_i| +|y_i|} -\text{cos}(\mathbf{x}_i, \mathbf{y}_i) \bigg\rVert
\end{align}
```

```shell
python train_dice.py
```

---

### Notice
Due to the NeurJudge model's different processing of input data, additional processing is required. And we offer both datasets in the link.
### Related link

1. [CAIL2018.zip](https://drive.google.com/file/d/1-OTqvewUJMT9dZ1fAbbRGhsolx6anQMl/view?usp=drive_link)(**exercise**for original small dataset; **first** + **restData** for original big dataset)
2. [cail_thulac.npy](https://drive.google.com/file/d/1_j1yYuG1VSblMuMCZrqrL0AtxadFUaXC/view?usp=drivesdk+ ) 
3. [w2id_thulac.pkl](https://drive.google.com/file/d/1jnNgilApBRnA2ihldOr1Ceaci_7aFtsD/view?usp=drive_link)
4. [Datasets](https://drive.google.com/file/d/1Ygm9QFsEEhwNPNaL786LmFGfCJoqvlQA/view?usp=drive_link) (We only upload the processed small dataset because the processed big dataset is too large. You can obtain the processed big dataset by following the data processing steps above) 
### Contact
If you have any issues or questions about this repo, feel free to contact **leileigan@zju.edu.cn.**
### License
[Apache License 2.0](./LICENSE)
