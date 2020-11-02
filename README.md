# UWMMSE
Tensorflow implementation of Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation (https://arxiv.org/abs/2009.10812)

## Overview
This library contains a Tensorflow implementation of Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation as presented in [[1]](#citation)(https://arxiv.org/abs/).
## Dependencies

* **python>=3.6**
* **tensorflow>=1.14.0**: https://tensorflow.org
* **numpy**
* **matplotlib**

## Structure
* [datagen](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/datagen.py): Code to generate dataset. Generates A.pkl ( Geometric graph ), H.pkl ( Dictionary containing train_H and test_H ) and coordinates.pkl ( node position coordinates ).  Run as *python3 datagen.py {dataset ID}*. User chosen \[dataset ID\] will be used as the foldername to store datset. 
* [data](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/data): should contain your dataset in folder {dataset ID}. 
* [main](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/main.py): Main code for running the experiments in the paper. Run as python3 main.py {dataset ID} {exp ID} {mode}. For ex. to train UWMMSE on dataset with ID set3, run *python3 main.py set3 uwmmse train*.
* [model](https://github.com/ArCho48/Unrolled-WMMSE/blob/master/model.py): Defines the UWMMSE model.
* [models](https://github.com/ArCho48/Unrolled-WMMSE/tree/master/models): Stores trained models in a folder with same name as {dataset ID}.

## Usage


Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Arindam Chowdhury](mailto:arindam.chowdhury@rice.edu).

## Citation
```
[1] Chowdhury A, Verma G, Rao C, Swami A, Segarra S. Unfolding WMMSE using Graph Neural Networks 
for Efficient Power Allocation. arXiv preprint arXiv:2009.10812. 2020 Sep 22.
```

BibTeX format:
```
@article{chowdhury2020unfolding,
  title={Unfolding WMMSE using Graph Neural Networks for Efficient Power Allocation},
  author={Chowdhury, Arindam and Verma, Gunjan and Rao, Chirag and Swami, Ananthram and Segarra, Santiago},
  journal={arXiv preprint arXiv:2009.10812},
  year={2020}
}

```
