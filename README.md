# Cohort Shapley Integrated Gradients
Cohort Shapley Integrated Gradients (CSIG) is a calculation method of local feature attribution
that apply multilinear extensions which make integrated gradients equivalent to Shapley values in an original space to the space of
indicator functions, not to data space.
Then it introduces integrated gradients on the indicator space. Our method is based on empirical distribution
similar to Cohort Shapley (CS), and can evaluate feature attributions with a linear complexity
to the number of features. In addition, CSIG does not require any differentiability on the model nor other assumptions, since line integrals of CSIG
are held in the indicator space.



See the [paper]():

And details of the cohort shapley for the [paper](https://arxiv.org/abs/1911.00467):
> Mase, M., Owen, A. B., & Seiler, B. (2019). Explaining black box decisions by Shapley cohort refinement. arXiv preprint arXiv:1911.00467.


# Install
Install the package locally with pip command.
```bash
git clone https://github.com/cohortshapley/cohortshapleyintegratedgradients
pip install -e cohortshapleyintegratedgradients
```

## Prerequisites
This code is tested on:
- Python 3.10.4
- pip 22.3
- NumPy 1.23.3
- Pytorch 1.12.1
- shap 0.41.0
- tqdm 4.64.1

For example notebooks, we need:
- scikit-learn 1.1.2

# Getting Started
See Jupyter notebook examples in [example](example) directory.