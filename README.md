## Matrix Factorization Recommender
This repo contains code to generate data and train an ML recommender using a Matrix Factorization library called [implicit](https://github.com/benfred/implicit)

- we have identified that we can leverage the Alternating Least Squares (ALS) algorithm to build this recommender system as described in the paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf).

- We will generate our own dataset similar to the method used in [Cevo Shopping Demo](https://github.com/cevoaustralia/cevo-shopping-demo)

- Our dataset does not have any rating information (dataset does not have explicit feedback), so we will use the implicit feedback method to generate the recommendations.

- There is a python library called [implicit](https://github.com/benfred/implicit) which implements this algorithm as defined in the paper. We will use this library instead of implementing the algorithm from scratch.

- This model is online serving (meaning that the predictions are done in real time), and because we are hosting this model in a serverless in AWS Lambda, it may not be grunty enough for a larger dataset.

- MLOps pipeline will take care of model building, hyperparameter tuning, model evaluation, model serving and monitoring. We will use <insert your tool choice> for this.

## Tech Stack
- GitHub Actions
- Metaflow
- AWS S3
- Comet ML for experiment tracking
- SAM
- FastAPI
- DynamoDB
- GridSearch CV (or manual grid search)
- Model Registry (?)
- Feature Store (?)

#### To run:
1. ```python recommender.py show```
1. ```python recommender.py run```
