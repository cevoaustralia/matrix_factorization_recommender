## Matrix Factorization Recommender
This repo contains End to end ML recommender using Matrix Factorization

- we have identified that we can leverage the Alternating Least Squares (ALS) algorithhm to build this recommender system as described in the paper [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf).

- We will generate our own dataset similar to the method used in [Cevo Shopping Demo](https://github.com/cevoaustralia/cevo-shopping-demo)

- Our dataset does not have any rating information (dataset does not have explicit feedback), so we will use the implicit feedback method to generate the recommendations.

- There is a python library called [implicit](https://github.com/benfred/implicit) which implements this algorithm as defined in the paper. We will use this library instead of implementing the algorithm from scratch.

- We will implement the **offline training-offline serving** pattern, this is with the assumption that the users and items are not changing that frequently. This will relax the training and inference requirements, so that we can take advantage of batch processing and caching, meaning our API can operate at massive scale.

- It is offline serving so that we don't have to use state of the art models that may be expensive to run. We will be hosting the API in a serverless offering, as it reads the user predictions off a fast storage like DynamoDB.

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
