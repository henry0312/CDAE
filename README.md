# CDAE

Implementation of [Collaborative Denoising Auto-Encoder (CDAE)](http://yaowu.co/ "CDAE") with the [Keras](http://keras.io/ "Keras Documentation").

## References

* Yao Wu, Christopher DuBois, Alice X. Zheng, Martin Ester. Collaborative Denoising Auto-Encoders for Top-N Recommender Systems. The 9th ACM International Conference on Web Search and Data Mining (WSDM'16), p153--162, 2016.
* F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.

## Usage

First, install libraries:

```sh
pip install -r requirements.txt
```

Then,

```sh
# CPU
python train.py

# GPU
THEANO_FLAGS=device=gpu,floatX=float32 python train.py
```

## TODO

- [ ] implement negative sampling
- [ ] change the way of init

## Licence

MIT License
Copyright (c) 2016 Tsukasa ŌMOTO