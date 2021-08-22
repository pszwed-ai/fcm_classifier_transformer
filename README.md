# fcm_classifier_transformer

This repository contains source code to the article:
[*Piotr Szwed: Classification and feature transformation with Fuzzy Cognitive Maps, Applied Soft Computing, Elsevier 2021*](https://arxiv.org/pdf/2103.05124.pdf)

## Abstract
Fuzzy Cognitive Maps (FCMs) are considered a soft computing technique combining elements of fuzzy logic and recurrent neural networks. They found multiple application in such domains as modeling of system behavior, prediction of time series, decision making and process control. Less attention, however, has
been turned towards using them in pattern classification. In this work we propose an FCM based classifier with a fully connected map structure. In contrast
to methods that expect reaching a steady system state during reasoning, we chose to execute a few FCM iterations (steps) before collecting output labels.
Weights were learned with a gradient algorithm and logloss or cross-entropy were used as the cost function. Our primary goal was to verify, whether such
design would result in a descent general purpose classifier, with performance comparable to off the shelf classical methods. As the preliminary results were
promising, we investigated the hypothesis that the performance of d-step classifier can be attributed to a fact that in previous d − 1 steps it transforms the
feature space by grouping observations belonging to a given class, so that they became more compact and separable. To verify this hypothesis we calculated
three clustering scores for the transformed feature space. We also evaluated performance of pipelines built from FCM-based data transformer followed by a
classification algorithm. The standard statistical analyzes confirmed both the performance of FCM based classifier and its capability to improve data. The
supporting prototype software was implemented in Python using TensorFlow library.

## Requirements
The code was written in 2018 using TensorFlow 1.6 library, however recently it was ported to TF 2.6. 

Current configuration:

* Python 3.8.3
* scikit-learn 0.24.2
* scipy 1.5.0
* numpy 1.19.3
* Tensorflow 2.6.0

## Code structure
* ```base``` package comprises the code of FCM classifier, binary and multiclass classifiers are defined in ```binary_classifier.py``` and ```mc_classifier.py```
* ```util``` package provides access to datasets used during experiments
* ```use_cases.multiclass``` 
  * ```cv_fcm_transformer.py``` - run this script. It implements [the pipeline in Fig. 7 pg. 20](https://arxiv.org/pdf/2103.05124.pdf) 
  * ```gmm_classifier.py``` - implementation of a classifier based on a mixture of Gaussian distributions (used during experiments)
  * ```fcm_best_params.py``` - a Bunch (dictionary) of parameters for particular datasets. These parameters were established manually, typically by random search

## Execution

After launching ```cv_fcm_transformer.py``` a file ```results_fcm_transformer.py``` is created in the folder ```use_cases/multiclass/results```. It comprises a giant multilevel dictionary gathering detailed results data. 

The current version was tested on Windows 10 laptop equiped with 4GB GPU. The whole data processing took about 5 hours. However, the TF1.6 version was tested on Windows and VMware virtual Linux machine not using GPU. The VMware environment was more efficient.

## Issues
During an execution two waring messages frequently appear. 

```W tensorflow/core/data/root_dataset.cc:167] Optimization loop failed: Cancelled: Operation was cancelled```

```validation.py:70: FutureWarning: Pass labels=[0 1 2 3 4 5 6 7 8 9] as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error   warnings.warn(f"Pass {args_msg} as keyword args. From version ```


## Citation
```
@article{szwed2021classification,
  title={Classification and feature transformation with Fuzzy Cognitive Maps},
  author={Szwed, Piotr},
  journal={Applied Soft Computing},
  volume={105},
  pages={107271},
  year={2021},
  publisher={Elsevier}
}
```
