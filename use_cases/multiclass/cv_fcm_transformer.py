# -*- coding: utf-8 -*-
"""
Created on 22.12.2018 18:29

@author: Piotr Szwed, pszwed@agh.edu.pl
"""
import os

import copy
import datetime
import sys
import time

from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch

from base.binary_classifier import FcmBinaryClassifier
from base.hooks import ReportTrainingLoss
from base.mc_classifier import FcmMulticlassClassifier
from use_cases.multiclass.fcm_best_params import get_best_params, get_cv_metrics
from use_cases.multiclass.gmm_classifier import GaussianMixtureClassifier
from use_cases.multiclass.model_losses import model_loss_l2_w, model_loss_l2_b, model_loss_l1_w, model_loss_l1_b, \
    model_loss_maxabs_w, model_loss_maxabs_b
from util.datasets import load_dataset
import tensorflow as tf
import numpy as np

def fit_transform_test(transformer, X_train, y_train, X_test, y_test, transformer_scoring, cls_scoring, return_train_score=True,return_confusion_matrix=True):
    labels = np.sort(np.unique(np.concatenate([np.unique(y_train),np.unique(y_test)])))

    start_time = time.process_time()
    transformer.fit(X_train, y_train)
    fit_time = time.process_time() - start_time

    start_time = time.process_time()
    Z_train = transformer.transform(X_train)
    Z_test = transformer.transform(X_test)
    transform_time = time.process_time() - start_time

    result = {}

    start_time = time.process_time()

    feature_scores = transformer_scoring.get('features')
    if feature_scores is not None:
        for ts in feature_scores:
            scorer = transformer_scoring['features'][ts]
            if return_train_score:
                train_score = scorer(X_train,y_train)
                result['train_original_' + ts] = train_score
                train_score = scorer(Z_train, y_train)
                result['train_transformed_' + ts] = train_score
            test_score = scorer(X_test,y_test)
            result['test_original_' + ts] = test_score
            train_score = scorer(Z_test, y_test)
            result['test_transformed_' + ts] = train_score

    model_scores = transformer_scoring.get('model')
    for ts in model_scores:
        score = transformer_scoring['model'][ts](transformer)
        result[ts]=score

    if return_confusion_matrix:
        # y_pred = estimators.predict(X_test)
        result['confusion_matrix'] = confusion_matrix(y_test, transformer.predict(X_test),labels)


    for m in cls_scoring:
        if return_train_score:
            train_score = get_scorer(m)(transformer,X_train,y_train)
            result['train_' + m] = train_score
        test_score = get_scorer(m)(transformer,X_test,y_test)
        result['test_' + m] = test_score
    score_time = time.process_time() - start_time


    result['fit_time'] = fit_time
    result['transform_time'] = transform_time
    result['score_time'] = score_time
    return Z_train, Z_test, result


def fit_test(classifier, X_train, y_train, X_test, y_test, scoring, return_train_score=True,return_confusion_matrix=True):
    labels = np.sort(np.unique(np.concatenate([np.unique(y_train),np.unique(y_test)])))

    # fix for potential negative coordinates resulting from scaling
    if isinstance(classifier, MultinomialNB):
        X_test = np.where(X_test >= 0, X_test, 0)

    start_time = time.process_time()
    classifier.fit(X_train, y_train)
    fit_time = time.process_time() - start_time

    start_time = time.process_time()

    result = {}
    for m in scoring:
        scorer = get_scorer(m)

        if return_train_score:
            train_score = scorer(classifier,X_train,y_train)
            result['train_' + m] = train_score
        test_score = scorer(classifier,X_test,y_test)
        result['test_' + m] = test_score
        score_time = time.process_time() - start_time

        result['fit_time'] = fit_time
        result['score_time'] = score_time

    if return_confusion_matrix:
        # y_pred = estimators.predict(X_test)
        result['confusion_matrix'] = confusion_matrix(y_test, classifier.predict(X_test),labels)

    return result




def _process_fold(fold_number,transformer_step,classifier_steps, X_train, y_train, X_test, y_test, transformer_scoring, cls_scoring, return_train_score=True, return_confusion_matrix=True,scale_transformed_features = True):

    transformer = copy.copy(transformer_step[1])
    Z_train, Z_test, result = fit_transform_test(transformer,
                                                 X_train, y_train, X_test, y_test,
                                                 transformer_scoring, cls_scoring,
                                                 return_confusion_matrix=return_confusion_matrix,
                                                 return_train_score=return_train_score)

    if scale_transformed_features:
        scaler = MinMaxScaler()
        Z_train = scaler.fit_transform(Z_train)
        Z_test = scaler.transform(Z_test)


    b = Bunch()
    b['info'] = {'fold_number':fold_number,'fcm_classifier':'{}'.format(transformer_step[1].__class__.__name__)}

    b[ transformer_step[0] ] = result

    for step in classifier_steps:
        classifier  = copy.copy(step[1])
        result = fit_test(classifier,X_train,y_train,X_test,y_test,cls_scoring,return_train_score=return_train_score,return_confusion_matrix=return_confusion_matrix)
        b[ step[0] ] = result


    for step in classifier_steps:
#        print(step[0])
#        if step[0] == 'gmc_1f' or step[0] == 'gmc_3f' or step[0] == 'gmc_3d':
#            print('skipping')
#            continue
        
        classifier  = copy.copy(step[1])
        result = fit_test(classifier,Z_train,y_train,Z_test,y_test,cls_scoring,return_train_score=return_train_score,return_confusion_matrix=return_confusion_matrix)
        b[ transformer_step[0]+'_'+step[0] ] = result
    return b



def cross_validate(transformer_step,classifier_steps, X, y, transformer_scoring, cls_scoring, n_splits=5, return_train_score=True,return_confusion_matrix=True,random_state=None,scale_transformed_features=True):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)

    splits = [(X[train_index],y[train_index],X[test_index],y[test_index]) for train_index, test_index in skf.split(X, y)]

    parallel = Parallel(n_jobs=n_splits,pre_dispatch=2*n_splits,verbose=0)
    results = parallel(
        delayed(_process_fold)(
            fold_number,
            transformer_step, classifier_steps, X_train, y_train, X_test, y_test,
            transformer_scoring, cls_scoring, return_train_score=return_train_score,
            return_confusion_matrix=return_confusion_matrix,
            scale_transformed_features=scale_transformed_features)
        for fold_number, (X_train, y_train, X_test, y_test) in enumerate(splits))
        # results is a bunch step -> score -> value

    aggregated_results = Bunch()
    for fold_result in results:
        for method in fold_result:
            # # -------- process fold number -------
            # key = method
            # if key == 'fold_number':
            #     fn = fold_result[key]
            #     fold_nmubers = aggregated_results.get(key)
            #     if fold_nmubers is None:
            #         aggregated_results[key] = [fn]
            #     else:
            #         aggregated_results[key].append(fn)
            #     continue
            # # ------------------------------------
            scores =  aggregated_results.get(method)
            if scores is None:
                aggregated_results[method] = Bunch()
                scores = aggregated_results[method]
            for score_name in fold_result[method]:
                score = fold_result[method][score_name]
                l = scores.get(score_name)
                if l is None:
                    scores[score_name]=[score]
                else:
                    scores[score_name].append(score)

        aggregated_results['params']=Bunch()
        aggregated_results.params[transformer_step[0]]=transformer_step[1].get_params()
        for cls_step in classifier_steps:
            aggregated_results.params[cls_step[0]] = cls_step[1].get_params()

    return aggregated_results


def cross_validate_sequential(transformer_step, classifier_steps, X, y, transformer_scoring, cls_scoring, n_splits=5,
                   return_train_score=True, return_confusion_matrix=True,random_state=None,scale_transformed_features=True):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)

    aggregated_results = Bunch()

    for fold_number, (train_index, test_index) in enumerate(skf.split(X, y)):
        print('Processing fold:%d' % fold_number)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        results = _process_fold(fold_number,transformer_step, classifier_steps, X_train, y_train, X_test, y_test,
                                transformer_scoring, cls_scoring, return_train_score=return_train_score,
                                return_confusion_matrix=return_confusion_matrix,
                                scale_transformed_features=scale_transformed_features)
        #            process_fold(X,Y,train_index,test_index)
        # results is a bunch step -> score -> value

        for method in results:
            scores = aggregated_results.get(method)
            if scores is None:
                aggregated_results[method] = Bunch()
                scores = aggregated_results[method]
            for score_name in results[method]:
                score = results[method][score_name]
                l = scores.get(score_name)
                if l is None:
                    scores[score_name] = [score]
                else:
                    scores[score_name].append(score)

        aggregated_results['params']=Bunch()
        aggregated_results.params[transformer_step[0]]=transformer_step[1].get_params()
        for cls_step in classifier_steps:
            aggregated_results.params[cls_step[0]] = cls_step[1].get_params()

    return aggregated_results


def configure_transformer(dataset):
    report_hook = ReportTrainingLoss(epoch_interval=100,batch_interval=-1)
    #TODO remove reference to get_cv_metrics
    if get_cv_metrics(dataset)['classifier'] == 'FcmMulticlassClassifier':
        fcm_transformer = FcmMulticlassClassifier(training_hook=[report_hook])
    else:
        fcm_transformer = FcmBinaryClassifier(training_hook=[report_hook])
    params = get_best_params(dataset)
    fcm_transformer.set_params(**params)
    return fcm_transformer


def print_results(dataset, cv_results, file=sys.stdout):
    # print('\n\n\n',file=file)
    # print('b.{} = Bunch()'.format(dataset),file=file)
    for method in cv_results:
        print('\n',file=file)
        print('# --------------------------------------------------------------------------------------------------------------------',file=file)
        print('\n',file=file)
        print('b.{}.{} = '.format(dataset,method),file=file,end='{\n')
        method_results = cv_results[method]

        for score in method_results:
            results = method_results[score]

            if method == 'params' or method == 'info':
                print('    \'{}\':{},'.format(score, results), file=file)
                continue

            if score == 'confusion_matrix':
                # print(results)
                cm = np.sum(np.array(results), axis=0)
                # print(cm)
                cm = np.array2string(cm, separator=', ').replace('\n', ' ')
                print('    \'{}\':{},'.format(score, cm), file=file)
            else:
                r = np.array(results)
                r = cm = np.array2string(r, separator=', ')

                print('    \'{}\':{},'.format(score, r), file=file)
                mean = np.mean(results)
                key = score.replace('test_', '') + '_mean'
                print('    \'{}\':{},'.format(key, mean), file=file)

                std = np.std(results)
                key = score.replace('test_', '') + '_std'
                print('    \'{}\':{},'.format(key, std), file=file)
        print('}', file=file)



def perform_cv(dataset,classifier_steps,transformer_scoring, cls_scoring,file=sys.stdout,random_state=None,return_confusion_matrix=True,scale_transformed_features=True):
    X,y = load_dataset(dataset)
    transformer = configure_transformer(dataset)
    # results = cross_validate_sequential(('fcm',transformer),classifier_steps,
    #                          X,y,
    #                          transformer_scoring, cls_scoring,
    #                          n_splits=5, return_train_score=True,random_state=random_state,
    #                          return_confusion_matrix=return_confusion_matrix,
    #                          scale_transformed_features=scale_transformed_features)
    results = cross_validate(('fcm',transformer),classifier_steps,
                             X,y,
                             transformer_scoring, cls_scoring,
                             n_splits=5, return_train_score=True,random_state=random_state,
                             return_confusion_matrix=return_confusion_matrix,
                             scale_transformed_features=scale_transformed_features)

    print_results(dataset, results, file=file)

def perform_cv_multiple(datasets,classifier_steps,transformer_scoring, cls_scoring,file=sys.stdout,random_state=None,return_confusion_matrix=True,scale_transformed_features=True):
    print('from sklearn.utils import Bunch',file=file)
    print('\nb = Bunch\n',file=file)
    print('# ====================================================================================================================', file=file)
    print('# Started at '+datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), file=file)
    print('# ====================================================================================================================',file=file)
    print('\n\n',file=file)
    for dataset in datasets:
        print('\n\n# Dataset: {}\n'.format(dataset),file=file)
        print('b.{} = Bunch()'.format(dataset), file=file)
        perform_cv(dataset, classifier_steps, transformer_scoring, cls_scoring, file=file, random_state=random_state,return_confusion_matrix=return_confusion_matrix,scale_transformed_features=scale_transformed_features)
        file.flush()
        os.fsync(file)
    print('\n\n',file=file)
    print('# ====================================================================================================================', file=file)
    print('# Finished at '+datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"), file=file)
    print('# ====================================================================================================================',file=file)


CLUSTERING_SCORES={'davies_bouldin':davies_bouldin_score,'silhouette':silhouette_score,
                   'calinski_harabasz':calinski_harabasz_score}

tsf_scores = Bunch()
tsf_scores.features=CLUSTERING_SCORES


tsf_scores.model={  'l2_w':model_loss_l2_w,
                    'l2_b':model_loss_l2_b,
                    'l1_w':model_loss_l1_w,
                    'l1_b':model_loss_l1_b,
                    'maxabs_w': model_loss_maxabs_w,
                    'maxabs_b': model_loss_maxabs_b,
                    }


scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']


classifiers = [('mnb',MultinomialNB(alpha=0.01)),
               ('gnb',GaussianNB()),
               ('knn3',KNeighborsClassifier(n_neighbors = 3)),
               ('knn5',KNeighborsClassifier(n_neighbors = 5)),
               ('svcrbf', SVC(kernel='rbf',gamma='scale')),
               ('svclin', SVC(kernel='linear')),
               ('logreg', LogisticRegression()),
               ('dtree', DecisionTreeClassifier()),
               ('rforest', RandomForestClassifier()),
               ('gmc_1f',GaussianMixtureClassifier(n_comp=1, covariance_type='full')),
               ('gmc_3f',GaussianMixtureClassifier(n_comp=3, covariance_type='full')),
               ('gmc_3d',GaussianMixtureClassifier(n_comp=3, covariance_type='diag')),
               ]


# dataset='iris'
# perform_cv(dataset,classifiers,tsf_scores, scoring,file=sys.stdout,random_state=None,scale_transformed_features=True)

datasets = [
       'iris','wine',
        'brest_cancer',
         'glass','seeds','ionosphere','sonar','blood_transfusion',
         'vehicle',
          'ecoli',
           'yeast','tic_tac_toe','heart','haberman','german_credit','diabets',
         'olivetti_faces_8','olivetti_faces_16','olivetti_faces_28',
          # 'olivetti_faces',
         'digits','fashion2000','fashion10000',
          # 'fashion','newsgroups'
            ]
if not os.path.exists('results'):
    os.makedirs('results')
with open('results/results_cv_fcm_transformer.py', 'a') as f:
    perform_cv_multiple(datasets,classifiers,tsf_scores, scoring,file=f,random_state=123,scale_transformed_features=True)


