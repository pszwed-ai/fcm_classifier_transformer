from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureClassifier(BaseEstimator, ClassifierMixin):

    MIN_LOG_LIKELIHOOD = -1e100

    def __init__(self,
               n_comp=1,
               covariance_type='full',
               ll_agggregation_method='max',
               random_state=0):
        self.n_comp=n_comp
        self.covariance_type=covariance_type
        self.random_state=random_state
        self.ll_agggregation_method = ll_agggregation_method

        self.n_labels_ = None
        self.n_class_components_ = None
        self.models_ = None

    def _select_number_of_class_components(self,X,y):
        # self.n_class_components_ = np.array(
        #     [self.n_comp if np.count_nonzero(y==i) != 0 else 0 for i in np.arange(self.n_labels_)]
        #     )

        self.n_class_components_ = np.array(
            [self.n_comp if np.count_nonzero(y==i) > 5*self.n_comp else 1 if np.count_nonzero(y==i) > 1 else 0 for i in np.arange(self.n_labels_)]
            )


    def fit(self,X,y):
        self.n_labels_ = np.max(y) + 1
        self._select_number_of_class_components(X,y)
        self.models_=[
            GaussianMixture(self.n_class_components_[i],covariance_type=self.covariance_type, random_state=self.random_state)
            .fit(X[y==i])
            if self.n_class_components_[i] > 0 else None
            for i in np.arange(self.n_labels_)
            ]
        return self


    def _get_log_likelihood(self, X, mu, sigma, weight):
        """
            @return: X.shape[0] x num_components array of log likelihoods for each component
            Number of components calculated as mu.shape[0]
        """
        #    print(X.shape)
        mixture_pdf = []
        for i in range(mu.shape[0]):
            logpdf = multivariate_normal.logpdf(X, mean=mu[i, ...], cov=sigma[i, ...]);
            logpdf = logpdf.reshape(X.shape[0])
            logpdf = logpdf + np.log(weight[i])
            mixture_pdf.append(logpdf)
        m_pdf = np.stack(mixture_pdf, axis=-1)
        return m_pdf

    def _aggregate(self, p, method='max'):
        """
        Actually it can be any fuzzy aggregation method
        max corresponds to fuzzy OR
        """
        if method == 'max':
            return np.max(p, axis=1)
        elif method == 'mean':
            return np.mean(p, axis=1)
        elif method == 'sum':
            return np.sum(p, axis=1)

    def _predict_log_likelihood_for_models(self, X):
        models_ll=[]
        for i in range(len(self.models_)):
            m = self.models_[i]
            if m is not None:
                log_prob = self._get_log_likelihood(X, m.means_, m.covariances_, m.weights_)
                log_prob_aggr=self._aggregate(log_prob, self.ll_agggregation_method)
            else:
                log_prob_aggr = np.full(X.shape[0],GaussianMixtureClassifier.MIN_LOG_LIKELIHOOD)
            models_ll.append(log_prob_aggr)
        models_ll =  np.stack(models_ll,axis=-1)
#        print('models_ll.shape=',end='')
#        print(models_ll.shape)
        return models_ll

    def predict(self,X):
        log_prob = self._predict_log_likelihood_for_models(X)
        return np.argmax(log_prob,axis=1)


    def predict_log_proba(self,X):
        log_prob = self._predict_log_likelihood_for_models(X)
        raise ValueError('Not implemented')

    def predict_proba(self,X):
        return np.exp(self.predict_log_proba(X))
