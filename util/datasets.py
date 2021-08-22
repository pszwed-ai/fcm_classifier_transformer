from tensorflow.keras.datasets import fashion_mnist
# from scipy.misc import imresize
import sklearn.datasets
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

# from base.losses import SoftmaxCrossEntropy
from util.data import load_local_glass, load_local_seeds, load_local_ionosphere, load_local_sonar, \
  load_local_blood_transfusion, load_local_balance, load_local_vehicle, load_local_ecoli, load_local_yeast, \
  load_local_tic_tac_toe, load_local_heart, load_local_haberman, load_local_german_credit, load_local_diabets

from sklearn.pipeline import Pipeline
import re


def get_list():
  return ['iris','wine','brest_cancer',
          'glass','seeds','ionosphere','sonar','blood_transfusion',
          'vehicle','ecoli','yeast','tic_tac_toe','heart','haberman','german_credit','diabets',
          'olivetti_faces_8','olivetti_faces_16','olivetti_faces_28','olivetti_faces',
          'digits','fashion2000','fashion10000','fashion','newsgroups']
# // covertype ?


from PIL import Image
def imresize(arr,target_size):
  return np.array(Image.fromarray(arr).resize(target_size))

def load_dataset(name,max_size=None,random_state=None):
  if False:
    pass
  elif name == 'iris':
    (X, y) = sklearn.datasets.load_iris(return_X_y=True)
  elif name == 'digits':
    (X, y) = sklearn.datasets.load_digits(return_X_y=True)
  elif name== 'fashion' or name=='fashion2000' or name == 'fashion10000':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    X=np.concatenate((np.array(x_train),np.array(x_test)),axis=0);
    y=np.concatenate((np.array(y_train),np.array(y_test)),axis=0);
    X=X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    X=X/255
    if name == 'fashion2000':
      max_size=2000
    if name == 'fashion10000':
      max_size = 10000
      #    print(X.shape)
  elif name == 'wine':
    (X, y) = sklearn.datasets.load_wine(return_X_y=True)
  elif name == 'brest_cancer':
    (X, y) = sklearn.datasets.load_breast_cancer(return_X_y=True)
  elif name == 'olivetti_faces_8':
    ds = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = ds.images
    X =  np.array([imresize(ds.images[i],(8,8)) for i in range(ds.images.shape[0])])
    X = X.reshape(X.shape[0], -1)
    X = X / 255
    y = ds.target
  elif name == 'olivetti_faces_16':
    ds = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = ds.images
    X =  np.array([imresize(ds.images[i],(16,16)) for i in range(ds.images.shape[0])])
    X = X.reshape(X.shape[0], -1)
    X = X / 255
    y = ds.target
  elif name == 'olivetti_faces_28':
    ds = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = ds.images
    X =  np.array([imresize(ds.images[i],(28,28)) for i in range(ds.images.shape[0])])
    X = X.reshape(X.shape[0], -1)
    X = X / 255
    y = ds.target
  elif name == 'olivetti_faces':
    ds = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = ds.images
    # X =  np.array([imresize(ds.images[i],(28,28)) for i in range(ds.images.shape[0])])
    X = X.reshape(X.shape[0], -1)
    X = X / 255
    y = ds.target
  #  print(y[:25])
  #  if apply_one_hot:
  #    label_binarizer = sklearn.preprocessing.LabelBinarizer()
  #    label_binarizer.fit(range(max(y)+1))
  #    Y = label_binarizer.transform(y)
  elif name == 'glass':
    X,y = load_local_glass()
  elif name == 'seeds':
    X,y = load_local_seeds()
  elif name == 'ionosphere':
    X,y =load_local_ionosphere()
  elif name == 'sonar':
    X,y =load_local_sonar()
  elif name == 'blood_transfusion':
    X,y =load_local_blood_transfusion()
  elif name == 'balance':
    X,y =load_local_balance()
  elif name == 'vehicle':
    X,y =load_local_vehicle()
  elif name == 'newsgroups':
    X,y = load_newsgroups_dataset()


  elif name == 'ecoli':
    X,y = load_local_ecoli()
  elif name == 'yeast':
    X,y = load_local_yeast()
  elif name == 'tic_tac_toe':
    X,y = load_local_tic_tac_toe()
  elif name == 'heart':
    X,y = load_local_heart()
  elif name == 'haberman':
    X,y = load_local_haberman()
  elif name == 'german_credit':
    X,y = load_local_german_credit()
  elif name == 'diabets':
    X,y = load_local_diabets()

  if max_size is not None and X.shape[0] > max_size:
    X_train, X, y_train, y = train_test_split(X, y, test_size=max_size, random_state=random_state)
  scaler = MinMaxScaler()
  X=scaler.fit_transform(X)
  X=X.astype(float)
  return X, y





# def _get_best_params(dataset,max_size):
#   if max_size is not None:
#     dataset=dataset+str(max_size)
#   if dataset=='digits':
#     return {'random_state': 123, 'depth': 3, 'epochs': 250, 'batch_size': 20, 'activation_m': 0.5, 'activation': 'sigmoid'}
#   elif dataset=='fashion2000':
#     return {'random_state': 123, 'depth': 3, 'epochs': 150, 'batch_size': 50, 'activation_m': 0.8, 'activation': 'sigmoid'}
#   elif dataset=='fashion': #f1=0.538
#     return{'random_state': 123, 'depth': 3, 'epochs': 4, 'batch_size': 500, 'activation_m': 5, 'activation': 'sigmoid'}
#   elif dataset=='brest_cancer':
#     return{'random_state': 123, 'depth': 5, 'epochs': 100, 'batch_size': 20, 'activation_m': 1, 'activation': 'sigmoid'}
#   elif dataset=='wine':
#     return{'random_state': 123, 'depth': 3, 'epochs': 100, 'batch_size': 20, 'activation_m': 10, 'activation': 'sigmoid'}
#   elif dataset=='olivetti_faces': # resized images
#     return{'random_state': 123, 'depth': 2, 'epochs': 350, 'batch_size': 10, 'activation_m': 1, 'activation': 'sigmoid'}
#   elif dataset=='glass':
#     # return{'random_state': 123, 'depth': 3, 'epochs': 350, 'batch_size': 5, 'activation_m': 3.5, 'activation': 'sigmoid'}
#     # return {'random_state': 123, 'depth': 4, 'epochs': 650, 'batch_size': 5, 'activation_m': 3.5, 'activation': 'sigmoid'}
#     return {
#     'activation': 'sigmoid',
#     'activation_m': 1,
#     'depth': 2,
#     'epochs': 3300,
#     'training_loss': SoftmaxCrossEntropy(),
#     'batch_size': -1,
#     'random_state': 123
#   }
#
#   else:
#     return {}


def load_newsgroups_dataset():
  dataset = sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
  stopwords = """i,me,my,myself,we,our,ours,ourselves,you,your,yours,yourself,yourselves,he,him,his,himself,she,her,hers,
  herself,it,its,itself,they,them,their,theirs,themselves,what,which,who,whom,this,that,these,those,am,is,are,was,were,
  be,been,being,have,has,had,having,do,does,did,doing,a,an,the,and,but,if,or,because,as,until,while,of,at,by,for,with,
  about,against,between,into,through,during,before,after,above,below,to,from,up,down,in,out,on,off,over,under,again,
  further,then,once,here,there,when,where,why,how,all,any,both,each,few,more,most,other,some,such,no,nor,not,only,own,
  same,so,than,too,very,s,t,can,will,just,don,should,now"""
  pipeline = Pipeline([
    ('vect', HashingVectorizer(n_features=2**9,non_negative=True,stop_words=re.split(',\\s*',stopwords))),
    # ('vect', HashingVectorizer(n_features=2 ** 8, stop_words=re.split(',\\s*', stopwords))),
    ('tfidf', TfidfTransformer())
  ])
  X = pipeline.fit_transform(dataset.data, dataset.target)
  X=X.toarray()
  y = dataset.target
  return X, y

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
  dataset = 'olivetti_faces_8'

  X,y = load_dataset(dataset)
  print('labels={}'.format(np.max(y) + 1))
  print('y.shape={}'.format(y.shape))
  print('X.shape={}'.format(X.shape))
  print(X[:10,:])
  print(y[:10])
