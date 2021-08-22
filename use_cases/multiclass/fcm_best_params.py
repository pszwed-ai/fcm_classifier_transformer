from sklearn.utils import Bunch
# from tensorflow.contrib.optimizer_v2.rmsprop import RMSPropOptimizer
#import tensorflow as tf


from base.losses import SoftmaxCrossEntropy, LogLoss

b = Bunch()

def get_best_params(dataset):
    return b[dataset].params

def get_param_grid(dataset):
    return b[dataset].param_grid

def get_metrics(dataset):
    return b[dataset].metrics

def get_cv_metrics(dataset):
    return b[dataset].cv_metrics


# ----------------------------------------------------------------------------------------------------------------------


b.iris=Bunch()
b.iris.params = {
    'activation':'sigmoid',
    'activation_m':3,
    'depth':4,
    'epochs':3000,
    # 'epochs':100,
    'training_loss': 'softmax',
    'optimizer':'rmsprop',
    'learning_rate':0.0005,
    'batch_size':-1,
    'random_state':123
}
b.iris.metrics={
    'accuracy':0.9778,
    'precision_macro':0.9792,
    'recall_macro':0.9778,
    'f1_macro':0.9778,
}

b.iris.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':29.32547721862793,
    'fit_time_std':0.1907267836928109,
    'score_time_mean':0.017001008987426756,
    'score_time_std':0.0020001649877881907,
    'accuracy_mean':0.9666666666666668,
    'accuracy_std':0.02108185106778919,
    'precision_micro_mean':0.9666666666666668,
    'precision_micro_std':0.02108185106778919,
    'recall_micro_mean':0.9666666666666668,
    'recall_micro_std':0.02108185106778919,
    'f1_micro_mean':0.9666666666666668,
    'f1_micro_std':0.02108185106778919,
    'precision_macro_mean':0.9707070707070707,
    'precision_macro_std':0.017611712903194614,
    'recall_macro_mean':0.9666666666666668,
    'recall_macro_std':0.021081851067789228,
    'f1_macro_mean':0.9664818612187034,
    'f1_macro_std':0.021295160208553578,
}

# ----------------------------------------------------------------------------------------------------------------------


b.glass=Bunch()

# b.glass.params = {
#     'activation':'sigmoid',
#     'activation_m':1,
#     'depth':2,
#     'epochs':4000,
#     'training_loss': 'softmax',
#     'optimizer':'rmsprop',
#     'learning_rate': 0.05,
#     'batch_size':-1,
#     'random_state':123
# }

b.glass.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':2,
    'epochs':3300,
    'training_loss': 'softmax',
    'optimizer':'rmsprop',
    'learning_rate': 0.125892541179417,
    'batch_size':-1,
    'random_state':123
}
b.glass.metrics={
    'accuracy':0.7077,
    'precision_macro':0.7247,
    'recall_macro':0.5623,
    'f1_macro':0.5972,
}
b.glass.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':17.070890045166017,
    'fit_time_std':0.2003617965262544,
    'score_time_mean':0.007181406021118164,
    'score_time_std':0.00039970895762655626,
    'accuracy_mean':0.5539546629081512,
    'accuracy_std':0.13214066679807132,
    'precision_micro_mean':0.5539546629081512,
    'precision_micro_std':0.13214066679807132,
    'recall_micro_mean':0.5539546629081512,
    'recall_micro_std':0.13214066679807132,
    'f1_micro_mean':0.5539546629081512,
    'f1_micro_std':0.13214066679807132,
    'precision_macro_mean':0.5299480611980613,
    'precision_macro_std':0.14979604918430775,
    'recall_macro_mean':0.5179365079365079,
    'recall_macro_std':0.1535591029055588,
    'f1_macro_mean':0.4843886370683396,
    'f1_macro_std':0.15925202834117544,
}


# ----------------------------------------------------------------------------------------------------------------------

b.seeds=Bunch()

b.seeds.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':2,
    'epochs':3300,
    'training_loss': 'softmax',
    'optimizer':'rmsprop',
    'learning_rate': 0.08,
    'batch_size':-1,
    'random_state':123
}

b.seeds.metrics={
	'accuracy':0.9683,
	'precision_macro':0.9710,
	'recall_macro':0.9683,
	'f1_macro':0.9682,
}

b.seeds.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':27.8263916015625,
    'fit_time_std':0.65577222028762,
    'score_time_mean':0.01240072250366211,
    'score_time_std':0.0008000850748536953,
    'accuracy_mean':0.938095238095238,
    'accuracy_std':0.05345224838248485,
    'precision_micro_mean':0.938095238095238,
    'precision_micro_std':0.05345224838248485,
    'recall_micro_mean':0.938095238095238,
    'recall_micro_std':0.05345224838248485,
    'f1_micro_mean':0.938095238095238,
    'f1_micro_std':0.05345224838248485,
    'precision_macro_mean':0.9466457811194655,
    'precision_macro_std':0.041128740148713726,
    'recall_macro_mean':0.9380952380952381,
    'recall_macro_std':0.05345224838248489,
    'f1_macro_mean':0.9389697437513529,
    'f1_macro_std':0.05159876675026719,
}

# ----------------------------------------------------------------------------------------------------------------------

b.wine = Bunch()
# b.wine.params = {
#     'activation':'sigmoid',
#     'activation_m':10,
#     'depth':3,
#     'epochs':200,
#     'training_loss': 'softmax',
#     'optimizer':'rmsprop',
#     'learning_rate': 0.001,
#     'batch_size':20,
#     'random_state':123
# }


b.wine.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':4,
    'epochs':3000,
    'training_loss': 'softmax',
    'optimizer':'rmsprop',
    'learning_rate': 0.001,
    'batch_size':-1,
    'random_state':123
}

b.wine.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':33.80913372039795,
    'fit_time_std':0.4506471728537017,
    'score_time_mean':0.01780104637145996,
    'score_time_std':0.0004000664041494148,
    'accuracy_mean':0.9718689277512806,
    'accuracy_std':0.025285834694918113,
    'precision_micro_mean':0.9718689277512806,
    'precision_micro_std':0.025285834694918113,
    'recall_micro_mean':0.9718689277512806,
    'recall_micro_std':0.025285834694918113,
    'f1_micro_mean':0.9718689277512806,
    'f1_micro_std':0.025285834694918113,
    'precision_macro_mean':0.9706060606060607,
    'precision_macro_std':0.026150498788338448,
    'recall_macro_mean':0.9768253968253969,
    'recall_macro_std':0.02061293089618782,
    'f1_macro_mean':0.9719982224177496,
    'f1_macro_std':0.025022691883190177,
}

# ----------------------------------------------------------------------------------------------------------------------

b.brest_cancer = Bunch()
b.brest_cancer.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':5,
    'epochs':1000,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.03,
    'batch_size':-1,
    'random_state':123
}

b.brest_cancer.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':20.608178663253785,
    'fit_time_std':0.5707155030428894,
    'score_time_mean':0.03040180206298828,
    'score_time_std':0.0037739052643225106,
    'accuracy_mean':0.9666794921123507,
    'accuracy_std':0.010005623307055985,
    'precision_micro_mean':0.9666794921123507,
    'precision_micro_std':0.010005623307055985,
    'recall_micro_mean':0.9666794921123507,
    'recall_micro_std':0.010005623307055985,
    'f1_micro_mean':0.9666794921123507,
    'f1_micro_std':0.010005623307055985,
    'precision_macro_mean':0.9651215772157947,
    'precision_macro_std':0.009949925898041728,
    'recall_macro_mean':0.9638890838571481,
    'recall_macro_std':0.012497649412106215,
    'f1_macro_mean':0.9643034019520563,
    'f1_macro_std':0.010722011342285771,
}

# ----------------------------------------------------------------------------------------------------------------------


b.balance=Bunch()



b.balance.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':4,
    'epochs':2000,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.05,
    'batch_size':-1,
    'random_state':123
}

b.balance.metrics={
	'accuracy':0.9840,
	'precision_macro':0.9570,
	'recall_macro':0.9884,
	'f1_macro':0.9714,
}


b.balance.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':26.59292106628418,
    'fit_time_std':0.7291136116593376,
    'score_time_mean':0.019201135635375975,
    'score_time_std':0.000400018706226792,
    'accuracy_mean':0.9330379778452162,
    'accuracy_std':0.04944957850694291,
    'precision_micro_mean':0.9330379778452162,
    'precision_micro_std':0.04944957850694291,
    'recall_micro_mean':0.9330379778452162,
    'recall_micro_std':0.04944957850694291,
    'f1_micro_mean':0.9330379778452162,
    'f1_micro_std':0.04944957850694291,
    'precision_macro_mean':0.8757261674736931,
    'precision_macro_std':0.08111816480602,
    'recall_macro_mean':0.879138267123748,
    'recall_macro_std':0.04844378957621146,
    'f1_macro_mean':0.8710163780016924,
    'f1_macro_std':0.06785084555070926,
}

# ----------------------------------------------------------------------------------------------------------------------


b.vehicle=Bunch()


b.vehicle.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':2000,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.06,
    'batch_size':-1,
    'random_state':123
}

b.vehicle.metrics={
	'accuracy':0.7677,
	'precision_macro':0.7692,
	'recall_macro':0.7708,
	'f1_macro':0.7663,
}

b.vehicle.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':31.748215913772583,
    'fit_time_std':1.579038666999039,
    'score_time_mean':0.018600940704345703,
    'score_time_std':0.0008000135475042733,
    'accuracy_mean':0.7636276807953697,
    'accuracy_std':0.01916769401532904,
    'precision_micro_mean':0.7636276807953697,
    'precision_micro_std':0.01916769401532904,
    'recall_micro_mean':0.7636276807953697,
    'recall_micro_std':0.01916769401532904,
    'f1_micro_mean':0.7636276807953697,
    'f1_micro_std':0.01916769401532904,
    'precision_macro_mean':0.7643501351557782,
    'precision_macro_std':0.025060358880810114,
    'recall_macro_mean':0.7659754489696349,
    'recall_macro_std':0.017928166635004694,
    'f1_macro_mean':0.7543099634277891,
    'f1_macro_std':0.022421372636529623,
}

# ----------------------------------------------------------------------------------------------------------------------


b.ionosphere=Bunch()

b.ionosphere.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':2,
    'epochs':3300,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.004,
    'batch_size':-1,
    'random_state':123
}

b.ionosphere.metrics={
	'accuracy':0.9340,
	'precision_macro':0.9438,
	'recall_macro':0.9137,
	'f1_macro':0.9259,
}

b.ionosphere.cv_metrics={
    'classifier':'FcmBinaryClassifier',
    'fit_time_mean':40.52891540527344,
    'fit_time_std':4.867603549592746,
    'score_time_mean':0.016201019287109375,
    'score_time_std':0.00849522693135438,
    'accuracy_mean':0.8974647887323945,
    'accuracy_std':0.04072299406858573,
    'precision_micro_mean':0.8974647887323945,
    'precision_micro_std':0.04072299406858573,
    'recall_micro_mean':0.8974647887323945,
    'recall_micro_std':0.04072299406858573,
    'f1_micro_mean':0.8974647887323945,
    'f1_micro_std':0.04072299406858573,
    'precision_macro_mean':0.9104740220097363,
    'precision_macro_std':0.044203493892436344,
    'recall_macro_mean':0.8675897435897436,
    'recall_macro_std':0.04802093682231141,
    'f1_macro_mean':0.882607665968127,
    'f1_macro_std':0.04734941152859012,
}

#Apply Naive Bayes afterwards

# ----------------------------------------------------------------------------------------------------------------------


b.sonar=Bunch()


b.sonar.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':2,
    'epochs':500,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.008,
    'batch_size':-1,
    'random_state':123
}

b.sonar.metrics={
	'accuracy':0.8889,
	'precision_macro':0.8940,
	'recall_macro':0.8844,
	'f1_macro':0.8871,
}
b.sonar.cv_metrics={
    'classifier':'FcmBinaryClassifier',
    'fit_time_mean':4.779473352432251,
    'fit_time_std':0.02391294833744853,
    'score_time_mean':0.013400745391845704,
    'score_time_std':0.0004898235152673564,
    'accuracy_mean':0.6788698916889502,
    'accuracy_std':0.08248735907218331,
    'precision_micro_mean':0.6788698916889502,
    'precision_micro_std':0.08248735907218331,
    'recall_micro_mean':0.6788698916889502,
    'recall_micro_std':0.08248735907218331,
    'f1_micro_mean':0.6788698916889502,
    'f1_micro_std':0.08248735907218334,
    'precision_macro_mean':0.7040138653616914,
    'precision_macro_std':0.09439913030710476,
    'recall_macro_mean':0.6762648221343873,
    'recall_macro_std':0.082683201345125,
    'f1_macro_mean':0.6683537349017226,
    'f1_macro_std':0.08167158976299442,
}

# ----------------------------------------------------------------------------------------------------------------------


b.blood_transfusion=Bunch()

b.blood_transfusion.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':3300,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.004,
    'batch_size':-1,
    'random_state':123
}

b.blood_transfusion.metrics={
	'accuracy':0.8089,
	'precision_macro':0.7700,
	'recall_macro':0.6462,
	'f1_macro':0.6699,
}

b.blood_transfusion.cv_metrics={
    'classifier':'FcmBinaryClassifier',
    'fit_time_mean':55.833793544769286,
    'fit_time_std':0.3013154655360862,
    'score_time_mean':0.02520136833190918,
    'score_time_std':0.00515381740441579,
    'accuracy_mean':0.7501208053691275,
    'accuracy_std':0.06037998678683345,
    'precision_micro_mean':0.7501208053691275,
    'precision_micro_std':0.06037998678683345,
    'recall_micro_mean':0.7501208053691275,
    'recall_micro_std':0.06037998678683345,
    'f1_micro_mean':0.7501208053691275,
    'f1_micro_std':0.06037998678683345,
    'precision_macro_mean':0.5199579435619703,
    'precision_macro_std':0.17133296582584118,
    'recall_macro_mean':0.6060651629072682,
    'recall_macro_std':0.1299089535171395,
    'f1_macro_mean':0.5379121264307625,
    'f1_macro_std':0.13578138150410626,
}


# ----------------------------------------------------------------------------------------------------------------------

b.digits = Bunch()
b.digits.params =  {
    'activation':'sigmoid',
    'activation_m':0.5,
    'depth':3,
    'epochs':120,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.01,
    'batch_size':20,
    'random_state':123
}

b.digits.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':120.29928073883056,
    'fit_time_std':0.7485343492719285,
    'score_time_mean':0.0470026969909668,
    'score_time_std':0.003633442076662901,
    'accuracy_mean':0.9438388423931086,
    'accuracy_std':0.02290345400444846,
    'precision_micro_mean':0.9438388423931086,
    'precision_micro_std':0.02290345400444846,
    'recall_micro_mean':0.9438388423931086,
    'recall_micro_std':0.02290345400444846,
    'f1_micro_mean':0.9438388423931086,
    'f1_micro_std':0.02290345400444846,
    'precision_macro_mean':0.9461075318462981,
    'precision_macro_std':0.02231631843658192,
    'recall_macro_mean':0.9438646377469908,
    'recall_macro_std':0.02260909688099666,
    'f1_macro_mean':0.9433563590751964,
    'f1_macro_std':0.02304166532015029,
}





# ----------------------------------------------------------------------------------------------------------------------

b.fashion2000=Bunch()

# 'activation': 'sigmoid',
# 'activation_m': 1,
# 'depth': 3,
# 'epochs': 4000,
# 'training_loss': 'softmax',
# 'optimizer': 'rmsprop',
# 'learning_rate': 0.0006,
# 'batch_size': -1,
# 'random_state': 123

b.fashion2000.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':600,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.005,
    'batch_size':1000,
    'random_state':123
}

b.fashion2000.metrics={
	'accuracy':0.8000,
	'precision_macro':0.8045,
	'recall_macro':0.8058,
	'f1_macro':0.8028,
}

b.fashion2000.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':175.1926654815674,
    'fit_time_std':2.8352386873492277,
    'score_time_mean':0.33570232391357424,
    'score_time_std':0.017623418661392314,
    'accuracy_mean':0.7766724882941327,
    'accuracy_std':0.04119888779582065,
    'precision_micro_mean':0.7766724882941327,
    'precision_micro_std':0.04119888779582065,
    'recall_micro_mean':0.7766724882941327,
    'recall_micro_std':0.04119888779582065,
    'f1_micro_mean':0.7766724882941327,
    'f1_micro_std':0.041198887795820656,
    'precision_macro_mean':0.7816490731639691,
    'precision_macro_std':0.052094379646047004,
    'recall_macro_mean':0.7780776526074911,
    'recall_macro_std':0.04104992433380724,
    'f1_macro_mean':0.7627913627761471,
    'f1_macro_std':0.057964991536185784,
}




b.fashion10000=Bunch()


b.fashion10000.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':600,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.005,
    'batch_size':1000,
    'random_state':123
}

b.fashion10000.metrics={
	'accuracy':0.8640,
	'precision_macro':0.8695,
	'recall_macro':0.8647,
	'f1_macro':0.8626,
}


b.fashion10000.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':1268.4343692779541,
    'fit_time_std':81.91302061237941,
    'score_time_mean':1.3807063579559327,
    'score_time_std':0.008978045023882142,
    'accuracy_mean':0.8550201065223562,
    'accuracy_std':0.02951852371711806,
    'precision_micro_mean':0.8550201065223562,
    'precision_micro_std':0.02951852371711806,
    'recall_micro_mean':0.8550201065223562,
    'recall_micro_std':0.02951852371711806,
    'f1_micro_mean':0.8550201065223562,
    'f1_micro_std':0.02951852371711806,
    'precision_macro_mean':0.8592812485472591,
    'precision_macro_std':0.02096120207423183,
    'recall_macro_mean':0.8548903920852226,
    'recall_macro_std':0.030168583254133897,
    'f1_macro_mean':0.8467524699042779,
    'f1_macro_std':0.04369406663535312,
}


b.fashion = Bunch()

b.fashion.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':600,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.005,
    'batch_size':1000,
    'random_state':123
}

b.fashion.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':9544.846431589127,
    'fit_time_std':20.858982551293952,
    'score_time_mean':9.255241537094117,
    'score_time_std':0.098624059103864,
    'accuracy_mean':0.8894571428571428,
    'accuracy_std':0.005058958511837201,
    'precision_micro_mean':0.8894571428571428,
    'precision_micro_std':0.005058958511837201,
    'recall_micro_mean':0.8894571428571428,
    'recall_micro_std':0.005058958511837201,
    'f1_micro_mean':0.8894571428571428,
    'f1_micro_std':0.005058958511837201,
    'precision_macro_mean':0.8903331796049819,
    'precision_macro_std':0.004527364253325881,
    'recall_macro_mean':0.8894571428571428,
    'recall_macro_std':0.005058958511837168,
    'f1_macro_mean':0.8889258146220591,
    'f1_macro_std':0.005008914056832853,
}


# ----------------------------------------------------------------------------------------------------------------------


b.olivetti_faces_8 = Bunch()

b.olivetti_faces_8.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.00045,
    'batch_size':-1,
    'random_state':123
}


b.olivetti_faces_8.metrics={
	'accuracy':0.8417,
	'precision_macro':0.8429,
	'recall_macro':0.8417,
	'f1_macro':0.8276,
}

b.olivetti_faces_8.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':75.7272979259491,
    'fit_time_std':0.7542442903651159,
    'score_time_mean':0.020337915420532225,
    'score_time_std':0.004351608158587888,
    'accuracy_mean':0.86,
    'accuracy_std':0.03657184709581949,
    'precision_micro_mean':0.86,
    'precision_micro_std':0.03657184709581949,
    'recall_micro_mean':0.86,
    'recall_micro_std':0.03657184709581949,
    'f1_micro_mean':0.86,
    'f1_micro_std':0.03657184709581949,
    'precision_macro_mean':0.8615,
    'precision_macro_std':0.047596101614406285,
    'recall_macro_mean':0.86,
    'recall_macro_std':0.03657184709581949,
    'f1_macro_mean':0.8408809523809524,
    'f1_macro_std':0.04444324261413604,
}

# ----------------------------------------------------------------------------------------------------------------------


b.olivetti_faces_16 = Bunch()


b.olivetti_faces_16.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.00045,
    'batch_size':-1,
    'random_state':123
}

b.olivetti_faces_16.metrics={
	'accuracy':0.9250,
	'precision_macro':0.9000,
	'recall_macro':0.9250,
	'f1_macro':0.9052,
}

b.olivetti_faces_16.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':141.5362645149231,
    'fit_time_std':6.851976179717227,
    'score_time_mean':0.029717445373535156,
    'score_time_std':0.0013220589989951434,
    'accuracy_mean':0.9574999999999999,
    'accuracy_std':0.012747548783981964,
    'precision_micro_mean':0.9574999999999999,
    'precision_micro_std':0.012747548783981964,
    'recall_micro_mean':0.9574999999999999,
    'recall_micro_std':0.012747548783981964,
    'f1_micro_mean':0.9575000000000001,
    'f1_micro_std':0.012747548783981969,
    'precision_macro_mean':0.9475,
    'precision_macro_std':0.019999999999999966,
    'recall_macro_mean':0.9574999999999999,
    'recall_macro_std':0.012747548783981964,
    'f1_macro_mean':0.9475,
    'f1_macro_std':0.016482313753434848,
}

# ----------------------------------------------------------------------------------------------------------------------


b.olivetti_faces_28 = Bunch()


b.olivetti_faces_28.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':4000,
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.006,
    'batch_size':-1,
    'random_state':123
}


b.olivetti_faces_28.metrics={
	'accuracy':0.8583,
	'precision_macro':0.8392,
	'recall_macro':0.8583,
	'f1_macro':0.8352,
}

b.olivetti_faces_28.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':495.8678561687469,
    'fit_time_std':3.6517185029141728,
    'score_time_mean':0.08636960983276368,
    'score_time_std':0.0014928269836198581,
    'accuracy_mean':0.8875,
    'accuracy_std':0.057554322166106696,
    'precision_micro_mean':0.8875,
    'precision_micro_std':0.057554322166106696,
    'recall_micro_mean':0.8875,
    'recall_micro_std':0.057554322166106696,
    'f1_micro_mean':0.8875,
    'f1_micro_std':0.05755432216610672,
    'precision_macro_mean':0.8520833333333332,
    'precision_macro_std':0.07448899993362186,
    'recall_macro_mean':0.8875,
    'recall_macro_std':0.057554322166106696,
    'f1_macro_mean':0.8586666666666668,
    'f1_macro_std':0.07005077523561827,
}

# ----------------------------------------------------------------------------------------------------------------------


b.olivetti_faces = Bunch()


b.olivetti_faces.params = {
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':4000, #4000
    'training_loss': 'softmax',
    'optimizer': 'rmsprop',
    'learning_rate': 0.0006,
    'batch_size':-1,
    'random_state':123
}

b.olivetti_faces.metrics={
        'accuracy':0.5750,
        'precision_macro':0.4745,
        'recall_macro':0.5750,
        'f1_macro':0.4937,
}

b.olivetti_faces.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':6279.611664009094,
    'fit_time_std':160.89494850611035,
    'score_time_mean':1.0807089805603027,
    'score_time_std':0.004211866967486432,
    'accuracy_mean':0.0225,
    'accuracy_std':0.005,
    'precision_micro_mean':0.0225,
    'precision_micro_std':0.005,
    'recall_micro_mean':0.0225,
    'recall_micro_std':0.005,
    'f1_micro_mean':0.022500000000000003,
    'f1_micro_std':0.005000000000000001,
    'precision_macro_mean':0.0006461939967243727,
    'precision_macro_std':0.00016144654699991838,
    'recall_macro_mean':0.0225,
    'recall_macro_std':0.005,
    'f1_macro_mean':0.0012558146460585485,
    'f1_macro_std':0.0003116607700159801,
}

b.newsgroups = Bunch()

b.newsgroups.params={
    'activation':'sigmoid',
    'activation_m':0.5,
    'depth':2,
    'epochs':300,
    'training_loss': 'logloss',
    'optimizer': 'rmsprop',
    'learning_rate': 0.002,
    'batch_size':500,
    'random_state':123
}


b.newsgroups.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
    'fit_time_mean':1367.5570197582244,
    'fit_time_std':82.34852618892583,
    'score_time_mean':2.609349250793457,
    'score_time_std':0.09683816001791416,
    'accuracy_mean':0.44151745614489163,
    'accuracy_std':0.01019667987989709,
    'precision_micro_mean':0.44151745614489163,
    'precision_micro_std':0.01019667987989709,
    'recall_micro_mean':0.44151745614489163,
    'recall_micro_std':0.01019667987989709,
    'f1_micro_mean':0.44151745614489163,
    'f1_micro_std':0.010196679879897072,
    'precision_macro_mean':0.4425973270016558,
    'precision_macro_std':0.007198278840717819,
    'recall_macro_mean':0.43327378428177,
    'recall_macro_std':0.009941949819476983,
    'f1_macro_mean':0.4320011547243331,
    'f1_macro_std':0.009765332447340033,
}

# for MSE, m=0.5
# train_accuracy=0.6078684050939963
# test_accuracy=0.4357976653696498
# train_precision_macro=0.6206413627060401
# test_precision_macro=0.4391431656040374
# train_recall_macro=0.6021116911311355
# test_recall_macro=0.42824702627407046
# train_f1_micro=0.6078684050939963
# test_f1_micro=0.4357976653696498
# train_precision_micro=0.6078684050939963
# test_precision_micro=0.4357976653696498
# train_recall_micro=0.6078684050939963
# test_recall_micro=0.4357976653696498
# train_f1_macro=0.6043721460625966
# test_f1_macro=0.4285806662029497
# fit_time=6915.1967279
# score_time=87.93776369999978
# metrics={
# 	'accuracy':0.4358,
# 	'precision_macro':0.4391,
# 	'recall_macro':0.4282,
# 	'f1_macro':0.4286,
# }


# ----------------------------------------------------------------------------------------------------------------------

b.diabets = Bunch()
b.diabets.params={
    'activation':'sigmoid',
    'activation_m':1,
    'depth':3,
    'epochs':3000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.diabets.metrics={
	'accuracy':0.7532,
	'precision_macro':0.7318,
	'recall_macro':0.7078,
	'f1_macro':0.7156,
}

b.diabets.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}


# ----------------------------------------------------------------------------------------------------------------------

b.ecoli = Bunch()
b.ecoli.params={
    'activation':'sigmoid',
    'activation_m':2,
    'depth':2,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.ecoli.metrics={
	'accuracy':0.7624,
	'precision_macro':0.5345,
	'recall_macro':0.5149,
	'f1_macro':0.5133,
}

b.ecoli.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}

# ----------------------------------------------------------------------------------------------------------------------

b.german_credit = Bunch()
b.german_credit.params={
    'activation':'sigmoid',
    'activation_m':1,
    'depth':2,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.german_credit.metrics={
	'accuracy':0.6967,
	'precision_macro':0.6328,
	'recall_macro':0.6246,
	'f1_macro':0.6279,
}

b.german_credit.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}

# ----------------------------------------------------------------------------------------------------------------------

b.haberman = Bunch()
b.haberman.params={
    'activation':'sigmoid',
    'activation_m':2,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.haberman.metrics={
	'accuracy':0.7500,
	'precision_macro':0.6616,
	'recall_macro':0.5882,
	'f1_macro':0.5942,
}

b.haberman.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}

# ----------------------------------------------------------------------------------------------------------------------

b.heart = Bunch()
b.heart.params={
    'activation':'sigmoid',
    'activation_m':2,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.heart.metrics={
	'accuracy':0.8148,
	'precision_macro':0.8153,
	'recall_macro':0.8083,
	'f1_macro':0.8107,
}

b.heart.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}



# ----------------------------------------------------------------------------------------------------------------------

b.tic_tac_toe = Bunch()
b.tic_tac_toe.params={
    'activation':'sigmoid',
    'activation_m':2,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    'optimizer': 'adam',
    # 'optimizer': 'rmsprop',
    # 'learning_rate': 0.0001,
    'batch_size':-1,
    'random_state':123
}

b.tic_tac_toe.metrics={
	'accuracy':0.8854,
	'precision_macro':0.8742,
	'recall_macro':0.8724,
	'f1_macro':0.8733,
}

b.tic_tac_toe.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}


# ----------------------------------------------------------------------------------------------------------------------

b.yeast = Bunch()




b.yeast.params={
    'activation':'sigmoid',
    'activation_m':2.8,
    'depth':3,
    'epochs':5000,
    'training_loss': 'softmax',
    # 'optimizer': 'adam',
    'optimizer': 'rmsprop',
    'learning_rate': 0.032,
    'batch_size':-1,
    'random_state':123
}

b.yeast.metrics={
	'accuracy':0.5179,
	'precision_macro':0.2723,
	'recall_macro':0.3003,
	'f1_macro':0.2815,
}

b.yeast.cv_metrics={
    'classifier':'FcmMulticlassClassifier',
}





if __name__ == '__main__':
    print(b['iris'])
    # for k in b:
    #     print(k)
    # print(b.__dict__)
    # print()

    print(get_cv_metrics('sonar')['classifier'])