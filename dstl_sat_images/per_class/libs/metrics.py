from keras import backend as K

from keras.callbacks import Callback
import numpy as np

smooth = 1e-12

# area masking metics
def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    #sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# binary class metics
def f1_score(y_true, y_pred):
    # https://stackoverflow.com/questions/43345909/when-using-mectrics-in-model-compile-in-keras-report-valueerror-unknown-metr
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # How many selected items are relevant?
    precision = c1 / (c2 + smooth)

    # How many relevant items are selected?
    recall = c1 / (c3 + smooth)

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall + smooth)

    return f1_score

def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))

    # How many selected items are relevant?
    return c1 / (c2 + smooth)

def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # How many relevant items are selected?
    return c1 / (c3  + smooth)

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        try:
            self.auc.append(sklm.roc_auc_score(targ, score))
        except:
            self.auc.append(np.nan)

        try:
            self.confusion.append(sklm.confusion_matrix(targ, predict))
        except:
            self.confusion.append(np.nan)

        try:
            self.precision.append(sklm.precision_score(targ, predict))
        except:
            self.precision.append(np.nan)

        try:
            self.recall.append(sklm.recall_score(targ, predict))
        except:
            self.recall.append(np.nan)

        try:
            self.f1s.append(sklm.f1_score(targ, predict))
        except:
            self.f1s.append(np.nan)

        try:
            self.kappa.append(sklm.cohen_kappa_score(targ, predict))
        except:
            self.kappa.append(np.nan)

        return