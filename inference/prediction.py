from typing import List, Dict
from pickle import load as pickle_load
from pathlib import Path

import numpy as np
from torch import Tensor, no_grad
from torch import load as torch_load
from torch.nn import Softmax
from skimage.transform import resize

from src.cnn_model import ConvNet, CLASSES


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_MODEL_PKL = REPO_ROOT.joinpath('models/log_model.pkl')
CNN_MODEL_STATE = REPO_ROOT.joinpath('models/cnn_model_state.pt')

log_model = None
if LOG_MODEL_PKL.is_file():
    # load the model which is an instance of sklearn.linear_model.SGDClassifier
    with LOG_MODEL_PKL.open('rb') as pkl:
        log_model = pickle_load(pkl)

cnn_model = None
if CNN_MODEL_STATE.is_file():
    # Create a CNN model and load the parameters
    cnn_model = ConvNet()
    cnn_model.load_state_dict(torch_load(CNN_MODEL_STATE))
    softmax = Softmax(dim=0)


def log_predict(data: List[List[float]]) -> Dict[str, float]:
    """Function that takes data of one image and returns predictions of the
    logistic regression model.

    Args:
        data: data of one image file as you would obtain it from
        skimage.io.imread(image_file, as_grey=True).to_list()

    Returns:
        dict: predicted probabilities for the classes
    """
    if not log_model:
        return {"boot": 0.33, "sandal": 0.33, "shoe": 0.33}
    classes = log_model.classes_
    features = np.array(data)
    features = resize(features, (102, 136))
    features = np.reshape(features, (-1))
    probs = log_model.predict_proba([features])[0]
    assert len(probs) == len(classes)
    return {classes[idx]: probs[idx] for idx in range(len(probs))}


def cnn_predict(data: List[List[float]]) -> Dict[str, float]:
    """Function that takes data of one image and returns predictions of the
    CNN model.

    Args:
        data: data of one image file as you would obtain it from
        skimage.io.imread(image_file, as_grey=True).to_list()

    Returns:
        dict: predicted probabilities for the classes
    """
    if not cnn_model:
        return {CLASSES[idx]: 0.33 for idx in range(len(CLASSES))}
    features = np.array(data)
    features = resize(features, (25, 34))
    features = Tensor(features)
    features = features[None, None, :, :]
    with no_grad():
        logits = cnn_model(features)[0]
    probs = softmax(logits)
    assert len(probs) == len(CLASSES)
    return {CLASSES[idx]: float(probs[idx]) for idx in range(len(CLASSES))}
