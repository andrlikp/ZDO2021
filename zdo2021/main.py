import numpy as np
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, img):
        output = np.logical_and(pavel_detector(img), najdi_brouka(img))
        return output
