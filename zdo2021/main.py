import numpy as np
# moduly v lokálním adresáři musí být v pythonu 3 importovány s tečkou
from . import podpurne_funkce

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, img):
        mask1 = pavel_detector(img)
        mask2 = najdi_brouka(img)
        output = np.multiply(mask1, mask2)
        return output
