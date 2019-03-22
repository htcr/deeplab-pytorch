from .voc import VOC, VOCAug
from .cocostuff import CocoStuff10k, CocoStuff164k
from .atr import ATR
from .lip import LIP
from .cihp import CIHP

def get_dataset(name):
    return {
        "cocostuff10k": CocoStuff10k,
        "cocostuff164k": CocoStuff164k,
        "voc": VOC,
        "vocaug": VOCAug,
        "atr": ATR,
        "lip": LIP,
        "cihp": CIHP
    }[name]
