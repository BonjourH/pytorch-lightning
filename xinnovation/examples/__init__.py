from .detector2D import *
from .detector4D import *

detector2D_module = []

detector4D_module = ['Sparse4DModule', 'Sparse4DDataModule']


__all__ = detector2D_module + detector4D_module