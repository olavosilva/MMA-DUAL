"""mma-dual: MMA/GCMMA subproblems solved by the dual method (GPLv3, see mmadual.py)."""
from .mmadual import mmasub, gcmmasub, subsolv_dual

__all__ = ['mmasub', 'gcmmasub', 'subsolv_dual']
__version__ = '0.1.0'
