"""

"""

__all__ = ['erp_path']

import os

def erp_path(path):
    """Converts a path that might be relative to the package into an
    absolute path::

        erp_path('data/3C277.1C.ms')
        '/Users/timcornwell/Code/eMERLIN_RASCIL_pipeline/data/3C277.1C.ms'

    :param path:
    :return: absolute path
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) + "/../../")
    erphome = os.getenv('ERP', project_root)
    return os.path.join(erphome, path)

