""" This is the script driver for the eMERLIN RASCIL pipeline

"""

__all__ = ["run_erp"]

from erp.functions.imaging_steps import *
from rascil.workflows.rsexecute.execution_support import rsexecute

def run_erp(erp_params, get_logger):
    """ Run the Imaging/Selfcalibration pipeline

    :param erp_params: Control parameters
    :return:
    """
    
    initialize_pipeline(erp_params, get_logger)
    
    bvis_list = ingest(erp_params)
    results = process(bvis_list, erp_params)
    bvis_list, results = rsexecute.compute(results, sync=True)
    stage(erp_params, bvis_list, results)
    
    finalize_pipeline(erp_params)
    
    return True
