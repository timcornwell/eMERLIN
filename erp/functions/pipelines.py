"""

"""

__all__ = ["run_erp_ical"]

from erp.functions.imaging_steps import *
from erp.functions.support import get_logger
from rascil.workflows.rsexecute.execution_support import rsexecute

def run_erp_ical(eMRP):
    """ Run the Imaging/Selfcalibration pipeline
    
    :param eMRP: Control parameters
    :return:
    """
    initialize_pipeline(eMRP, get_logger=get_logger)
    
    bvis_list = None
    if eMRP['input_steps']['list_ms'] > 0:
        list_ms(eMRP)
    if eMRP['input_steps']['load_ms'] > 0:
        bvis_list = load_ms(eMRP)
    if eMRP['input_steps']['flag'] > 0:
        bvis_list = flag(bvis_list, eMRP)
    if eMRP['input_steps']['average_channels'] > 0:
        bvis_list = average_channels(bvis_list, eMRP)
    if eMRP['input_steps']['plot_vis'] > 0:
        plot_vis(bvis_list, eMRP)
    if eMRP['input_steps']['convert_stokesI'] > 0:
        bvis_list = convert_stokesI(bvis_list, eMRP)
    if eMRP['input_steps']['get_advice'] > 0:
        advice = get_advice(bvis_list, eMRP)
    model_list = list()
    if eMRP['input_steps']['create_images'] > 0:
        model_list = create_images(bvis_list, eMRP)
    if eMRP['input_steps']['weight'] > 0:
        bvis_list = weight(bvis_list, model_list, eMRP)
    if eMRP['input_steps']['cip'] > 0:
        results = cip(bvis_list, model_list, eMRP)
        results = rsexecute.compute(results, sync=True)
        if eMRP['input_steps']['write_images'] > 0:
            write_images(eMRP, 'cip', results)
    if eMRP['input_steps']['ical'] > 0:
        results = ical(bvis_list, model_list, eMRP)
        bvis_list, results = rsexecute.compute(results, sync=True)
        if eMRP['input_steps']['write_images'] > 0:
            write_images(eMRP, 'ical', results[0:3])
        if eMRP['input_steps']['write_gaintables'] > 0:
            write_gaintables(eMRP, 'ical', results[3])
        if eMRP['input_steps']['write_ms'] > 0:
            bvis_list = apply_calibration(results[3], bvis_list, eMRP)
            write_ms(bvis_list, eMRP)

    finalize_pipeline(eMRP)
