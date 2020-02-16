"""

"""

import numpy
import logging

log = logging.getLogger('logger')

from rascil.processing_components.visibility import create_blockvisibility_from_ms

def load_ms(msfile, selected_sources=None, selected_dds=None,
                data_column='DATA', **kwargs):
    """ Load the selected sources and data descriptors from the measurement set

    :param msfile:
    :param selected_sources:
    :param selected_dds:
    :param data_column:
    :return:
    """
    bvis_list = create_blockvisibility_from_ms(msfile, datacolumn=data_column,
                                               selected_dds=selected_dds,
                                               selected_sources=selected_sources)
    sources = numpy.unique([bv.source for bv in bvis_list])
    log.info(
        'Sources loaded {sources}'.format(sources=sources))
    return bvis_list

def list_ms(msfile, **kwargs):
    """ List the contents of the measurement set

    :param msfile:
    :return:
    """
    log.info('list_ms not yet implemented')
