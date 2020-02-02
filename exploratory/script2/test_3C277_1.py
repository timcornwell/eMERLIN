
import logging
import os

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams['image.cmap'] = 'rainbow'

import numpy

from rascil.processing_components import show_image, qa_image, export_image_to_fits, qa_gaintable, \
    create_blockvisibility_from_ms, list_ms, integrate_visibility_by_channel, \
    concatenate_blockvisibility_frequency, plot_uvcoverage, plot_visibility, convert_blockvisibility_to_stokesI, \
    convert_blockvisibility_to_visibility, convert_visibility_to_blockvisibility, weight_visibility, \
    create_image_from_visibility, advise_wide_field, create_calibration_controls, gaintable_plot, \
    deconvolve_cube, restore_cube, create_blockvisibility_from_uvfits

from rascil.processing_components.imaging.ng import invert_ng

cwd = os.getcwd()

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.FileHandler('%s/eMERLIN_test.log' % cwd))

logging.basicConfig(filename='%s/eMERLIN_test.log' % cwd,
                    filemode='w',
                    format='%(date)s %(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
log.info("Logging to %s/eMERLIN_test.log" % cwd)

# %% md

#### List the contents of the MeasurementSet: sources and data descriptors

# %%

bvis_list = create_blockvisibility_from_uvfits('../../data/3C277.1.MULTTB')

