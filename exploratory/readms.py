import logging
import sys

import matplotlib.pyplot as plt
import numpy

from processing_components.image.operations import show_image, qa_image, export_image_to_fits
from processing_components.visibility.base import create_blockvisibility_from_ms, list_ms

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
# log.addHandler(logging.StreamHandler(sys.stderr))

# %%

from matplotlib import pylab
pylab.rcParams['figure.figsize'] = (12.0, 12.0)
pylab.rcParams['image.cmap'] = 'rainbow'

# %%
print(list_ms('../../data/3C277.1_avg.ms'))
# %%
selected_sources = ['1252+5634']
bvis_list = create_blockvisibility_from_ms('../../data/3C277.1_avg.ms', datacolumn='CORRECTED_DATA',
                                           selected_sources=['1252+5634'])
sources = numpy.unique([bv.source for bv in bvis_list])
print(sources)

# %%
from processing_components.visibility.operations import integrate_visibility_by_channel
avis_list = [integrate_visibility_by_channel(bvis) for bvis in bvis_list]
# %%
from processing_components.visibility.operations import concatenate_blockvisibility_frequency
blockvis = [concatenate_blockvisibility_frequency(avis_list[isource * 4:(isource * 4 + 4)])
            for isource, source in enumerate(sources)]

# %%
from processing_components.simulation.simulation_helpers import plot_uvcoverage, plot_visibility

for svis in blockvis:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_uvcoverage([svis], ax=ax, title='UV Coverage {source:s}'.format(source=svis.source))
    plt.tight_layout()
    plt.show(block=False)

for svis in blockvis:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_visibility([svis], ax=ax, title='Visibility amplitude {source:s}'.format(source=svis.source))
    plt.tight_layout()
    plt.show(block=False)

# %%
from processing_components.imaging.ng import invert_ng
from processing_components.visibility.operations import convert_blockvisibility_to_stokesI
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from processing_components.imaging.weighting import weight_visibility
from processing_components.imaging.base import create_image_from_visibility
from workflows.serial.pipelines.pipeline_serial import continuum_imaging_list_serial_workflow, \
    ical_list_serial_workflow
from processing_components.imaging.base import advise_wide_field
from processing_components.calibration.calibration_control import create_calibration_controls

advice = advise_wide_field(avis_list[0], verbose=False)
for svis in blockvis:
    frequency = [numpy.mean(svis.frequency)]
    channel_bandwidth = [numpy.sum(svis.channel_bandwidth)]
    ivis = convert_blockvisibility_to_stokesI(svis)
    model = create_image_from_visibility(ivis, npixel=1024, cellsize=advice['cellsize'] / 3.0, nchan=1,
                                         frequency=frequency, channel_bandwidth=channel_bandwidth)
    cvis = convert_blockvisibility_to_visibility(ivis)
    cvis = weight_visibility(cvis, model)
    ivis = convert_visibility_to_blockvisibility(cvis)
    
    mode = "ical"
    if mode == "ical":
        controls = create_calibration_controls()
        controls['T']['first_selfcal'] = 1
        controls['T']['phase_only'] = True
        controls['T']['timescale'] = 'auto'
        
        deconvolved, residual, restored, gt_list = ical_list_serial_workflow([ivis], [model],
                                                                             context='ng',
                                                                             nmajor=5,
                                                                             niter=1000, algorithm='msclean',
                                                                             scales=[0, 3, 10], gain=0.1,
                                                                             fractional_threshold=0.5,
                                                                             threshold=0.003,
                                                                             window_shape='quarter',
                                                                             do_wstacking=False,
                                                                             global_solution=False,
                                                                             calibration_context='T',
                                                                             do_selfcal=True,
                                                                             controls=controls)
        deconvolved = deconvolved[0]
        residual = residual[0][0]
        restored = restored[0]
    
    elif mode == "cip":
        deconvolved, residual, restored = continuum_imaging_list_serial_workflow([ivis], [model], context='ng',
                                                                                 nmajor=5,
                                                                                 niter=1000, algorithm='msclean',
                                                                                 scales=[0, 3, 10], gain=0.1,
                                                                                 fractional_threshold=0.5,
                                                                                 threshold=0.003,
                                                                                 window_shape='quarter',
                                                                                 do_wstacking=False)
        
        deconvolved = deconvolved[0]
        residual = residual[0][0]
        restored = restored[0]
    
    else:
        mode = "invert"
        dirty, sumwt = invert_ng(ivis, model, do_wstacking=False)
        print(sumwt)
        plt.clf()
        show_image(dirty, title=svis.source + " Dirty image")
        plt.show(block=False)
        
        psf, sumwt = invert_ng(ivis, model, do_wstacking=False, dopsf=True)
        plt.clf()
        show_image(psf, title=svis.source + " PSF")
        plt.show(block=False)
        
        from processing_components.image.deconvolution import deconvolve_cube, restore_cube
        
        deconvolved, residual = deconvolve_cube(dirty, psf, niter=1000, algorithm='msclean',
                                                fractional_threshold=0.5,
                                                scales=[0, 3, 10], gain=0.1, threshold=0.003,
                                                window_shape='quarter')
        restored = restore_cube(deconvolved, psf, residual)
        print(qa_image(deconvolved))
    
    plt.clf()
    print(qa_image(residual, context='Residual image'))
    show_image(residual, title=svis.source + " residual image")
    plt.tight_layout()
    plt.show(block=False)
    
    plt.clf()
    print(qa_image(restored, context='Restored image'))
    show_image(restored, title=svis.source + " restored image")
    plt.tight_layout()
    plt.show(block=False)
    
    filename = "3C277.1_avg_%s_%s_restored.fits" % (svis.source, mode)
    export_image_to_fits(restored, filename)
