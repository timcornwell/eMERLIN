""" Functions used in ERP

• Loading and inspecting data 
• Editing
• Initial phase-calibration of calibration sources
• Setting the flux scale
• Time-dependent amplitude calibration of calibration sources
• Polarization leakage and angle calibration
• Target imaging and self-calibration
• Imaging in polarization
• Image measurements

"""

import logging
import matplotlib.pyplot as plt
import numpy

from rascil.processing_components.calibration import qa_gaintable, create_calibration_controls, gaintable_plot
from rascil.processing_components.image import show_image, qa_image, export_image_to_fits
from rascil.processing_components.imaging import weight_visibility, create_image_from_visibility, advise_wide_field
from rascil.processing_components.visibility import create_blockvisibility_from_ms, list_ms, \
    integrate_visibility_by_channel, concatenate_blockvisibility_frequency, \
    convert_blockvisibility_to_stokesI, convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from rascil.workflows.serial.pipelines import continuum_imaging_list_serial_workflow
from rascil.workflows.serial.pipelines import ical_list_serial_workflow

log = logging.getLogger(__file__)


def erp_uvfits_to_ms(uvfits, msfile):
    """ List the contents of an MS

    :param uvfits:
    :param msfile:
    :return:
    """
    log.info('erp_load_sources: {lms}'.format(lms=list_ms(msfile)))


def erp_list_ms(msfile):
    """ List the contents of an MS
    
    :param msfile:
    :return:
    """
    log.info('erp_load_sources: {lms}'.format(lms=list_ms(msfile)))


def erp_load_ms(msfile, selected_sources=None, selected_dds=None, data_column='DATA',
                **kwargs):
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
    log.info('erp_load_sources: Sources loaded {sources}'.format(sources=sources))
    return bvis_list


def erp_flag(bvis, strategy, **kwargs):
    """ Flag with AOFlagger strategy
    
    :param bvis:
    :param strategy:
    :return:
    """
    
    import aoflagger as aof
    
    ntimes, nant, _, nch, npol = bvis.vis.shape
    
    aoflagger = aof.AOFlagger()
    # Shape of returned buffer is actually nch, ntimes
    data = aoflagger.make_image_set(ntimes, nch, npol * 2)
    
    print("Number of times: " + str(data.width()))
    print("Number of antennas:" + str(nant))
    print("Number of channels: " + str(data.height()))
    print("Number of polarisations: " + str(npol))
    
    for a2 in range(0, nant - 1):
        for a1 in range(a2 + 1, nant):
            for pol in range(npol):
                data.set_image_buffer(2 * pol, numpy.real(bvis.vis[:, a1, a2, :, pol]).T)
                data.set_image_buffer(2 * pol + 1, numpy.imag(bvis.vis[:, a1, a2, :, pol]).T)
            
            flags = aoflagger.run(strategy, data)
            flagvalues = flags.get_buffer() * 1
            bvis.data['flags'][:, a1, a2, :, :] = flagvalues.T[..., numpy.newaxis]
            flagcount = sum(sum(flagvalues))
            print(str(a1) + " " + str(a2) + ": percentage flags on zero data: "
                  + str(flagcount * 100.0 / (nch * ntimes)) + "%")
    
    return bvis


def erp_initial_phase_cal(bvis, model, **kwargs):
    """
    
    :return:
    """
    log.info("erp_initial_phase_cal: NYI")
    pass

def erp_find_delay(bvis, model, gt, **kwargs):
    """ Find delay solution
    
    :param bvis:
    :param gt:
    :param kwargs:
    :return:
    """

def erp_set_flux_scale(bvis, source, **kwargs):
    """
    
    :return:
    """
    log.info("erp_set_flux_scale: NYI")


def erp_amplitude_cal(bvis, model, **kwargs):
    """

    :return:
    """
    log.info("erp_amplitude_cal: NYI")
    pass


def erp_pol_cal(bvis, polmodel, **kwargs):
    """

    :return:
    """
    log.info("erp_pol_cal: NYI")
    pass


def integrate_frequency(bvis_list, nspw, sources, **kwargs):
    """
    
    :param bvis_list:
    :param nspw:
    :param sources:
    :return:
    """
    avis_list = [integrate_visibility_by_channel(bvis) for bvis in bvis_list]
    print(numpy.max(avis_list[0].flagged_vis))
    blockvis = [concatenate_blockvisibility_frequency(avis_list[isource * nspw * (isource * nspw + nspw)])
                for isource, source in enumerate(sources)]
    print(numpy.max(blockvis[0].flagged_vis))
    return avis_list


def erp_cip(bvis, **kwargs):
    """ Continuum Imaging Pipeline

    :return:
    """
    log.info("erp_cip: NYI")
    pass


def erp_ical(bvis, model, **kwargs):
    """ ICAL pipeline (selfcalibration)

    :return:
    """
    log.info("erp_ical: NYI")
    
    controls = create_calibration_controls()
    controls['T']['first_selfcal'] = 1
    controls['T']['phase_only'] = True
    controls['T']['timeslice'] = 3.0
    controls['G']['first_selfcal'] = 10
    controls['G']['phase_only'] = False
    controls['G']['timeslice'] = 3600.0
    
    deconvolved_list, residual_list, restored_list, gt_list = ical_list_serial_workflow([bvis], [model],
                                                                                        context='ng',
                                                                                        nmajor=15,
                                                                                        niter=1000, algorithm='msclean',
                                                                                        scales=[0, 3, 10], gain=0.1,
                                                                                        fractional_threshold=0.5,
                                                                                        threshold=0.0015,
                                                                                        window_shape='quarter',
                                                                                        do_wstacking=False,
                                                                                        global_solution=False,
                                                                                        calibration_context='TG',
                                                                                        do_selfcal=True,
                                                                                        controls=controls)
    deconvolved = deconvolved_list[0]
    residual = residual_list[0][0]
    restored = restored_list[0]
    gt = gt_list[0]['T']
    
    log.info('erp_ical: Restored image {}'.format(qa_gaintable(restored)))
    log.info('erp_ical: Residual image {}'.format(qa_gaintable(residual)))
    
    log.info('erp_ical: GainTable {}'.format(qa_gaintable(gt)))
    
    return deconvolved, residual, restored, gt


def plot_gaintable(gt, **kwargs):
    """
    
    :param gt:
    :param kwargs:
    :return:
    """
    print(qa_gaintable(gt))
    fig, ax = plt.subplots(1, 1)
    gaintable_plot(gt, ax, value='amp')
    plt.show(block=False)


def erp_advise(bvis, **kwargs):
    """
    
    :param bvis:
    :param kwargs:
    :return:
    """
    advice = advise_wide_field(bvis, verbose=False)
    return advice


def erp_weight(bvis, model, method="uniform"):
    """
    
    :param bvis:
    :param model:
    :param method:
    :return:
    """
    advice = erp_advise(bvis)
    frequency = [numpy.mean(bvis.frequency)]
    channel_bandwidth = [numpy.sum(bvis.channel_bandwidth)]
    ivis = convert_blockvisibility_to_stokesI(bvis)
    print(numpy.max(ivis.flagged_weight))
    model = create_image_from_visibility(ivis, npixel=1024, cellsize=advice['cellsize'] / 3.0, nchan=1,
                                         frequency=frequency, channel_bandwidth=channel_bandwidth)
    vis = convert_blockvisibility_to_visibility(bvis)
    print(numpy.max(vis.flagged_weight))
    cvis = weight_visibility(vis, model)
    print(numpy.max(cvis.flagged_weight))
    ivis = convert_visibility_to_blockvisibility(cvis)
    print(numpy.max(ivis.flagged_weight))
    
    return vis


def erp_show_image(im, filename_root, title='Restored image'):
    """
    
    :param im:
    :param filename_root:
    :param title:
    :return:
    """
    print(qa_image(im, context=title))
    plt.clf()
    show_image(im, title=filename_root + " " + title, cm="rainbow")
    plt.tight_layout()
    plt.show(block=False)
    filename = "{root:s}_{title:s}.fits".format(root=filename_root, title=title)
    export_image_to_fits(im, filename)


def image_pol(bvis, model, **kwargs):
    """
    
    :param bvis:
    :param model:
    :param kwargs:
    :return:
    """
    
    deconvolved_list, residual_list, restored_list, gt_list = \
        continuum_imaging_list_serial_workflow([bvis], [model],
                                               context='ng',
                                               nmajor=15,
                                               niter=1000, algorithm='msclean',
                                               scales=[0, 3, 10], gain=0.1,
                                               fractional_threshold=0.5,
                                               threshold=0.0015,
                                               window_shape='quarter',
                                               do_wstacking=False,
                                               global_solution=False,
                                               calibration_context='TG',
                                               do_selfcal=True)
    deconvolved = deconvolved_list[0]
    residual = residual_list[0][0]
    restored = restored_list[0]
    gt = gt_list[0]['T']
    
    log.info('erp_ical: Restored image {}'.format(qa_gaintable(restored)))
    log.info('erp_ical: Residual image {}'.format(qa_gaintable(residual)))
    
    log.info('erp_ical: GainTable {}'.format(qa_gaintable(gt)))
    
    return deconvolved, residual, restored, gt

def erp_calculate_polarisation(im, **kwargs):
    """
    
    :param im:
    :param kwargs:
    :return:
    """
    log.info('erp_calculate_polarisation: Extracting polarisation images')
    return im