"""
Functions used in eMERLIN RASCIL pipelie
"""

__all__ = ["ms_list",
           "ms_load",
           "flag",
           "plot_vis",
           "average_channels",
           "combine_spw",
           "get_advice",
           "convert_stokesI",
           "create_images",
           "weight",
           "cip",
           "ical",
           "write_results",
           "save_calibrated"]

import logging

import matplotlib.pyplot as plt
import numpy

from rascil.data_models import Image
from rascil.processing_components import show_image, qa_image, \
    export_image_to_fits, \
    qa_gaintable, create_blockvisibility_from_ms, \
    integrate_visibility_by_channel, convert_blockvisibility_to_stokesI, \
    convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility, \
    weight_visibility, create_image_from_visibility, \
    advise_wide_field, create_calibration_controls, gaintable_plot, \
    list_ms, concatenate_blockvisibility_frequency, \
    plot_uvcoverage, plot_visibility, apply_gaintable, \
    export_blockvisibility_to_ms
from rascil.workflows import continuum_imaging_list_serial_workflow, \
    ical_list_serial_workflow

log = logging.getLogger('logger')


def ms_list(eMRP):
    log.info("Listing Measurement Set {0}".format(eMRP['inputs']['ms_path']))
    sources, dds = list_ms(eMRP['inputs']['ms_path'])
    log.info("MS contains sources {}".format(sources))
    log.info("MS contains data descriptors {}".format(dds))
    return sources, dds


def ms_load(eMRP):
    log.info("Loading Measurement Set {0}".format(eMRP['inputs']['ms_path']))
    bvis_list = \
        create_blockvisibility_from_ms(eMRP['inputs']['ms_path'],
                                       datacolumn=
                                       eMRP['defaults']['load_ms'][
                                           'data_column'],
                                       selected_sources=
                                       eMRP['defaults']['load_ms']['sources'],
                                       selected_dds=eMRP['defaults']['load_ms'][
                                           'dds'])
    sources = numpy.unique([bv.source for bv in bvis_list])
    log.info('Loaded sources {}'.format(sources))
    if eMRP['defaults']['load_ms']['verbose']:
        for bvis in bvis_list:
            log.info(str(bvis))
    return bvis_list


def flag(bvis_list, eMRP):
    log.info("Flagging visibility data")
    try:
        import aoflagger as aof
        
        for bvis in bvis_list:
            ntimes, nant, _, nch, npol = bvis.vis.shape
            
            aoflagger = aof.AOFlagger()
            # Shape of returned buffer is actually nch, ntimes
            data = aoflagger.make_image_set(ntimes, nch, npol * 2)
            
            log.info("Number of times: " + str(data.width()))
            log.info("Number of antennas:" + str(nant))
            log.info("Number of channels: " + str(data.height()))
            log.info("Number of polarisations: " + str(npol))
            eMERLIN_strategy = \
                aoflagger.load_strategy(eMRP['defaults']['flag']['strategy'])
            
            for a2 in range(0, nant - 1):
                for a1 in range(a2 + 1, nant):
                    for pol in range(npol):
                        data.set_image_buffer(2 * pol,
                                              numpy.real(bvis.vis[:, a1, a2, :,
                                                         pol]).T)
                        data.set_image_buffer(2 * pol + 1,
                                              numpy.imag(bvis.vis[:, a1, a2, :,
                                                         pol]).T)
                    
                    flags = aoflagger.run(eMERLIN_strategy, data)
                    flagvalues = flags.get_buffer() * 1
                    bvis.data['flags'][:, a1, a2, :, :] = flagvalues.T[
                        ..., numpy.newaxis]
                    flagcount = sum(sum(flagvalues))
                    log.info(str(a1) + " " + str(
                        a2) + ": percentage flags on zero data: "
                             + str(flagcount * 100.0 / (nch * ntimes)) + "%")
    except ModuleNotFoundError:
        log.error('aoflagger is not loaded - cannot flag')
    
    return bvis_list


def plot_vis(bvis_list, eMRP):
    """
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    for bvis in bvis_list:
        plt.clf()
        plot_uvcoverage([bvis], title='UV Coverage {source:s}'.format(
            source=bvis.source))
        plt.tight_layout()
        plt.show(block=False)
        plt.clf()
        plot_visibility([bvis],
                        title='Visibility amplitude {source:s}'.format(
                            source=bvis.source))
        plt.tight_layout()
        plt.show(block=False)


def average_channels(bvis_list, eMRP):
    log.info("Averaging within spectral windows")
    avis_list = [integrate_visibility_by_channel(bvis) for bvis in bvis_list]
    if eMRP['defaults']['average_channels']['verbose']:
        for avis in avis_list:
            log.info(str(avis))
    return avis_list


def combine_spw(bvis_list, eMRP):
    log.info("Combining across spectral windows")
    return [concatenate_blockvisibility_frequency(bvis_list)]


def get_advice(bvis_list, eMRP):
    """
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    return [advise_wide_field(bvis,
                              delA=eMRP['defaults']["get_advice"]["delA"],
                              oversampling_synthesised_beam=eMRP['defaults']
                              ["get_advice"]["oversampling_synthesised_beam"],
                              guard_band_image=eMRP['defaults']["get_advice"]
                              ["guard_band_image"],
                              wprojection_planes=eMRP['defaults']["get_advice"]
                              ["wprojection_planes"],
                              verbose=eMRP['defaults']["get_advice"]["verbose"])
            for bvis in bvis_list]


def convert_stokesI(bvis_list, eMRP):
    """

    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Converting to stokesI visibility")
    return [convert_blockvisibility_to_stokesI(bvis)
            for bvis in bvis_list]


def create_images(bvis_list, eMRP):
    """
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    model_list = list()
    for bvis in bvis_list:
        frequency = [numpy.mean(bvis.frequency)]
        channel_bandwidth = [numpy.sum(bvis.channel_bandwidth)]
        model_list.append(
            create_image_from_visibility(bvis,
                                         npixel=eMRP['defaults']
                                         ["create_images"]["npixel"],
                                         cellsize=eMRP['defaults']
                                         ["create_images"]['cellsize'],
                                         nchan=1,
                                         frequency=frequency,
                                         channel_bandwidth=channel_bandwidth))
    if eMRP['defaults']["create_images"]["verbose"]:
        for model in model_list:
            log.info(str(model))
    return model_list


def weight(bvis_list, model_list, eMRP):
    """
    
    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    log.info("Applying {} weighting".format(eMRP['defaults']['weight']
                                            ['algorithm']))
    new_bvis_list = list()
    for bvis, model in zip(bvis_list, model_list):
        cvis = convert_blockvisibility_to_visibility(bvis)
        cvis = weight_visibility(cvis, model,
                                 weighting=eMRP['defaults']['weight']
                                 ['algorithm'])
        new_bvis_list.append(convert_visibility_to_blockvisibility(cvis))
    return bvis_list


def cip(bvis_list, model_list, eMRP):
    """ Continuum imaging pipeline

    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    for bvis, model in zip(bvis_list, model_list):
        log.info("Processing {source:s} via continuum imaging pipeline".format(
            source=bvis.source))
    
    results = continuum_imaging_list_serial_workflow(
        bvis_list, model_list,
        context=eMRP['defaults']['global']['imaging_context'],
        nmajor=eMRP['defaults']['cip']['nmajor'],
        niter=eMRP['defaults']['cip']['niter'],
        algorithm=eMRP['defaults']['cip']['algorithm'],
        nmoment=eMRP['defaults']['cip']['nmoment'],
        scales=eMRP['defaults']['cip']['scales'],
        gain=eMRP['defaults']['cip']['gain'],
        fractional_threshold=eMRP['defaults']['cip']['fractional_threshold'],
        threshold=eMRP['defaults']['cip']['threshold'],
        window_shape=eMRP['defaults']['cip']['window_shape'],
        do_wstacking=eMRP['defaults']['cip']['do_wstacking'])
    
    return results


def ical(bvis_list, model_list, eMRP):
    """ Continuum imaging pipeline with self-calibration

    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    
    controls = create_calibration_controls()
    controls['T']['first_selfcal'] = eMRP['defaults']["ical"]["T_first_selfcal"]
    controls['T']['phase_only'] = eMRP['defaults']["ical"]["T_phase_only"]
    controls['T']['timeslice'] = eMRP['defaults']["ical"]["T_timeslice"]
    controls['G']['first_selfcal'] = eMRP['defaults']["ical"]["G_first_selfcal"]
    controls['G']['phase_only'] = eMRP['defaults']["ical"]["G_phase_only"]
    controls['G']['timeslice'] = eMRP['defaults']["ical"]["G_timeslice"]
    
    for bvis, model in zip(bvis_list, model_list):
        log.info("Processing {source:s} via ICAL pipeline".format(
            source=bvis.source))
    
    results = ical_list_serial_workflow(
        bvis_list, model_list,
        context=eMRP['defaults']['global']['imaging_context'],
        nmajor=eMRP['defaults']['ical']['nmajor'],
        niter=eMRP['defaults']['ical']['niter'],
        algorithm=eMRP['defaults']['ical']['algorithm'],
        nmoment=eMRP['defaults']['ical']['nmoment'],
        scales=eMRP['defaults']['ical']['scales'],
        gain=eMRP['defaults']['ical']['gain'],
        fractional_threshold=eMRP['defaults']['ical']['fractional_threshold'],
        threshold=eMRP['defaults']['ical']['threshold'],
        window_shape=eMRP['defaults']['ical']['window_shape'],
        calibration_context=eMRP['defaults']['ical']['calibration_context'],
        do_selfcal=eMRP['defaults']['ical']['do_selfcal'],
        do_wstacking=eMRP['defaults']['ical']['do_wstacking'],
        controls=controls)
    
    return results


def write_results(eMRP, bvis_list, mode, results):
    """
    
    :param deconvolved:
    :param residual:
    :param restored:
    :return:
    """
    
    origins = ["deconvolved", "residual", "restored", "gt_list"]
    
    def write_image(im, origin, ivis):
        filename_root = \
            "{project:s}_{source:s}_{origin:s}_{mode}".format(
                project=eMRP['defaults']["global"]["project"],
                source=bvis.source,
                origin=origin,
                mode=mode)
        log.info(qa_image(im, context=filename_root))
        plt.clf()
        show_image(im, title="{0} image".format(filename_root),
                   cm=eMRP["defaults"]["global"]["cmap"])
        plt.tight_layout()
        plt.show(block=False)
        filename = "{root:s}_{ivis}.fits".format(root=filename_root,
                                                 ivis=ivis)
        export_image_to_fits(im, filename)
    
    for iorigin, _ in enumerate(results):
        origin = origins[iorigin]
        if iorigin <3:
            for ivis, bvis in enumerate(bvis_list):
                result = results[iorigin][ivis]
                if origin == "residual":
                    result = result[0]
                if isinstance(result, Image):
                    write_image(result, origin, ivis)
        else:
            for gt in results[iorigin]:
                filename_root = \
                "{project:s}_{source:s}_{origin:s}_{mode}".format(
                    project=eMRP['defaults']["global"]["project"],
                    source="",
                    origin=origin,
                    mode=mode)
                for cc in eMRP['defaults']['ical']['calibration_context']:
                    log.info(qa_gaintable(gt[cc],
                                        context=filename_root + " " + cc))
                    fig, ax = plt.subplots(1, 1)
                    gaintable_plot(gt[cc], ax, value='amp')
                    plt.tight_layout()
                    plt.show(block=False)
                    fig, ax = plt.subplots(1, 1)
                    gaintable_plot(gt[cc], ax, value='phase')
                    plt.tight_layout()
                    plt.show(block=False)

def save_calibrated(gt_list, bvis_list, eMRP):
    """
    
    :param bvis_list:
    :param gt_list:
    :param eMRP:
    :return:
    """
    cvis_list = list()
    for bvis in bvis_list:
        cvis = None
        for gt in gt_list:
            cvis = bvis
            for cc in eMRP['defaults']['ical']['calibration_context']:
                cvis = apply_gaintable(cvis, gt[cc])
        cvis_list.append(cvis)
        
    ms_out = eMRP['inputs']['ms_path'] + "_out"
    log.info("Writing Measurement Set {0}".format(ms_out))

    export_blockvisibility_to_ms(ms_out, cvis_list)