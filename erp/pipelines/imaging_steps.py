#!/usr/bin/env python

__all__ = ["load_ms", "flag", "average_channels", "get_advice",
           "create_images", "weight", "cip", "ical", "write_results"]

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
    advise_wide_field, create_calibration_controls, gaintable_plot
from rascil.workflows import continuum_imaging_list_serial_workflow, \
    ical_list_serial_workflow

log = logging.getLogger(__file__)


def load_ms(eMRP):
    bvis_list = create_blockvisibility_from_ms(eMRP['global']['ms'],
                                               datacolumn=eMRP['global'][
                                                   'data_column'],
                                               selected_sources=eMRP['load_ms'][
                                                   'sources'])
    sources = numpy.unique([bv.source for bv in bvis_list])
    log.info('Loaded sources {}'.format(sources))
    return bvis_list


def flag(bvis_list, eMRP):
    try:
        import aoflagger as aof
        
        for bvis in bvis_list:
            ntimes, nant, _, nch, npol = bvis.vis.shape
            
            aoflagger = aof.AOFlagger()
            # Shape of returned buffer is actually nch, ntimes
            data = aoflagger.make_image_set(ntimes, nch, npol * 2)
            
            print("Number of times: " + str(data.width()))
            print("Number of antennas:" + str(nant))
            print("Number of channels: " + str(data.height()))
            print("Number of polarisations: " + str(npol))
            eMERLIN_strategy = \
                aoflagger.load_strategy(eMRP['flag']['strategy'])
            
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
                    print(str(a1) + " " + str(
                        a2) + ": percentage flags on zero data: "
                          + str(flagcount * 100.0 / (nch * ntimes)) + "%")
    except ModuleNotFoundError:
        log.error('aoflagger is not loaded - cannot flag')
    
    return bvis_list


def average_channels(bvis_list, eMRP):
    return [integrate_visibility_by_channel(bvis) for bvis in bvis_list]


def get_advice(bvis_list, eMRP):
    """
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    return [advise_wide_field(bvis,
                              delA=eMRP["get_advice"]["delA"],
                              oversampling_synthesised_beam=eMRP["get_advice"][
                                  "oversampling_synthesised_beam"],
                              guard_band_image=eMRP["get_advice"][
                                  "guard_band_image"],
                              wprojection_planes=eMRP["get_advice"][
                                  "wprojection_planes"],
                              verbose=False) for bvis in bvis_list]


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
        ivis = convert_blockvisibility_to_stokesI(bvis)
        model_list.append(
            create_image_from_visibility(ivis,
                                         npixel=eMRP["global"]["npixel"],
                                         cellsize=eMRP["global"][
                                                      'cellsize'] / 3.0,
                                         nchan=1,
                                         frequency=frequency,
                                         channel_bandwidth=channel_bandwidth))
    return model_list


def weight(bvis_list, model_list, eMRP):
    """
    
    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    new_bvis_list = list()
    for bvis, model in zip(bvis_list, model_list):
        cvis = convert_blockvisibility_to_visibility(bvis)
        cvis = weight_visibility(cvis, model, eMRP['weighting']['weight'])
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
        print("Processing {source:s} via continuum imaging pipeline".format(
            source=bvis.source))
    
    results = continuum_imaging_list_serial_workflow(
        bvis_list, model_list,
        context=eMRP['global']['imaging_context'],
        nmajor=eMRP['cip']['nmajor'],
        niter=eMRP['cip']['niter'],
        algorithm=eMRP['cip']['algorithm'],
        scales=eMRP['cip']['scales'],
        gain=eMRP['cip']['gain'],
        fractional_threshold=eMRP['cip']['fractional_threshold'],
        threshold=eMRP['cip']['threshold'],
        window_shape=eMRP['cip']['window_shape'],
        do_wstacking=eMRP['cip']['do_wstacking'])
    
    return results


def ical(bvis_list, model_list, eMRP):
    """ Continuum imaging pipeline with self-calibration

    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    
    controls = create_calibration_controls()
    controls['T']['first_selfcal'] = eMRP["ical"]["T_first_selfcal"]
    controls['T']['phase_only'] = eMRP["ical"]["T_phase_only"]
    controls['T']['timeslice'] = eMRP["ical"]["T_timeslice"]
    controls['G']['first_selfcal'] = eMRP["ical"]["G_first_selfcal"]
    controls['G']['phase_only'] = eMRP["ical"]["T_phase_only"]
    controls['G']['timeslice'] = eMRP["ical"]["T_timeslice"]
    
    for bvis, model in zip(bvis_list, model_list):
        print("Processing {source:s} via ICAL pipeline".format(
            source=bvis.source))
    
    results = ical_list_serial_workflow(
        bvis_list, model_list,
        context=eMRP['global']['imaging_context'],
        nmajor=eMRP['ical']['nmajor'],
        niter=eMRP['ical']['niter'],
        algorithm=eMRP['ical']['algorithm'],
        scales=eMRP['ical']['scales'],
        gain=eMRP['ical']['gain'],
        fractional_threshold=eMRP['ical']['fractional_threshold'],
        threshold=eMRP['ical']['threshold'],
        window_shape=eMRP['ical']['window_shape'],
        calibration_context=eMRP['ical']['calibration_context'],
        do_selfcal=eMRP['ical']['do_selfcal'],
        do_wstacking=eMRP['ical']['do_wstacking'],
        controls=controls)
    
    return results


def write_results(eMRP, bvis_list, origin, results):
    """
    
    :param deconvolved:
    :param residual:
    :param restored:
    :return:
    """
    names = ["deconvolved", "residual", "restored", "gt_list"]
    
    for i, bvis in enumerate(bvis_list):
        filename_root = \
            "{project:s}_{source:s}_{origin:s}".format(
                project=eMRP["global"]["project"],
                source=bvis.source,
                origin=origin)
        for i, result in enumerate(results):
            if isinstance(result, Image):
                print(qa_image(result, context='{0} image'.format(names[i])))
                plt.clf()
                show_image(result, title="{0}: {1} image".format(filename_root,
                                                                 names[i],
                                                                 cm=
                                                                 eMRP["global"][
                                                                     "cmap"]))
                plt.tight_layout()
                plt.show(block=False)
                filename = "{root:s}_{name}.fits".format(root=filename_root,
                                                         name=names[i])
                export_image_to_fits(result, filename)
            elif isinstance(result, list):
                for gt in result:
                    print(qa_gaintable(gt['T'], context=filename_root))
                    fig, ax = plt.subplots(1, 1)
                    gaintable_plot(gt, ax, value='phase')
                    plt.show(block=False)
                    print(qa_gaintable(gt['G'], context=filename_root))
                    fig, ax = plt.subplots(1, 1)
                    gaintable_plot(gt, ax, value='amp')
                    plt.show(block=False)
