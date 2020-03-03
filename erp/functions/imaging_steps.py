"""
Functions used in eMERLIN RASCIL pipeline.

Following the design of the eMERLIN CASA pipeline, parameters are passed in the eMRP dictionary.
The "inputs" sub-dict contains values read from the inputs.ini file, and the "defaults" sub-dict contains
values read from the defaults_params.json file.


The major functions are written using Dask.delayed to delay execution. This is wrapped in the
rsexecute execute and compute classes: rsexecute.execute(myfunc)(pars) says to delay or defer the
actual computation to later. rsexecute.compute(my_list) actually forces the computation of a
list of delayed function calls.

The data for a particular source from a MeasurementSet are read in, one BlockVis per data description.
The visbility data are kept internally as a list of RASCIL BlockVis. This list can be transformed by
for example averaging and splitting. Thus 4 spectral windows each of 128 channels can be split and
averaged into 16 BlockVis, in each of which 32 of the original channels averaged into a single channel.
Imaging and calibration proceeds by processing these BlockVis in parallel, using Dask/rsexecute to
distribute the processing over the available cores or nodes available.

"""

__all__ = ["initialize_pipeline",
           "list_ms",
           "load_ms",
           "flag",
           "plot_vis",
           "average_channels",
           "get_advice",
           "convert_stokesI",
           "create_images",
           "weight",
           "cip",
           "ical",
           "write_images",
           "write_gaintables",
           "apply_calibration",
           "write_ms",
           "finalize_pipeline"]

import logging

import matplotlib.pyplot as plt
import numpy

from rascil.data_models import Image, export_gaintable_to_hdf5
from rascil.processing_components import show_image, qa_image, export_image_to_fits, \
    calculate_image_frequency_moments, image_gather_channels, image_scatter_channels, \
    qa_gaintable, create_blockvisibility_from_ms, average_blockvisibility_by_channel, \
    convert_blockvisibility_to_stokesI, create_image_from_visibility, advise_wide_field, \
    create_calibration_controls, gaintable_plot, concatenate_blockvisibility_frequency, \
    plot_uvcoverage, plot_visibility, apply_gaintable, export_blockvisibility_to_ms
from rascil.processing_components.visibility import list_ms as rascil_list_ms
from rascil.workflows import continuum_imaging_list_rsexecute_workflow, \
    ical_list_rsexecute_workflow, weight_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support import rsexecute

log = logging.getLogger('logger')


def initialize_pipeline(eMRP, get_logger):
    """ Initialise the pipeline: set up dask if we are using it.

    :param eMRP: Parameters for the pipeline read from template_default_params.json
    :param get_logger: Function to get a logger.
    """
    if eMRP['defaults']['global']['distributed']:
        log.info("Distributed processing using Dask")
        rsexecute.set_client(use_dask=True,
                             n_workers=eMRP['defaults']['global']['nworkers'])
        rsexecute.run(get_logger)
        rsexecute.init_statistics()
    else:
        log.info("Serial processing")
        rsexecute.set_client(use_dask=False)


def finalize_pipeline(eMRP):
    """ Finalise the pipeline

    :param eMRP:
    """
    log.info("Finalising pipeline")
    results_directory = eMRP['defaults']["global"]["results_directory"]

    if eMRP['defaults']['global']['distributed']:
        rsexecute.save_statistics\
            ('{results_directory}/eMERLIN_RASCIL_pipeline'.format(results_directory=results_directory))
        rsexecute.close()


def list_ms(eMRP):
    """ List the contents of MeasurementSet
    
    :param eMRP:
    :return: list of source names in MS, list of data descriptions in MS
    """
    log.info("Listing Measurement Set {0}".format(eMRP['defaults']['load_ms']['ms_path']))
    sources, dds = rascil_list_ms(eMRP['defaults']['load_ms']['ms_path'])
    log.info("MS contains sources {}".format(sources))
    log.info("MS contains data descriptors {}".format(dds))
    return sources, dds


def load_ms(eMRP):
    """ Load the MeasurementSet into a list of RASCIL BlockVis
    
    :param eMRP:
    :return: list of BlockVis
    """
    log.info("Loading Measurement Set {0}".format(eMRP['defaults']['load_ms']['ms_path']))
    
    def load_and_list(dd):
        bv = create_blockvisibility_from_ms(eMRP['defaults']['load_ms']['ms_path'],
                                            datacolumn=eMRP['defaults']['load_ms']['data_column'],
                                            selected_sources=[eMRP['defaults']['load_ms']['source']],
                                            selected_dds=[dd])[0]
        
        if eMRP['defaults']['load_ms']['verbose']:
            log.info(str(bv))
        return bv
    
    bvis_list = [rsexecute.execute(load_and_list)(dd) for dd in eMRP['defaults']['load_ms']['dds']]
    
    # Put this to the Dask cluster
    bvis_list = rsexecute.persist(bvis_list)
    
    return bvis_list


def flag(bvis_list, eMRP):
    """ Flag the visibility using aoflagger
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Flagging visibility data")
    try:
        import aoflagger as aof
        
        def flag_bvis(bvis):
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
        return [rsexecute.execute(flag_bvis(bv)) for bv in bvis_list]
    
    except ModuleNotFoundError:
        log.error('aoflagger is not loaded - cannot flag - will continue pipeline')

    return bvis_list


def plot_vis(bvis_list, eMRP):
    """ Plot the uv coverage and vis amplitude
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Plotting visibility data")

    def plot(bvis):
        plt.clf()
        plot_uvcoverage([bvis], title='UV Coverage {source:s}'.format(source=bvis.source))
        plt.tight_layout()
        plt.show(block=False)
        plt.clf()
        plot_visibility([bvis],
                        title='Visibility amplitude {source:s}'.format(source=bvis.source))
        plt.tight_layout()
        plt.show(block=False)
        return bvis
    
    return [rsexecute.execute(plot)(bv) for bv in bvis_list]


def average_channels(bvis_list, eMRP):
    """ Average each BlockVis across frequency, producing a new BlockVis for each new channel
    
    :param bvis_list:
    :param eMRP:
    :return: list of BlockVisibilities
    """
    nchan = eMRP['defaults']["average_channels"]["nchan"]
    nout = eMRP['defaults']["average_channels"]["original_nchan"] // \
           eMRP['defaults']["average_channels"]["nchan"]
    log.info("Averaging by {nchan} channels and splitting into {nout} BlockVis".format(nchan=nchan, nout=nout))
    average_vis_list = [rsexecute.execute(average_blockvisibility_by_channel, nout=nout)
                        (bvis_list[idd], eMRP['defaults']["average_channels"]["nchan"])
                        for idd, dd in enumerate(eMRP['defaults']['load_ms']['dds'])]
    average_vis_list = [item for sublist in average_vis_list for item in sublist]
    
    bvis_list = rsexecute.persist(average_vis_list)
    return bvis_list



def get_advice(bvis_list, eMRP):
    """ Get advice on imaging parameters
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    return bvis_list, \
           [rsexecute.execute(advise_wide_field)(bvis,
                              delA=eMRP['defaults']["get_advice"]["delA"],
                              oversampling_synthesised_beam=eMRP['defaults']["get_advice"]["oversampling_synthesised_beam"],
                              guard_band_image=eMRP['defaults']["get_advice"]["guard_band_image"],
                              wprojection_planes=eMRP['defaults']["get_advice"]["wprojection_planes"],
                              verbose=eMRP['defaults']["get_advice"]["verbose"])
            for bvis in bvis_list]


def convert_stokesI(bvis_list, eMRP):
    """ Convert BlockVis to stokeI

    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Converting to stokes I visibility")
    bvis_list = [rsexecute.execute(convert_blockvisibility_to_stokesI)(bvis) for bvis in bvis_list]
    
    bvis_list = rsexecute.persist(bvis_list)
    return bvis_list


def create_images(bvis_list, eMRP):
    """ Create the template images
    
    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Creating template images")
    model_list = [rsexecute.execute(create_image_from_visibility, nout=1)
                  (bvis,
                   npixel=eMRP['defaults']["create_images"]["npixel"],
                   cellsize=eMRP['defaults']["create_images"]['cellsize'],
                   nchan=1,
                   frequency=[numpy.mean(bvis.frequency)],
                   channel_bandwidth=[numpy.sum(bvis.channel_bandwidth)])
                  for bvis in bvis_list]
    
    model_list = rsexecute.persist(model_list)
    return model_list


def weight(bvis_list, model_list, eMRP):
    """ Apply visibility weighting
    
    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    log.info("Applying {} weighting".format(eMRP['defaults']['weight']['algorithm']))
    
    bvis_list = weight_list_rsexecute_workflow(bvis_list, model_list,
                                               weighting=eMRP['defaults']['weight']['algorithm'])
    
    bvis_list = rsexecute.persist(bvis_list)
    return bvis_list


def cip(bvis_list, model_list, eMRP):
    """ Continuum imaging pipeline

    :param bvis_list:
    :param model_list:
    :param eMRP:
    :return:
    """
    log.info("Processing with RASCIL continuum imaging pipeline")

    results = continuum_imaging_list_rsexecute_workflow(
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
    controls['B']['first_selfcal'] = eMRP['defaults']["ical"]["B_first_selfcal"]
    controls['B']['phase_only'] = eMRP['defaults']["ical"]["B_phase_only"]
    controls['B']['timeslice'] = eMRP['defaults']["ical"]["B_timeslice"]
    
    log.info("Processing with RASCIL ICAL pipeline")
    
    results = ical_list_rsexecute_workflow(
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
        global_solution=eMRP['defaults']['ical']['global_solution'],
        tol=eMRP['defaults']['ical']['tol'],
        controls=controls)
    
    return bvis_list, results


def write_images(eMRP, pipeline, results):
    """ Plot and save the images

    :param eMRP:
    :param pipeline:
    :param results:
    :return:
    """
    log.info("Writing images")
    results_directory = eMRP['defaults']["global"]["results_directory"]
    
    im_types = ["deconvolved", "residual", "restored"]
    
    def write_image(im, im_type="restored", index=0, axis='spw'):
        filename_root = \
            "{results_directory}/{project:s}_{source:s}_{pipeline}_{im_type:s}_{axis}{index:d}".format(
                results_directory=results_directory,
                project=eMRP['defaults']["global"]["project"],
                source=eMRP['defaults']["load_ms"]["source"],
                im_type=im_type,
                pipeline=pipeline,
                axis=axis,
                index=index)
        log.info(qa_image(im, context=filename_root))
        plt.clf()
        show_image(im, title=filename_root,
                   cm=eMRP["defaults"]["global"]["cmap"])
        plotfile = "{0}.png".format(filename_root)
        plt.savefig(plotfile)
        plt.show(block=False)
        filename = "{0}.fits".format(filename_root)
        export_image_to_fits(im, filename)
    
    # Write moment images
    if eMRP["defaults"]["write_images"]["write_moments"]:
        nspw = len(results[0])
        nmoment = eMRP["defaults"]["write_images"]["number_moments"]
        log.info("Forming {0} moment images from {1} spws".format(nmoment, nspw))
        for it, im_type in enumerate(im_types):
            channel_images = list()
            for result in results[it]:
                if im_type == "residual":
                    channel_images.append(result[0])
                else:
                    channel_images.append(result)
            channel_image = image_gather_channels(channel_images)
            moment_image = calculate_image_frequency_moments(channel_image, nmoment=nmoment)
            moment_images = image_scatter_channels(moment_image)
            for moment, im in enumerate(moment_images):
                write_image(im, im_type, moment, 'moment')
                
    # Write spectral images
    else:
        nspw = len(results[0])
        log.info("Writing {0} spws of images".format(nspw))
        for it, im_type in enumerate(im_types):
            for spw in range(nspw):
                result = results[it][spw]
                if im_type == "residual":
                    result = result[0]
                if isinstance(result, Image):
                    write_image(result, im_type, spw, 'spw')


def write_gaintables(eMRP, mode, gt_list):
    """ Plot and save gaintables

    :param eMRP:
    :param mode:
    :param results:
    :return:
    """
    log.info("Writing gaintables")
    results_directory = eMRP['defaults']["global"]["results_directory"]

    for igt, gt in enumerate(gt_list):
        for cc in eMRP['defaults']['ical']['calibration_context']:
            filename_root = \
                "{results_directory}/{project:s}_{origin:d}_{mode}".format(
                    results_directory=results_directory,
                    project=eMRP['defaults']["global"]["project"],
                    origin=igt,
                    mode=mode)
            log.info(qa_gaintable(gt[cc],
                                  context="{0} {1}".format(filename_root, cc)))
            plt.clf()
            gaintable_plot(gt[cc], title='Frequency window {igt}'.format(igt=igt), cc=cc)
            plotfile = "{0}_{1}.png".format(filename_root, cc)
            plt.savefig(plotfile)
            plt.show(block=False)
    
            gtfile = "{0}_{1}.hdf5".format(filename_root, cc)
            export_gaintable_to_hdf5(gt[cc], gtfile)


def apply_calibration(gt_list, bvis_list, eMRP):
    """ Apply the calibration to the BlockVis
    
    :param bvis_list:
    :param gt_list:
    :param eMRP:
    :return:
    """
    log.info("Applying calibration")

    cvis_list = list()
    for bvis in bvis_list:
        cvis = None
        for gt in gt_list:
            cvis = bvis
            for cc in eMRP['defaults']['ical']['calibration_context']:
                cvis = apply_gaintable(cvis, gt[cc])
        cvis_list.append(cvis)
    
    assert len(cvis_list) == len(bvis_list)
    return cvis_list


def write_ms(bvis_list, eMRP):
    """ Save the averaged and calibrated BlockVis to an MS

    :param bvis_list:
    :param eMRP:
    :return:
    """
    log.info("Combining across spectral windows")
    bvis_list = [concatenate_blockvisibility_frequency(bvis_list)]

    ms_out = eMRP['defaults']['write_ms']['msout']
    log.info("Writing Measurement Set {0}".format(ms_out))
    
    export_blockvisibility_to_ms(ms_out, bvis_list)
