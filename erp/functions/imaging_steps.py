"""
Functions used in eMERLIN RASCIL pipeline.

Control parameters are passed in the erp_params dictionary.

The major functions are written using Dask.delayed to delay execution. This is wrapped in the
rsexecute execute and compute classes: rsexecute.execute(myfunc)(pars) says to delay or defer the
actual computation to later. rsexecute.compute(my_list) actually forces the computation of a
list of delayed function calls.

The data for a particular source from a MeasurementSet are read in, one BlockVis per data description.
The visibility data are kept internally as a list of RASCIL BlockVis. This list can be transformed by
for example averaging and splitting. Thus 4 spectral windows each of 128 channels can be split and
averaged into 16 BlockVis, in each of which 32 of the original channels averaged into a single channel.
Imaging and calibration proceeds by processing these BlockVis in parallel, using Dask/rsexecute to
distribute the processing over the available cores or nodes available.

"""

__all__ = ["initialize_pipeline",
           "ingest",
           "process",
           "stage",
           "finalize_pipeline"]

import logging

import matplotlib.pyplot as plt
import numpy
from dask.distributed import Client

from rascil.data_models import Image, export_gaintable_to_hdf5
from rascil.processing_components import show_image, qa_image, export_image_to_fits, \
    calculate_image_frequency_moments, image_gather_channels, image_scatter_channels, \
    qa_gaintable, create_blockvisibility_from_ms, average_blockvisibility_by_channel, \
    convert_blockvisibility_to_stokesI, create_image_from_visibility, create_calibration_controls, \
    gaintable_plot, concatenate_blockvisibility_frequency, \
    plot_uvcoverage, plot_visibility, apply_gaintable, export_blockvisibility_to_ms
from rascil.processing_components.visibility import list_ms as rascil_list_ms
from rascil.workflows import ical_list_rsexecute_workflow, weight_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support import rsexecute

log = logging.getLogger('logger')

def initialize_pipeline(erp_params, get_logger=None):
    """ Initialise the pipeline: set up dask if we are using it.

    :param erp_params: Parameters for the pipeline read from erp_params.json
    :param get_logger: Optional function to get a logger that is shared across distributed processing

    The relevant parameters are::
    
        {
            'configure':
                {
                    'execution_engine': execution engine i.e. 'rascil',
                    'project': Name of the project e.g. 'eMERLIN_3C277_1',
                    'distributed': Distributed processing with Dask? e.g. True,
                    'memory': Memory per Dask worker e.g. None (Dask guesses) or 64GB,
                    'nworkers': Number of Dask workers e.g. None (Dask guesses) or 8
                },
        }

    
    """
    if erp_params['configure']['distributed']:
        log.info("Distributed processing using Dask")
        client = Client(n_workers=erp_params['configure']['nworkers'],
                        memory_limit=erp_params['configure']['memory'])
        rsexecute.set_client(use_dask=True, client=client)
        log.info("Dask dashboard is at http://127.0.0.1:{}".format(client.scheduler_info()['services']['dashboard']))
        if get_logger is not None: rsexecute.run(get_logger)
        rsexecute.init_statistics()
    else:
        log.info("Serial processing")
        rsexecute.set_client(use_dask=False)


def finalize_pipeline(erp_params):
    """ Finalise the pipeline
    
    If appropriate, stage the Dask statistics and close Dask.

    :param erp_params:
    
    The relevant fields in the dictionary are:
    
    """
    log.info("Finalising pipeline")
    results_directory = erp_params["stage"]["results_directory"]
    
    if rsexecute.using_dask:
        rsexecute.save_statistics \
            ('{results_directory}/eMERLIN_RASCIL_pipeline'.format(results_directory=results_directory))
        rsexecute.close()


def ingest(erp_params):
    """ Ingest the data and flag, plot, average
    
    1. Read from a MeasurementSet
    2. Optional flag with aoflagger
    3. Standard visibility plots
    4. Average over a number of channels
    5. Convert BlockVis to stokesI
    
    :param erp_params:
    :return: List of BlockVis (or graph)
    
    The relevant fields of the dictoionary are::

        'ingest':
            {
                'ms_path': MeasurementSet name e.g. 'data/3C277.1C_avg.ms',
                'data_column': 'MeasurementSet data column: DATA, MODEL_DATA, or CORRECTED_DATA,
                'source': Name of source to load e.g. '1252+5634',
                'dds': List of data descriptors to load e.g. [0, 1, 2, 3]
                'flag_strategy': AOFlaffer strategy file e.g. 'eMERLIN_strategy.rfis',
                'nchan': Number of channels to average e.g. 16,
                'original_nchan': Number of original channels e.g. 128,
                'verbose': False,
            }
            

    """
    log.info("Listing Measurement Set {0}".format(erp_params['ingest']['ms_path']))
    sources, dds = rascil_list_ms(erp_params['ingest']['ms_path'])
    log.info("MS contains sources {}".format(sources))
    log.info("MS contains data descriptors {}".format(dds))
    
    log.info("Loading Measurement Set {0}".format(erp_params['ingest']['ms_path']))
    
    def load_and_list(dd):
        bv = create_blockvisibility_from_ms(erp_params['ingest']['ms_path'],
                                            datacolumn=erp_params['ingest']['data_column'],
                                            selected_sources=[erp_params['ingest']['source']],
                                            selected_dds=[dd])[0]
        
        if erp_params['ingest']['verbose']:
            log.info(str(bv))
        return bv
    
    bvis_list = [rsexecute.execute(load_and_list)(dd) for dd in erp_params['ingest']['dds']]
    
    # Put this to the Dask cluster
    bvis_list = rsexecute.persist(bvis_list)
    
    if erp_params['ingest']['flag_strategy'] is not None:
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
                    aoflagger.load_strategy(erp_params['flag_strategy'])
                
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
            log.warning('aoflagger is not loaded - cannot flag - will continue pipeline')
    
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
    
    bvis_list = [rsexecute.execute(plot)(bv) for bv in bvis_list]
    
    nchan = erp_params["ingest"]["nchan"]
    nout = erp_params["ingest"]["original_nchan"] // \
           erp_params["ingest"]["nchan"]
    ntotal = nout * len(erp_params['ingest']['dds'])
    log.info("Averaging by {nchan} channels and splitting into {ntotal} BlockVis".format(nchan=nchan, ntotal=ntotal))
    average_vis_list = [rsexecute.execute(average_blockvisibility_by_channel, nout=nout)
                        (bvis_list[idd], erp_params["ingest"]["nchan"])
                        for idd, dd in enumerate(erp_params['ingest']['dds'])]
    bvis_list = [item for sublist in average_vis_list for item in sublist]
    
    log.info("Converting to stokes I visibility")
    bvis_list = [rsexecute.execute(convert_blockvisibility_to_stokesI)(bvis) for bvis in bvis_list]
    
    bvis_list = rsexecute.persist(bvis_list)
    
    return bvis_list


def process(bvis_list, erp_params):
    """ Actually process
    
    1. Create template images
    2. Optionally weight the data
    3. Run the ICAL pipeline (imaging and self-calibration)
    
    :param bvis_list: List of BlockVis to progress in parallel (or graph)
    :param erp_params:
    :return: List of BlockVis (or graph), results from ical
    
    The relevant fields of the dictionary are::
    
        'process':
            {
                'cellsize': Image cellsize in radians e.g. 9e-08,
                'npixel': Image size in pixels e.g. 256,
                'imaging_context': Type of gridder e.g. 2d or ng,
                'do_wstacking': Correct for w term in imaging_context='ng' e.g. False,
                'algorithm': Clean algorithm, 'hogbom', 'mmclean', 'mfsmsclean',
                'fractional_threshold': Fractional of peak to end a major cycle e.g. 0.3,
                'gain': Loop gain e.g. 0.1,
                'niter': Number of clean interations per major cycle e.g. 1000,
                'nmajor': Number of major cycles e.g. 8,
                'nmoment': Number of moments for MSMFS algorithm e.g. 2,
                'scales': Scales for MSMFS algorithm e.g. [0, 3, 10],
                'threshold': Absolute threshold to stop all cleaning e.g. 0.003,
                'weighting_algorithm': 'natural' or 'uniform',
                'window_shape': 'quarter' or 'no_edge',
                'do_selfcal': No self-calibration at the end of each major cycle e.g. True,
                'calibration_context': Jones terms to solve for e.g. 'TG',
                'global_solution': Is the solution across all frequencies e.g. True,
                'T_first_selfcal': First major cycle to perform T selfcalibration e.g. 2,
                'T_phase_only': Phase only solution? e.g. True
                'T_timeslice': Solution interval 'auto' or time in seconds
                'G_first_selfcal': First major cycle to perform G selfcalibration e.g. 5,
                'G_phase_only': False,
                'G_timeslice': Solution interval 'auto' or time in seconds e.g. 1200,
                'B_first_selfcal': First major cycle to perform B selfcalibration e.g. 8,
                'B_phase_only': False,
                'B_timeslice': Solution interval 'auto' or time in seconds e.g 100000.0,
                'tol': Tolerance for gain solution e.g. 1e-8,
                'verbose': False,
            },
   
    
    """
    log.info("Creating template images")
    model_list = [rsexecute.execute(create_image_from_visibility, nout=1)
                  (bvis,
                   npixel=erp_params["process"]["npixel"],
                   cellsize=erp_params["process"]['cellsize'],
                   nchan=1,
                   frequency=[numpy.mean(bvis.frequency)],
                   channel_bandwidth=[numpy.sum(bvis.channel_bandwidth)])
                  for bvis in bvis_list]
    
    model_list = rsexecute.persist(model_list)
    
    log.info("Applying {} weighting".format(erp_params['process']['weighting_algorithm']))
    
    bvis_list = weight_list_rsexecute_workflow(bvis_list, model_list,
                                               weighting=erp_params['process']['weighting_algorithm'])
    
    bvis_list = rsexecute.persist(bvis_list)
    
    controls = create_calibration_controls()
    controls['T']['first_selfcal'] = erp_params["process"]["T_first_selfcal"]
    controls['T']['phase_only'] = erp_params["process"]["T_phase_only"]
    controls['T']['timeslice'] = erp_params["process"]["T_timeslice"]
    controls['G']['first_selfcal'] = erp_params["process"]["G_first_selfcal"]
    controls['G']['phase_only'] = erp_params["process"]["G_phase_only"]
    controls['G']['timeslice'] = erp_params["process"]["G_timeslice"]
    controls['B']['first_selfcal'] = erp_params["process"]["B_first_selfcal"]
    controls['B']['phase_only'] = erp_params["process"]["B_phase_only"]
    controls['B']['timeslice'] = erp_params["process"]["B_timeslice"]
    
    log.info("Processing with RASCIL ICAL pipeline")
    
    results = ical_list_rsexecute_workflow(
        bvis_list, model_list,
        context=erp_params['process']['imaging_context'],
        nmajor=erp_params["process"]['nmajor'],
        niter=erp_params["process"]['niter'],
        algorithm=erp_params["process"]['algorithm'],
        nmoment=erp_params["process"]['nmoment'],
        scales=erp_params["process"]['scales'],
        gain=erp_params["process"]['gain'],
        fractional_threshold=erp_params["process"]['fractional_threshold'],
        threshold=erp_params["process"]['threshold'],
        window_shape=erp_params["process"]['window_shape'],
        calibration_context=erp_params["process"]['calibration_context'],
        do_selfcal=erp_params["process"]['do_selfcal'],
        do_wstacking=erp_params["process"]['do_wstacking'],
        global_solution=erp_params["process"]['global_solution'],
        tol=erp_params["process"]['tol'],
        controls=controls)
    
    return bvis_list, results


def stage(erp_params, bvis_list, results):
    """ Stage the results

    :param erp_params:
    :param results: Results from ICAL (not a graph!)
    :return:
    
    The relevant parameters are::
    
        'stage':
            {
                'msout': Name of output MeasurementSet,
                'write_moments': True or False (write moments instead of spectral cubes),
                'number_moments': number of moments to write,
                'results_directory': Directory to write results e.g. 'results/'
                'verbose': False
            }
        }


    """
    log.info("Writing images")
    results_directory = erp_params["stage"]["results_directory"]
    
    im_types = ["deconvolved", "residual", "restored"]
    
    pipeline = "ical"
    
    def write_image(im, im_type="restored", index=0, axis='spw'):
        filename_root = \
            "{results_directory}/{project:s}_{source:s}_{pipeline}_{im_type:s}_{axis}{index:d}".format(
                results_directory=results_directory,
                project=erp_params["configure"]["project"],
                source=erp_params["ingest"]["source"],
                im_type=im_type,
                pipeline=pipeline,
                axis=axis,
                index=index)
        log.info(qa_image(im, context=filename_root))
        plt.clf()
        show_image(im, title=filename_root)
        plotfile = "{0}.png".format(filename_root)
        plt.savefig(plotfile)
        plt.show(block=False)
        filename = "{0}.fits".format(filename_root)
        export_image_to_fits(im, filename)
    
    # Write moment images
    if erp_params["stage"]["write_moments"]:
        nspw = len(results[0])
        nmoment = erp_params["stage"]["number_moments"]
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
    
    log.info("Writing gaintables")
    results_directory = erp_params["stage"]["results_directory"]
    
    gt_list = results[3]
    for igt, gt in enumerate(gt_list):
        for cc in erp_params["process"]['calibration_context']:
            filename_root = \
                "{results_directory}/{project:s}_{origin:d}_{mode}".format(
                    results_directory=results_directory,
                    project=erp_params["configure"]["project"],
                    origin=igt,
                    mode='ical')
            log.info(qa_gaintable(gt[cc],
                                  context="{0} {1}".format(filename_root, cc)))
            plt.clf()
            gaintable_plot(gt[cc], title='Frequency window {igt}'.format(igt=igt), cc=cc)
            plotfile = "{0}_{1}.png".format(filename_root, cc)
            plt.tight_layout()
            plt.savefig(plotfile)
            plt.show(block=False)
            
            gtfile = "{0}_{1}.hdf5".format(filename_root, cc)
            export_gaintable_to_hdf5(gt[cc], gtfile)
    
    log.info("Applying calibration")
    
    cvis_list = list()
    for bvis in bvis_list:
        cvis = None
        for gt in gt_list:
            cvis = bvis
            for cc in erp_params["process"]['calibration_context']:
                cvis = apply_gaintable(cvis, gt[cc])
        cvis_list.append(cvis)
    
    assert len(cvis_list) == len(bvis_list)
    bvis_list = cvis_list
    log.info("Combining across spectral windows")
    bvis_list = [concatenate_blockvisibility_frequency(bvis_list)]
    
    ms_out = erp_params["stage"]['msout']
    log.info("Writing Measurement Set {0}".format(ms_out))
    
    export_blockvisibility_to_ms(ms_out, bvis_list)
    
    return True
