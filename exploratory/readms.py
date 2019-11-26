
import sys
import logging
import numpy

from processing_components.visibility.base import create_blockvisibility_from_ms

from processing_components.image.operations import show_image
import matplotlib.pyplot as plt

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

# %%

from matplotlib import pylab

pylab.rcParams['figure.figsize'] = (14.0, 14.0)
pylab.rcParams['image.cmap'] = 'Greys'

# %%

bvis_list = create_blockvisibility_from_ms('../data/3C277.1_avg.ms', datacolumn='MODEL_DATA')
sources = numpy.unique([bv.source for bv in bvis_list])
print(sources)

# %%
from processing_components.visibility.operations import integrate_visibility_by_channel
avis_list = [integrate_visibility_by_channel(bvis) for bvis in bvis_list]

# %%

from processing_components.imaging.base import advise_wide_field

advice = advise_wide_field(avis_list[0])

# %%

from processing_components.visibility.operations import concatenate_blockvisibility_frequency

blockvis = [concatenate_blockvisibility_frequency(avis_list[isource * 4:(isource * 4 + 4)])
            for isource, source in enumerate(sources)]

print(blockvis[0].source)

# %%

from processing_components.simulation.simulation_helpers import plot_uvcoverage, plot_visibility
plt.clf()
for svis in blockvis:
    plot_uvcoverage([svis], title='UV Coverage {source:s}'.format(source=svis.source))
plt.show(block=False)

plt.clf()
for svis in blockvis:
    plot_visibility([svis], title='Visibility amplitude {source:s}'.format(source=svis.source))
plt.show(block=False)

# %%

from processing_components.imaging.ng import invert_ng
from processing_components.visibility.operations import convert_blockvisibility_to_stokesI
from processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from processing_components.imaging.weighting import weight_visibility

from processing_components.imaging.base import create_image_from_visibility

for svis in blockvis:
    frequency = [numpy.mean(svis.frequency)]
    channel_bandwidth = [numpy.sum(svis.channel_bandwidth)]
    ivis = convert_blockvisibility_to_stokesI(svis)
    model = create_image_from_visibility(ivis, npixel=512, cellsize=advice['cellsize']/2.0, nchan=1,
                                         frequency=frequency, channel_bandwidth=channel_bandwidth)
    cvis = convert_blockvisibility_to_visibility(ivis)
    cvis = weight_visibility(cvis, model)
    ivis = convert_visibility_to_blockvisibility(cvis)
    dirty, sumwt = invert_ng(ivis, model, do_wstacking=False)
    plt.clf()
    show_image(dirty, title=svis.source + " Dirty image")
    plt.show(block=False)
    print(sumwt)
    psf, sumwt = invert_ng(ivis, model, do_wstacking=False, dopsf=True)
    plt.clf()
    show_image(dirty, title=svis.source + " PSF")
    plt.show(block=False)



