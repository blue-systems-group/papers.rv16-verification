import os
import sys
import logging
import collections
import traceback
import numpy as np
from itertools import cycle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, rc

TABLEAU10 = {
    'blue': (31, 119, 180),
    'orange': (255, 127, 14),
    'green': (44, 160, 44),
    'red': (214, 39, 40),
    'purple': (148, 103, 189),
    'pink': (227, 119, 194),
    'grey': (127, 127, 127),
    'yellow': (188, 189, 34),
    'cyan': (23, 190, 207),
    }

for name, rgb in TABLEAU10.items():
  r, g, b  = rgb
  TABLEAU10[name] = (r/255.0, g/255.0, b/255.0)



def pool_worker(f):
  def wrapper(*args, **kwargs):
    try:
      f(*args, **kwargs)
    except:
      raise Exception("".join(traceback.format_exception(*sys.exc_info())))
  return wrapper


class FigureBase(object):

  def __init__(self, basedir=os.path.dirname(os.path.realpath(__file__)), width=3.3, height=2.5):
    try:
      os.makedirs(basedir)
    except:
      pass

    self.basedir = basedir
    self.width = width
    self.height = height
    self.legend_kwargs = {'numpoints': 1, 'markerscale': 1.2, 'fontsize': 'small'}
    self.plot_kwargs = {'linewidth': 0.3, 'markersize': 3, 'markeredgewidth': 0}

    self.add_figure()

    logging.basicConfig(format='[%(asctime)s] %(levelname)s [%(filename)32s:%(lineno)4d] %(message)s', level=logging.DEBUG)
    self.logger = logging.getLogger('FigureBase')

    self.logger.debug("Plotting %s" % (self.__class__.__name__))

  @property
  def marker(self):
    return next(self.markers)

  @property
  def color(self):
    return next(self.colors)

  def add_figure(self):
    self.fig = plt.figure()
    self.markers = cycle('osD^')
    self.colors = cycle(TABLEAU10.values())
    self.set_font()

  def add_subplot(self, *args, **kwargs):
    ax = self.fig.add_subplot(*args, **kwargs)
    self.thin_border(ax)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # for tick in ax.xaxis.get_major_ticks():
      # tick.label.set_fontsize(8)
    return ax

  def set_font(self):
    rc('font', **{'family': 'serif', 'serif': ['Times'], 'size':'10'})
    rc('text', usetex=True)

  def save(self, name=None, extension='.pdf', **kwargs):
    if name is None:
      name = self.__class__.__name__

    self.fig.set_size_inches(self.width, self.height)
    path = os.path.join(self.basedir, name + extension)
    self.fig.savefig(path, bbox_inches='tight', **kwargs)


  def plot(self, *args, **kwargs):
    self.logger.error("plot for %s is not implemented." % (self.__class__.__name__))

  def thin_border(self, ax, lw=0.5):
    for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(lw)

  def plot_cdf(self, ax, data, *args, **kwargs):
    count = collections.Counter(data)
    X = sorted(count.keys())
    Y = np.divide(np.cumsum([count[x] for x in X]).astype(float), sum(count.values())).tolist()
    if len(X) == 1:
      X = [X[0]-1e-6, X[0]]
      Y = [0.01, Y[0]]

    ax.plot(X, Y, *args, **kwargs)

    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel('\\textbf{CDF}')
    ax.grid(True)

    self.thin_border(ax)

  def escape(self, s):
    for special in ['_', '&']:
      s = ('\\'+special).join(s.split(special))
    return s
