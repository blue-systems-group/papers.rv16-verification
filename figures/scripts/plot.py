#!/usr/bin/env python

import json
import os
import collections
import gzip
import re
import argparse
import cPickle
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn import linear_model

from matplotlib.artist import setp

DATA_CACHE = '.cache.json'

import common


MIN_PKT_COUNT = 100


class BoxFigureBase(common.FigureBase):

  def __init__(self, args):
    super(BoxFigureBase, self).__init__(width=7.0/3, height=7.0/3/3*2)
    self.args = args

  def get_key(self, pr_ds, pr_es, pr_ed):
    pass

  def get_value(self, value):
    pass

  def scatter_plot(self):
    X, Y = [], []
    for key, value in self.args.data.items():
      if value['type_0_tx'] < MIN_PKT_COUNT:
        continue
      X.append(key[0]+key[1])
      Y.append(value['process_duration'] / value['trace_duration'])

    ax = self.add_subplot(111)

    ax.scatter(X, Y, s=2, linewidth=0, color='k')
    return ax

  def post_process(self, data):
    return data


  def box_plot(self):
    data = dict()
    for key, value in self.args.data.items():
      if value['type_0_tx'] < MIN_PKT_COUNT:
        continue
      k = self.get_key(key[0], key[1], key[2])
      if k is None:
        continue
      k = '%.2f' % (k)
      if k not in data:
        data[k] = []
      v = self.get_value(value)
      if v is not None:
        data[k].append(v)

    data = self.post_process(data)

    ax = self.add_subplot(111)
    X = sorted(data.keys(), key=lambda t: float(t))
    Y = [data[x] for x in X]
    lines = ax.boxplot(Y, sym='', whis=[10, 90], widths=0.4, patch_artist=True)

    setp(lines['medians'], color='k', linewidth=1)
    setp(lines['boxes'], color='grey', linewidth=0.5)
    setp(lines['whiskers'], linewidth=0.5, color='grey', linestyle='solid')
    setp(lines['caps'], linewidth=0.5, color='grey')

    xmax = max([float(x) for x in X])
    if xmax > 1:
      step = 0.5
    else:
      step = 0.1

    xticks = [X.index('%.2f' % (x))+1 for x in np.arange(0, xmax+0.01, step)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(['%.1f' % (x) for x in np.arange(0, xmax+0.01, step)])

    ax.set_xlabel(self.xlabel)
    ax.set_ylabel(self.ylabel)
    return ax

  def plot(self):
    self.box_plot()
    self.save()




class Type1TxRatioFigure(BoxFigureBase):

  @property
  def xlabel(self):
    return '\\textbf{$Pr_{ds}$}'

  @property
  def ylabel(self):
    return '\\textbf{Type-1 Tx Ratio}'

  def get_key(self, pr_ds, pr_es, pr_ed):
    return pr_ds

  def get_value(self, value):
    return float(value['type_1_tx']) / (value['type_1_tx'] + value['type_0_tx'])

  def plot(self):
    ax = self.box_plot()
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    self.save()





class Type1RxRatioFigure(BoxFigureBase):

  @property
  def xlabel(self):
    return '\\textbf{$Pr_{es}(1-P_{ed})$}'

  @property
  def ylabel(self):
    return '\\textbf{Type-1 Rx Ratio}'

  def get_key(self, pr_ds, pr_es, pr_ed):
    return pr_es * (1 - pr_ed)

  def get_value(self, value):
    return float(value['type_1_rx']) / (value['type_1_rx'] + value['type_0_rx'])

  def plot(self):
    ax = self.box_plot()
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    self.save()



class Type2RatioFigure(BoxFigureBase):

  @property
  def xlabel(self):
    return '\\textbf{$Pr_{ed}(1-Pr_{es})$}'

  @property
  def ylabel(self):
    return '\\textbf{Type-2 Ratio}'

  def get_key(self, pr_ds, pr_es, pr_ed):
    return pr_ed * (1 - pr_es)

  def get_value(self, value):
    return float(value['type_2']) / (value['type_0_rx'] + value['type_1_rx'] + value['type_2'])

  def plot(self):
    ax = self.box_plot()
    ax.set_ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.61, 0.1))
    self.save()



class ProcessTimeFigure(BoxFigureBase):

  @property
  def xlabel(self):
    return '\\textbf{$Pr_{ds} + Pr_{es} + Pr_{ed}$}'

  @property
  def ylabel(self):
    return '\\textbf{Processing Speed} (Pkt/Sec)'

  def get_key(self, pr_ds, pr_es, pr_ed):
    return pr_ds + pr_es + pr_ed

  def get_value(self, value):
    total_packets = sum(value['pkt_step_counts'].values())
    return float(total_packets) / value['process_duration']


class StepCountFigure(BoxFigureBase):

  @property
  def xlabel(self):
    return '\\textbf{$Pr_{ds} + Pr_{es} + Pr_{ed}}'

  @property
  def ylabel(self):
    return 'Average Step Per Packet'

  def get_key(self, pr_ds, pr_es, pr_ed):
    return pr_ds + pr_es + pr_ed

  def get_value(self, value):
    total_packets = sum(value['pkt_step_counts'].values())
    total_steps = sum([int(s)*c for s, c in value['pkt_step_counts'].items()])
    return float(total_steps)/total_packets


class FigureBase3D(common.FigureBase):

  def __init__(self, args):
    super(FigureBase3D, self).__init__(width=7.0/3, height=7.0/3/3*2)
    self.args = args

  def get_key(self, pr_ds, pr_es, pr_ed):
    if float('%.2f' % (pr_ed)) in self.fix:
      return '%.2f_%.2f' % (pr_ds, pr_es)
    return None

  def get_data(self):
    raw_data = dict()
    for key, value in self.args.data.items():
      if value['type_0_tx'] < MIN_PKT_COUNT:
        continue
      k = self.get_key(key[0], key[1], key[2])
      if k is None:
        continue
      if k not in raw_data:
        raw_data[k] = []
      try:
        raw_data[k].append(self.get_value(value))
      except:
        print key
        print value
        raise

    return raw_data

  def post_process(self, raw_data):
    data = dict()
    for k, v in raw_data.items():
      data[k] = np.mean(v)
    return data

  def plot(self):
    for fix in [[0.1, 0, 0.05, 0.15], [0.3, 0.20, 0.25, 0.35], [0.5, 0.45, 0.40]]:
      self.fix = fix

      data = self.get_data()
      data = self.post_process(data)

      X = np.arange(0, 0.51, 0.05)
      Y = np.arange(0, 0.51, 0.05)
      Z = np.asarray([[data.get('%.2f_%.2f' % (x, y), np.nan) for x in X] for y in Y])
      X, Y = np.meshgrid(X, Y)

      self.add_figure()
      ax = self.fig.gca(projection='3d')
      self.fig.gca().invert_xaxis()
      ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
          linewidth=0, antialiased=False)

      ax.set_zlim3d(0, self.ylim)
      ax.set_zticks(np.arange(0, self.ylim+0.01, self.ystep))

      ax.xaxis._axinfo['label']['space_factor'] = 2.8
      ax.yaxis._axinfo['label']['space_factor'] = 2.8
      ax.zaxis._axinfo['label']['space_factor'] = 2.8
      ax.set_xlabel('\\textbf{$P_{ds}$}')
      ax.set_ylabel('\\textbf{$P_{es}$}')

      for axis in ['x', 'y', 'z']:
        for tick in getattr(ax, '%saxis' % (axis)).get_major_ticks():
          tick.label.set_fontsize(6)
          tick.set_pad(3)

      self.save('%s_%s' % (self.__class__.__name__, ('%.2f' % fix[0]).replace('.', '_')))


class PrecisionFigure(common.FigureBase):

  def __init__(self, args):
    super(PrecisionFigure, self).__init__()
    self.args = args
    self.thresholds = np.arange(0.1, 0.31, 0.05)

  def get_data(self):
    raw = dict()
    for key, value in self.args.data.items():
      pr_ds, pr_es, pr_ed = key[:3]
      k = '%.2f' % (pr_ed)
      if k not in raw:
        raw[k] = []
      raw[k].append((value['violating_pkt'], value['true_violating_pkt'], value['max_thres'][0]))

    precision = dict((t, dict()) for t in self.thresholds)
    recall = dict((t, dict()) for t in self.thresholds)
    for k, v in sorted(raw.items(), key=lambda t: float(t[0])):
      if len(v) < 50:
        continue

      true_violations = [t for t in v if t[1] != -1]

      for thres in self.thresholds:
        reported_violations = [t for t in v if t[0] != -1 or t[2] > thres]
        p = len([t for t in reported_violations if t in true_violations])/float(len(reported_violations))
        r = len([t for t in true_violations if t in reported_violations])/float(len(true_violations))
        print '%s[%.2f]: total: %d, reported: %d, true: %d, precision: %.2f, recall: %.2f' % (k, thres, len(v), len(reported_violations), len(true_violations), p, r)
        precision[thres][k] = p
        recall[thres][k] = r
    return precision, recall


  def plot(self):
    precision, recall = self.get_data()

    ax = self.add_subplot(111)
    for thres in self.thresholds:
      pres = precision[thres]
      X = sorted(pres.keys(), key=lambda t: float(t))
      P = [pres[x] for x in X]
      X = [float(x) for x in X]
      self.plot_cdf(ax, P, color=self.color, marker=self.marker, markersize=5, label='$k=%d$' % (int(thres*100)))

    ax.set_ylim(0, 1.01)
    ax.set_xlim(0.5, 1.01)

    ax.set_xlabel('\\textbf{Precision}')
    ax.legend(loc='upper left', **self.legend_kwargs)
    self.save()

    self.add_figure()
    ax = self.add_subplot(111)
    for thres in self.thresholds:
      rec = recall[thres]
      X = sorted(pres.keys(), key=lambda t: float(t))
      R = [rec[x] for x in X]
      X = [float(x) for x in X]
      self.plot_cdf(ax, R, color=self.color, marker=self.marker, markersize=5, label='$k=%d$' % (int(thres*100)))


    ax.set_ylim(0, 1.01)
    ax.set_xlim(0.5, 1.01)

    ax.set_xlabel('\\textbf{Recall}')
    ax.legend(loc='upper left', **self.legend_kwargs)
    self.save('RecallFigure')


class MutationSnifferJaccard3DFigure(FigureBase3D):

  @property
  def ylim(self):
    return 0.5

  @property
  def ystep(self):
    return 0.1

  def get_value(self, value):
    augmented_transitions = value['type_1_tx'] + value['type_1_rx'] + value['type_2']
    original_transitions = value['type_0_tx'] + value['type_0_rx']
    return float(augmented_transitions)/(augmented_transitions + original_transitions)


class MutationDUTJaccard3DFigure(FigureBase3D):

  @property
  def ylim(self):
    return 0.3

  @property
  def ystep(self):
    return 0.1

  def get_value(self, value):
    return value['jaccard']


class ProcessTime3DFigure(FigureBase3D):

  @property
  def ylim(self):
    return 5000

  @property
  def ystep(self):
    return 1000

  def get_value(self, value):
    total_packets = sum(value['pkt_step_counts'].values())
    return float(total_packets) / value['process_duration']


class StepCount3DFigure(FigureBase3D):

  @property
  def ylim(self):
    return 1000

  @property
  def ystep(self):
    return 20

  def get_value(self, value):
    total_packets = sum(value['pkt_step_counts'].values())
    total_steps = sum([int(s)*c for s, c in value['pkt_step_counts'].items()])
    return float(total_steps)/total_packets


class ConsecAug3DFigure(FigureBase3D):

  @property
  def ylim(self):
    return 16

  @property
  def ystep(self):
    return 4

  def get_value(self, value):
    if len(value['cost_count']) == 0:
      return 0
    else:
      return max([int(k) for k in value['cost_count'].keys()])

class JaccardFigureBase(BoxFigureBase):

  @property
  def ylabel(self):
    return '\\textbf{Jaccard Distance}'

  def get_key_overall(self, pr_ds, pr_es, pr_ed):
    return pr_ds + pr_es + pr_ed

  def get_key_pr_ds(self, pr_ds, pr_es, pr_ed):
    if pr_es == self.fix and pr_ed == self.fix:
      return pr_ds
    return None

  def get_key_pr_es(self, pr_ds, pr_es, pr_ed):
    if pr_ds == self.fix and pr_ed == self.fix:
      return pr_es
    return None

  def get_key_pr_ed(self, pr_ds, pr_es, pr_ed):
    if pr_ds == self.fix and pr_es == self.fix:
      return pr_ed
    return None

  def plot(self):
    self.fix = 0.1

    for xlabel, key_func, suffix, ylim in [\
        ('\\textbf{$Pr_{ds} + Pr_{es} + Pr_{ed}$}', self.get_key_overall, '', 0.5),\
        ('\\textbf{$Pr_{ds}$}', self.get_key_pr_ds, 'VaryPrds', 0.5),\
        ('\\textbf{$Pr_{es}$}', self.get_key_pr_es, 'VaryPres', 0.5),\
        ('\\textbf{$Pr_{ed}$}', self.get_key_pr_ed, 'VaryPred', 0.5),\
        ]:
      self.xlabel = xlabel
      self.get_key = key_func
      self.add_figure()
      ax = self.box_plot()
      ax.set_ylim(0, ylim)
      ax.set_yticks(np.arange(0, ylim+0.1, 0.1))
      self.save('%s%s' % (self.__class__.__name__, suffix))


class MutationDUTJaccardFigure(JaccardFigureBase):

  def get_value(self, value):
    return value['jaccard']


class MutationSnifferJaccardFigure(JaccardFigureBase):



  def get_value(self, value):
    augmented_transitions = value['type_1_tx'] + value['type_1_rx'] + value['type_2']
    original_transitions = value['type_0_tx'] + value['type_0_rx']
    return float(augmented_transitions)/(augmented_transitions + original_transitions)


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', required=True)
  parser.add_argument('--force', action='store_true', default=False)
  return parser


FILENAME_PATTERN = re.compile(r"""^(P?<pr_ds>[\d\.]*)-(P?<pr_es>[\d\.]*)-(P?<pr_ed>[\d\.]*)-(P?<run>\d*)""")


def calc_jaccard_distance(pr_ds, pr_es, pr_ed, run, args):
  dut_trace_file = os.path.join(args.data_dir, '%.2f-%.2f-%.2f-%d-1-0_mutation.gz' % (pr_ds, pr_es, pr_ed, run))
  sniffer_trace_file = os.path.join(args.data_dir, '%.2f-%.2f-%.2f-%d-2-0_mutation.gz' % (pr_ds, pr_es, pr_ed, run))

  with gzip.open(dut_trace_file, 'rb') as f:
    dut_trace = set([l.strip() for l in f.readlines()])
  with gzip.open(sniffer_trace_file, 'rb') as f:
    sniffer_trace = set([l.strip() for l in f.readlines()])

  return float(len(dut_trace ^ sniffer_trace))/len(dut_trace | sniffer_trace)


def get_true_violating_pkt(pr_ds, pr_es, pr_ed, run, args):
  dut_report_file = os.path.join(args.data_dir, '%.2f-%.2f-%.2f-%d-1-0_report.json.gz' % (pr_ds, pr_es, pr_ed, run))
  with gzip.open(dut_report_file, 'rb') as f:
    data = json.loads(f.read())
    return data['violating_pkt']


def dot11_load_data(args):
  data = dict()
  for filename in os.listdir(args.data_dir):
    if not filename.endswith('2-0_report.json.gz'):
      continue
    path = os.path.join(args.data_dir, filename)
    print 'Loading %s' % (path)

    parts = filename.split('-')
    pr_ds = float(parts[0])
    pr_es = float(parts[1])
    pr_ed = float(parts[2])
    run = int(parts[3])
    key = (pr_ds, pr_es, pr_ed, run)

    with gzip.open(path, 'rb') as f:
      entry = json.loads(f.read())

    if 'violating_pkt' in entry:
      try:
        entry['true_violating_pkt'] = get_true_violating_pkt(pr_ds, pr_es, pr_ed, run, args)
      except:
        print 'Failed to process %s' % (path)
        continue
    try:
      entry['jaccard'] = calc_jaccard_distance(pr_ds, pr_es, pr_ed, run, args)
    except:
      continue
    data[key] = entry

  args.data = data


def xbox_load_data(args):
  data = dict()
  for filename in os.listdir(args.data_dir):
    path = os.path.join(args.data_dir, filename)
    if not path.endswith('report.json.gz'):
      continue

    print 'Loading %s' % (path)

    key = path.replace('_report.json.gz', '')
    with gzip.open(path, 'rb') as f:
      entry = json.loads(f.read())
    data[key] = entry

  args.data = data


def load_data(args):
  cache_file = '.cache_%s' % (args.data_dir.replace(os.sep, '_'))
  if os.path.isfile(cache_file) and not args.force:
    print 'Loading from cache...'
    with gzip.open(cache_file, 'rb') as f:
      args.data = cPickle.load(f)
    return

  if 'dot11' in args.data_dir:
    dot11_load_data(args)
  elif 'xbox' in args.data_dir:
    xbox_load_data(args)

  print 'Saving data cache...'
  with gzip.open(cache_file, 'wb') as f:
    cPickle.dump(args.data, f)



FIGURES = [
    # Type1TxRatioFigure,
    # Type1RxRatioFigure,
    # Type2RatioFigure,
    # MutationSnifferJaccard3DFigure,
    # MutationDUTJaccard3DFigure,
    # ProcessTime3DFigure,
    # StepCount3DFigure,
    # ConsecAug3DFigure,
    # PrecisionFigure,
    ]


def xbox_analysis(args):
  monitor_traces = collections.Counter()
  monitor_violations = collections.Counter()
  dragon_deep_fades = collections.Counter()
  phoenix_deep_fades = collections.Counter()
  for key, value in args.data.items():
    monitor = key.split('_')[-1]
    monitor_traces[monitor] += 1
    if value['violating_pkt'] != -1:
      monitor_violations[monitor] += 1
    if monitor == 'DragonDeepFade':
      dragon_deep_fades[len(value['deepfades'])] += 1
      if value['violating_pkt'] > 0:
        print '%s: %d' % (key, value['violating_pkt'])
    if monitor == 'PhoenixDeepFade':
      phoenix_deep_fades[len(value['deepfades'])] += 1
      if value['violating_pkt'] > 0:
        print '%s: %d' % (key, value['violating_pkt'])

  for monitor in monitor_traces:
    print '%s: %d / %d (%.2f) violations' % (monitor, monitor_violations[monitor], monitor_traces[monitor], 100.0*monitor_violations[monitor]/monitor_traces[monitor])

  print 'Dragon deep fades: %s' % (dragon_deep_fades)
  print 'Phoenix deep fades: %s' % (phoenix_deep_fades)


def main():
  args = arg_parser().parse_args()
  args.data_dir = os.path.abspath(args.data_dir)

  load_data(args)
  if 'xbox' in args.data_dir:
    xbox_analysis(args)
  else:
    for fig in FIGURES:
      fig(args).plot()


if __name__ == '__main__':
  main()
