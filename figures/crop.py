#!/usr/bin/python

import os
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('files', nargs='*', default=[os.getcwd(),], help="Files to crop")

args = parser.parse_args()

FIG_DIR = os.path.dirname(os.path.realpath(__file__))

devnull = open('/dev/null', 'w')

def crop_dir(dir):
  for file in os.listdir(FIG_DIR):
    crop_file(file)

def crop_file(file):
  name, extension = os.path.splitext(file)
  if 'logo' in name:
    print "Ignoring " + file
    return
  if extension == '.pdf':
    print "Croping " + file
    subprocess.check_call('pdfcrop %s %s' % (file, file), stdout=devnull, shell=True)
  elif extension in ['.jpg', '.png']:
    print "Croping " + file
    subprocess.check_call('convert -trim %s %s' % (file, file), stdout=devnull, shell=True)

for file in args.files:
  file_path = os.path.abspath(file)
  if os.path.isdir(file_path):
    crop_dir(file_path)
  else:
    crop_file(file_path)
