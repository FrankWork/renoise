import os
import pickle
import argparse
import subprocess
import re
import time
import sys


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--memory', default=35, type=int, help='')
arg_parser.add_argument('--n_threads', default=20, type=int, help='')
arg_parser.add_argument('--n_files', default=0, type=int, help='')
arg_parser.add_argument('--length', default=60, type=int, help='')
arg_parser.add_argument('--basename', default="train", help='')
arg_parser.add_argument('--data_dir', default='', help='project dir')
arg_parser.add_argument('--parser_path', default='', help='')
arg_parser.add_argument('--out_dir', default='', help='')

args = arg_parser.parse_args()

out_dir = os.path.join(args.out_dir, args.basename, 'len%d'%args.length)
if not os.path.exists(out_dir):
  os.makedirs(out_dir)

data_dir = os.path.join(args.data_dir, args.basename, 'len%d'%args.length)
file_list = [os.path.join(data_dir, 'data.%d' % i) for i in range(args.n_files)]
files = ' '.join(file_list)

cmd_template = "java -mx%dg "\
      "-cp %s/stanford-parser.jar:%s/stanford-parser-3.9.1-models.jar "\
      "edu.stanford.nlp.parser.lexparser.LexicalizedParser "\
      "-sentences newline "\
      "-writeOutputFiles  "\
      '-outputFormat "typedDependencies" '\
      '-nthreads %d '\
      '-outputFilesDirectory %s '\
      'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz %s'

cmd = cmd_template % (args.memory, 
                      args.parser_path, args.parser_path,
                      args.n_threads,
                      out_dir,
                      files)
ret = subprocess.call(cmd, shell=True)
assert ret == 0
