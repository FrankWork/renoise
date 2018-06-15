import re
import time
import sys
import os
import argparse


home_dir = os.environ['HOME']

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="", help='')
parser.add_argument('--output_dir', default="", help='')
parser.add_argument('--max_len', default=220, type=int, help='max length of sentence')
parser.add_argument('--max_instance', default=100, type=int, help='max instance per chunk')
args = parser.parse_args()

def clean_str(line):
  line = line.lower()
  line = re.sub('-lrb-', ' ', line)
  line = re.sub('-rrb-', ' ', line)
  line = re.sub("[&'-\._]", ' ', line)
  line = re.sub('###end###', ' ', line)
  line = re.sub('\\\/', ' ', line)  # remove '\/'
  line = re.sub(r'\\\*', ' ', line) # remove '\*'
  line = re.sub(' {2,}', ' ', line)
  line = line.strip()
  return line

def write_chunks(sentences, dir_name, max_instance=args.max_instance):
    '''sharding sentences into small files
    '''
    sentences = list(sentences)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)   

    n_shards = 0
    for i in range(0, len(sentences), max_instance):
        filename = os.path.join(dir_name, "data.%d" %(n_shards))
        with open(filename, 'w') as f:
            for sent in sentences[i:i+max_instance]:
                f.write(sent+'\n')
            n_shards+=1
    print("write %d files" % n_shards)

def get_sentence_from_data(in_dir, out_dir, basename='train'):
  in_file = "%s.txt" % basename
  print("extract  sentences from %s"   % in_file)
  in_file = os.path.join(in_dir, in_file)

  len60, len100,  len200, len3000 = set(), set(), set(), set()
  max_len = 0
  ignored = 0
  with open(in_file) as f:
    for line in f:
      parts = line.strip().split('\t')
      sentence = parts[5]
      sentence = clean_str(sentence)
      tokens = sentence.split()
      n = len(tokens)
      if args.max_len > 0 and n > args.max_len:
          ignored += 1
          continue

      max_len = max(max_len, n)
      if n <= 60:
        len60.add(sentence)
      elif n>60 and n<=100:
        len100.add(sentence)
      elif n>100 and n<=200:
        len200.add(sentence)
      elif n>200 and n<=220:
        len3000.add(sentence)
  
  print('max_len %d' % max_len)
  print('<=60: %d (60,100]: %d (100, 200]: %d >200: %d'   % (len(len60), len(len100), len(len200), len(len3000)))
  print('ignored: %d' % ignored)

  out_dir = os.path.join(out_dir, basename)
  write_chunks(len60, out_dir+"/len60")
  write_chunks(len100, out_dir+"/len100")
  write_chunks(len200, out_dir+"/len200")
  write_chunks(len3000, out_dir+"/len3000")

  sys.stdout.flush()

get_sentence_from_data(args.input_dir, args.output_dir, "train")
get_sentence_from_data(args.input_dir, args.output_dir, "test")

# arr = ["$data_dir/train.%d" %i for i in range(0,21)]
# arr = ["$data_dir/test.%d" %i for i in range(0,4)]
# " ".join(arr)








# > python extract_raw_text.py --input_dir data/RE/ --output_dir ~/tmp/
# extract  sentences from train.txt
# max_len 10124
# <=60: 264420 (60,100]: 9887 (100, 200]: 420 >200: 1
# ignored: 0
# write 2645 files
# write 99 files
# write 5 files
# write 1 files
# extract  sentences from test.txt
# max_len 214
# <=60: 43988 (60,100]: 1617 (100, 200]: 71 >200: 1
# ignored: 0
# write 440 files
# write 17 files
# write 1 files
# write 1 files


# > python extract_raw_text.py --input_dir data/RE/ --output_dir ~/tmp/ --max_len 220
# extract  sentences from train.txt
# max_len 216
# <=60: 264420 (60,100]: 9887 (100, 200]: 420 >200: 1
# ignored: 172
# write 2645 files
# write 99 files
# write 5 files
# write 1 files
# extract  sentences from test.txt
# max_len 214
# <=60: 43988 (60,100]: 1617 (100, 200]: 71 >200: 1
# ignored: 0
# write 440 files
# write 17 files
# write 1 files
# write 1 files

