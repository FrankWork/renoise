import re
import os

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

def get_stp_map():
  stp_txt = []
  for file in ['train.uniq', 'test.uniq']:
    with open(file) as f:
      for line in f:
        line = line.strip()
        stp_txt.append(line)

  stp_lex = []
  for file in ['train.uniq.stp', 'test.uniq.stp']:
    tmp = []
    with open(file) as f:
      for line in f:
        tmp.append(line)
        if line == '\n':
          stp_lex.append("".join(tmp))
          tmp.clear()

  assert len(stp_txt) == len(stp_lex)

  stp_map = {}
  for txt, lex in zip(stp_txt, stp_lex):
    stp_map[txt]=lex
  return stp_map



stp_map = get_stp_map()

def align(in_file, out_file):
  of = open(out_file, 'w')
  with open(in_file) as f:
    for line in f:
      parts = line.strip().split('\t')
      sentence = parts[5]
      sentence = clean_str(sentence)
      # orig_set.add(sentence)
      lex = 'None\n\n'
      if sentence in stp_map:
        lex = stp_map[sentence]
      of.write(lex)
  
  of.close()

align("test.txt", 'test.stp')
align("train.txt", 'train.stp')
