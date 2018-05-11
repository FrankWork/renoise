import re
import os

def clean_str(line):
  line = line.lower()
  line = re.sub('-lrb-', '(', line)
  line = re.sub('-rrb-', ')', line)
  line = re.sub("''", ' ', line)
  line = re.sub("_", ' ', line)
  line = re.sub('###end###', ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(' {2,}', ' ', line)
  line = line.strip()
  return line

def get_stp_map():
  stp_txt = []
  for file in ['train.stp.txt', 'test.stp.txt', 'todo.stp.txt']:#
    with open(file) as f:
      for line in f:
        line = re.sub("_", ' ', line)
        line = line.strip()
        stp_txt.append(line)

  stp_lex = []
  for file in ['train.stp', 'test.stp', 'todo.stp']:#
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
  # orig_set = set()
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
  
  # for txt in orig_set:
  #   if txt not in stp_map:
  #     print(txt)
  of.close()

align("tmp_test.txt", 'test.stp.align')
align("tmp_train.txt", 'train.stp.align')
