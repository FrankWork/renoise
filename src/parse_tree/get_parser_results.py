import os
import re

regex = re.compile("\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def restore_sentence_from_tree(lines):
  words = [0]*300
  n_words = 0
  for line in lines:
    if line != '\n':
      m=regex.search(line)
      if not m:
        print(line)
        exit()
      w1, idx1, w2, idx2 = m.groups()
      idx1 = int(idx1)
      idx2 = int(idx2)
      n_words = max(idx1, idx2, n_words)
      if idx1 > 0:
        words[idx1-1] = w1
      if idx2 > 0:
        words[idx2-1] = w2
  tmp = []
  for w in words[:n_words]:
    if w != 0:
      tmp.append(w)
  s = " ".join(tmp)
  s = re.sub(" _ ", "_", s)
  s = re.sub("\d", ' ', s)
  s = re.sub("[':&`_\.,!?=]", " ", s)
  s = re.sub("-", " ", s)
  s = re.sub(" ", " ", s)#'\xa0'
  s = re.sub("LRB|RRB", " ", s)

  s = re.sub("am|pm|an|to", ' ', s)
  s = re.sub(r'\\\*', ' ', s) # replace '\*'
  s = re.sub(' {2,}', ' ', s)
  s = s.strip()

  return s+'\n'

def is_tree_of(tree, txt, keys):
  # txt = txt.split()

  tree = tree.split()
  for w in keys:
    if w not in tree:
      return False

  for w in tree:
    if w not in txt:
      return False
  return True

def clean_str(line):
  line = line.strip()
  line = re.sub("[\(\):?!;\-'&`_\.,=]", ' ', line)
  line = re.sub("\.\.\.", ' ', line)
  line = re.sub("''|--", ' ', line)
  line = re.sub("\d", ' ', line)
  line = re.sub("am|pm|an|to", ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' in line
  line = re.sub(r'\\\*', ' ', line) # replace '\*'
  line = re.sub(' {2,}', ' ', line)
  line = line.strip()
  return line+'\n'

def get_results(txt_dirs, stp_dirs, file_type):
  all_txt = set()
  all_stp = {}

  for txt_dir in txt_dirs:
    for txt_file in os.listdir(txt_dir):
      with open(os.path.join(txt_dir, txt_file)) as f:
        sentences = f.readlines()
        all_txt.update(sentences)
        # all_txt.extend([clean_str(sent) for sent in sentences])
  for stp_dir in stp_dirs:
    for stp_file in os.listdir(stp_dir):
      with open(os.path.join(stp_dir, stp_file)) as f:
        tmp = []
        for line in f:
          tmp.append(line)
          if line == '\n':
            # stp.append("".join(tmp))
            s = restore_sentence_from_tree(tmp)
            all_stp[s] = "".join(tmp)
            tmp.clear()
  
  f_txt = open('%s.stp.txt'%file_type, 'w')
  f_stp = open('%s.stp'%file_type, 'w')
  f_txt_todo = open('%s.todo'%file_type, 'w')
  for txt in all_txt:
    s = clean_str(txt)
    if s in all_stp:
      f_txt.write(txt)
      f_stp.write(all_stp[s])
    else:
      f_txt_todo.write(txt)

  # stp_set = set(all_stp)
  # txt_set = set(all_txt)
  # with open('%s.stp.txt'%file_type, 'w') as f:
  #   for line in sorted(list(txt_set.difference(stp_set))):
  #     f.write(line)
  # with open('%s.stp'%file_type, 'w') as f:
  #   for line in sorted(stp_set.difference(txt_set)):
  #     f.write(line)


# get_results(
#   ['tmp_test/len60', 'tmp_test/len100','tmp_test/len200','tmp_test/len3000'],
#   ['tmp_test/len60_lex', 'tmp_test/len100_lex','tmp_test/len200_lex','tmp_test/len3000_lex'],
#   'test')

# get_results(
#   ['tmp_train/len60', 'tmp_train/len100','tmp_train/len200','tmp_train/len3000'],
#   ['tmp_train/len60_lex', 'tmp_train/len100_lex','tmp_train/len200_lex','tmp_train/len3000_lex'],
#   'train')


def get_results_v2(txt_dirs, stp_dirs, file_type):
  all_txt = []
  all_stp = []

  for txt_dir in txt_dirs:
    for txt_file in sorted(os.listdir(txt_dir)):
      # all_txt.append(txt_file+'\n')
      with open(os.path.join(txt_dir, txt_file)) as f:
        sentences = f.readlines()
        # all_txt.update(sentences)
        # all_txt.extend([clean_str(sent) for sent in sentences])
        all_txt.extend([sent for sent in sentences])
  for stp_dir in stp_dirs:
    for stp_file in sorted(os.listdir(stp_dir)):
      # all_stp.append(stp_file+'\n')
      with open(os.path.join(stp_dir, stp_file)) as f:
        tmp = []
        for line in f:
          tmp.append(line)
          if line == '\n':
            # s = restore_sentence_from_tree(tmp)
            all_stp.append("".join(tmp))
            # all_stp[s] = 
            tmp.clear()
    
  with open('%s.stp.txt'%file_type, 'w') as f:
    for line in all_txt:
      f.write(line)
  with open('%s.stp'%file_type, 'w') as f:
    for line in all_stp:
      f.write(line)


  # f_txt = open('%s.stp.txt'%file_type, 'w')
  # f_stp = open('%s.stp'%file_type, 'w')
  # f_txt_todo = open('%s.todo'%file_type, 'w')
  # for txt in all_txt:
  #   s = clean_str(txt)
  #   if s in all_stp:
  #     f_txt.write(txt)
  #     f_stp.write(all_stp[s])
  #   else:
  #     f_txt_todo.write(txt)

get_results_v2(['len70', 'len200'],['len70_lex', 'len200_lex'], 'todo')
