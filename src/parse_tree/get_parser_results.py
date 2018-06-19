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
  return s+'\n'

def get_results_v2(txt_dir, stp_dir, file_type):
  all_txt = []
  all_stp = []

  for length in ['len60','len100','len200','len3000']:
    path = os.path.join(txt_dir, file_type, length)
    for txt_file in sorted(os.listdir(path)):
      with open(os.path.join(path, txt_file)) as f:
        sentences = f.readlines()
        all_txt.extend(sentences)
  
  for length in ['len60','len100','len200','len3000']:
    path = os.path.join(stp_dir, file_type, length)
    for stp_file in sorted(os.listdir(path)):
      with open(os.path.join(path, stp_file)) as f:
        tmp = []
        for line in f:
          tmp.append(line)
          if line == '\n':
            # s = restore_sentence_from_tree(tmp)
            s = "".join(tmp)
            all_stp.append(s)
            tmp.clear()
  
  assert len(all_txt) == len(all_stp)

  with open('%s.uniq'%file_type, 'w') as f:
    for line in all_txt:
      f.write(line)
  with open('%s.uniq.stp'%file_type, 'w') as f:
    for line in all_stp:
      f.write(line)

get_results_v2('tmp','lex', 'train')
get_results_v2('tmp','lex', 'test')