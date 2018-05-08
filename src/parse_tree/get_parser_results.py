import os
import re

regex = re.compile("\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def restore_sentence_from_tree(lines):
  words = [0]*250
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
  return " ".join(tmp)+'\n'

def get_results(txt_dirs, stp_dirs, file_type):
  all_txt = []
  all_stp = []
  for i in range(len(txt_dirs)):
    txt_dir = txt_dirs[i]
    stp_dir = stp_dirs[i]
    print(txt_dir)
    zip_files = zip(sorted(os.listdir(txt_dir)), 
                sorted(os.listdir(stp_dir))
                )
  
    for txt_file, stp_file in zip_files:
      assert txt_file == stp_file[:-4]
      with open(os.path.join(txt_dir, txt_file)) as f:
        sentences = f.readlines()
      with open(os.path.join(stp_dir, stp_file)) as f:
        stp = []
        tmp = []
        for line in f:
          tmp.append(line)
          if line == '\n':
            # stp.append("".join(tmp))
            stp.append(restore_sentence_from_tree(tmp))
            tmp.clear()

      assert len(sentences) == len(stp)
      all_txt.extend(sentences)
      all_stp.extend(stp)
  all_txt.sort()
  all_stp.sort()
  with open('%s.stp.txt'%file_type, 'w') as f:
    for line in all_txt:
      f.write(line)
  with open('%s.stp'%file_type, 'w') as f:
    for line in all_stp:
      f.write(line)


get_results(
  ['tmp_test/len200'],
  ['tmp_test/len200_lex'],
  'test')
# get_results(
#   ['tmp_test/len60', 'tmp_test/len100','tmp_test/len200','tmp_test/len3000'],
#   ['tmp_test/len60_lex', 'tmp_test/len100_lex','tmp_test/len200_lex','tmp_test/len3000_lex'],
#   'test')

# get_results(
#   ['tmp_train/len60', 'tmp_train/len100','tmp_train/len200','tmp_train/len3000'],
#   ['tmp_train/len60_lex', 'tmp_train/len100_lex','tmp_train/len200_lex','tmp_train/len3000_lex'],
#   'train')

"(?<=[(])[^()]+\.[^()]+(?=[)])"