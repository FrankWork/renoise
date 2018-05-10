import re



def clean_str(line):
  line = re.sub('-lrb-', '(', line)
  line = re.sub('-rrb-', ')', line)
  line = re.sub("''", ' ', line)
  line = re.sub('\\\/', ' ', line) # remove '\/' 
  line = re.sub(r'\\\*', ' ', line) # remove '\*'
  line = re.sub(' {2,}', ' ', line)
  return line
  # 

lines_set = set()
with open("tmp_test.txt") as f:
  for line in f:
    parts = line.strip().split('\t')
    sentence = parts[5].strip('###END###').lower().strip()
    sentence = clean_str(sentence)
    lines_set.add(sentence)

lines = list(lines_set)
lines.sort()

with open('test.uniq', 'w') as f:
  for line in lines:
    f.write(line+'\n')

