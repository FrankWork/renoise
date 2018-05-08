import re

lines_set = set()
with open("train.txt") as f:
  for line in f:
    parts = line.strip().split('\t')
    sentence = parts[5].strip('###END###').lower().strip()
    lines_set.add(re.sub('_', ' ', sentence))

lines = list(lines_set)
lines.sort()

with open('train.uniq', 'w') as f:
  for line in lines:
    f.write(line+'\n')

