with open('tmp_train/len3000/train.0') as f:
  lines = f.readlines()

with open('tmp_train/len3000/train.0', 'w') as f:
  for line in lines:
    tokens = line.strip().split()
    if len(tokens) <= 220:
      f.write(line)
