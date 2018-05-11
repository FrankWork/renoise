import os
txt70 = []
txt200 = []
with open('test.todo') as f:
  for line in f:
    tokens = line.strip().split()
    n = len(tokens)
    if n <= 70:
      txt70.append(line)
    else:
      txt200.append(line)

with open('train.todo') as f:
  for line in f:
    tokens = line.strip().split()
    n = len(tokens)
    if n <= 70:
      txt70.append(line)
    else:
      txt200.append(line)

def write_shrads(sentences, dir_name):
    sentences = list(sentences)
    if not os.path.exists(dir_name):
      os.makedirs(dir_name)

    n_shards = 0
    for i in range(0, len(sentences), 100):
      filename = os.path.join(dir_name, "txt.%d" % n_shards)
      with open(filename, 'w') as f:
        for sent in sentences[i:i+100]:
          f.write(sent)
      n_shards+=1
    print("write %d files" % n_shards)
  
write_shrads(txt70, "len70")
write_shrads(txt200, "len200")

# len 70: 569 files 56873 sentences 35g 20th (462.32 wds/sec; 13.86 sents/sec
arr = ["$data_dir/txt.%d" %i for i in range(0,569)]
# len 200: 1  files  41   sentences 50g 10th (25.54 wds/sec; 0.25 sents/sec).

