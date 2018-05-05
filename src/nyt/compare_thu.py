import argparse
import os
import re
import Document_pb2 as pb

from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32

def parseDelimitedFrom(filename, MessageType):
  with open(filename, 'rb') as f:
    buf = f.read()
    n = 0
    while n < len(buf):
      msg_len, new_pos = _DecodeVarint32(buf, n)
      n = new_pos
      msg_buf = buf[n:n+msg_len]
      n += msg_len
      message = MessageType()
      message.ParseFromString(msg_buf)
      yield message

def parseFrom(filename, MessageType):
  with open(filename, 'rb') as f:
    message = MessageType()
    message.ParseFromString(f.read())
  return message
        
home_dir = os.environ['HOME']

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default="%s/work/data/nyt2010" % home_dir, help='dir')
args = parser.parse_args()

heldout_dir = os.path.join(args.data_dir, "heldout_relations")
manual_dir = os.path.join(args.data_dir, "kb_manual")
relations_dir = os.path.join(args.data_dir, "nyt-2005-2006.backup")


# ## ========= test set ==================================
# sentences_set = set()
# for filename in ["testNegative.pb",  "testPositive.pb"  ]:
#   file_path = os.path.join(heldout_dir, filename)
#   for relation in parseDelimitedFrom(file_path, pb.Relation):
#     for mt in relation.mention:
#       if mt.HasField("sentence"):
#         s = mt.sentence.lower()
#         # if s.startswith('france honors clint eastwood clint eastwood'):
#         #   print(s)
#         sentences_set.add(s)
# print('init set')

# unknown_set = set()
# with open('%s/work/data/nyt_text_data/RE/test.txt' % home_dir) as f:
#   for line in f:
#     parts = line.strip().split('\t')
#     sentence = parts[5].strip('###END###').lower()
#     sentence = re.sub('_', ' ', sentence)
#     if sentence not in sentences_set:
#       unknown_set.add(sentence)

# for s in unknown_set:
#   print(s)
#   for t in sentences_set:
#     if t.startswith(s[:20]):
#       print(t)
#   print()


## ========= train set ==================================
sentences_set = set()
for filename in ["heldout_relations/trainNegative.pb",  
                 "heldout_relations/trainPositive.pb",
                 "kb_manual/trainNegative.pb",  
                 "kb_manual/trainPositive.pb"]:
  file_path = os.path.join(args.data_dir, filename)
  for relation in parseDelimitedFrom(file_path, pb.Relation):
    for mt in relation.mention:
      if mt.HasField("sentence"):
        s = mt.sentence.lower().strip()
        sentences_set.add(s)
print('init set')

unknown_set = set()
with open('%s/work/data/nyt_text_data/RE/train.txt' % home_dir) as f:
  for line in f:
    parts = line.strip().split('\t')
    sentence = parts[5].strip('###END###').lower().strip()
    sentence = re.sub('_', ' ', sentence)
    if sentence not in sentences_set:
      unknown_set.add(sentence)

for s in unknown_set:
  print(s)
  for t in sentences_set:
    if t.startswith(s[:20]):
      print(t)
  print()




