import argparse
import os
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

## ========= heldout relations ==================================
for file_name in ["testNegative.pb",  "testPositive.pb",  "trainNegative.pb",  
                  "trainPositive.pb"]:
  file_path = os.path.join(heldout_dir, file_name)
  sum=0
  for relation in parseDelimitedFrom(file_path, pb.Relation):
    sum+=len(relation.mention)
  print(sum)
# 166004
# 6444
# 91373
# 34811

# test: 172448
# train: 126184

## ========== manual =============================
for file_name in ["testNewEntities.pb", "testNewRelations.pb", "trainEntities.pb",  
                  "trainNegative.pb",  "trainPositive.pb"]:
  file_path = os.path.join(manual_dir, file_name)
  MsgType = pb.Entity if "Entities" in file_name else pb.Relation
  sum=0
  for msg in parseDelimitedFrom(file_path, MsgType):
    sum+=len(msg.mention)
  print(sum)
# 647271
# 502202
# 3487651
# 322249
# 121867

# test: 502202
# train: 444116

## ========== relations =============================
# sum=0
# for i, file_name in enumerate(os.listdir(relations_dir)):
#   if i!=0 and i%10000==0:
#     print("%d-th file, %d sentences" % (i, sum))
#   file_path = os.path.join(relations_dir, file_name)
#   document = parseFrom(file_path, pb.Document)
#   sum += len(document.sentences)
# print(sum)
# # 177,000+ file, 5,107,223 sentences



