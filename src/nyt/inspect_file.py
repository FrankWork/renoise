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
file_path = os.path.join(heldout_dir, "trainPositive.pb")
for relation in parseDelimitedFrom(file_path, pb.Relation):
  print("sourceGuid: %s" % relation.sourceGuid)
  print("destGuid: %s" % relation.destGuid)
  print("relType: %s" % relation.relType)
  for mt in relation.mention:
    print("\nmention:")
    print("filename %s sourceId %d destId %d" %
          (mt.filename, mt.sourceId, mt.destId))
    print("feature: %s" % "\t".join(mt.feature))
    if mt.HasField("sentence"):
      print("sentence: %s" % mt.sentence)
  break
# sourceGuid: /guid/9202a8c04000641f800000000005af5c
# destGuid: /guid/9202a8c04000641f8000000000573408
# relType: /location/location/contains

# mention:
# filename /m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/nyt-2005-2006-joint/1662854.xml.pb sourceId -2147483646 destId -2147483645
# feature: LOCATION->LOCATION     inverse_true|LOCATION|,|LOCATION        inverse_true|in|LOCATION|,|LOCATION|,   inverse_true|jet in|LOCATION|,|LOCATION|, because       inverse_true|LOCATION|,|LOCATION        inverse_true|in|LOCATION|,|LOCATION|,   inverse_true|jet in|LOCATION|,|LOCATION|, because       str:Harbor[PMOD]<-|LOCATION|[NMOD]->|LOCATION|[NMOD]->Belle     str:Harbor[PMOD]<-|LOCATION|[NMOD]->|LOCATION|[P]->,    str:Harbor[PMOD]<-|LOCATION|[NMOD]->|LOCATION|[NMOD]->Queens    str:Harbor[PMOD]<-|LOCATION|[NMOD]->|LOCATION|[P]->,    str:in[ADV]<-|LOCATION|[NMOD]->|LOCATION|[NMOD]->Belle  str:in[ADV]<-|LOCATION|[NMOD]->|LOCATION|[P]->, str:in[ADV]<-|LOCATION|[NMOD]->|LOCATION|[NMOD]->Queens str:in[ADV]<-|LOCATION|[NMOD]->|LOCATION|[P]->, str:Harbor[PMOD]<-|LOCATION|[NMOD]->|LOCATION   dep:[PMOD]<-|LOCATION|[NMOD]->|LOCATION dir:<-|LOCATION|->|LOCATION     str:in[ADV]<-|LOCATION|[NMOD]->|LOCATION        dep:[ADV]<-|LOCATION|[NMOD]->|LOCATION  dir:<-|LOCATION|->|LOCATION     str:LOCATION|[NMOD]->|LOCATION|[NMOD]->Belle    dep:LOCATION|[NMOD]->|LOCATION|[NMOD]-> dir:LOCATION|->|LOCATION|->     str:LOCATION|[NMOD]->|LOCATION|[P]->,   dep:LOCATION|[NMOD]->|LOCATION|[P]->    dir:LOCATION|->|LOCATION|->     str:LOCATION|[NMOD]->|LOCATION|[NMOD]->Queens   dep:LOCATION|[NMOD]->|LOCATION|[NMOD]-> dir:LOCATION|->|LOCATION|->     str:LOCATION|[NMOD]->|LOCATION|[P]->,   dep:LOCATION|[NMOD]->|LOCATION|[P]->    dir:LOCATION|->|LOCATION|->     str:LOCATION|[NMOD]->|LOCATION
# sentence: Sen. Charles E. Schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in Belle Harbor , Queens , because equipment failure , not pilot error , might have been the cause .

  

# ## ========== manual =============================
# file_path = os.path.join(manual_dir, "testNewEntities.pb")
# for ent in parseDelimitedFrom(file_path, pb.Entity):
#   print("guid %s" % ent.guid)
#   if ent.HasField("name"):
#     print("name %s" % ent.name)
#   if ent.HasField("type"):
#     print("type %s" % ent.type)
#   if ent.HasField("pred"):
#     print("pred %s" % ent.pred)
#   for mt in ent.mention:
#     print("\nmention:")
#     print("id %d filename %s" % (mt.id, mt.filename))
#     print("feature: %s" % "\t".join(mt.feature))
#   break
# # guid guid0
# # name WKCR
# # type
# # pred person
# # mention:
# # id -2147483641 filename /m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/2007/1850511.xml.pb
# # feature: by@head        IN@hpos ORGANIZATION    WKCR    WKCR    by@-1   VBN IN@-2-1     the@+1  the student-run@+1+2
# # mention:
# # id -2147483646 filename /m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/2007/1849689.xml.pb
# # feature: station@head   NN@hpos ORGANIZATION    WKCR    WKCR    ,@-1    NN ,@-2-1       B_1@+1  B_1 B_2@+1+2
# # mention:
# # id -2147483635 filename /m/vinci8/data1/riedel/projects/relation/kb/nyt1/docstore/2007/1849689.xml.pb
# # feature: marched@head   VBD@hpos        ORGANIZATION    WKCR    WKCR    week@-1 DT NN@-2-1      through@+1      through that@+1+2

  
# ## ========== relations =============================
# file_names = os.listdir(relations_dir)
# # file_path = os.path.join(relations_dir, file_names[0])
# file_path = os.path.join(relations_dir, "1662854.xml.pb")
# document = parseFrom(file_path, pb.Document)
# print("filename: %s" % document.filename)
# for sent in document.sentences:
#   s = ""
#   for tok in sent.tokens:
#     s += (tok.word + '||' + tok.tag + '||' + tok.ner + '\t')
#   print("tokens: %s" % s)
#   # Hollis||NNP||PERSON     Marion-Dene||NNP||PERSON        von||NNP||O     Summer||NNP||O  ,||,||O a||DT||O
#   for mt in sent.mentions:
#     print("mention: %d entityGuid %s from %d to %d label: %s"%(mt.id, mt.entityGuid, getattr(mt, "from"), mt.to, mt.label))
#     # mention: -2147483639 entityGuid  from 0 to 1 label: PERSON
#     # mention: -2147483640 entityGuid  from 8 to 13 label: ORGANIZATION

  


