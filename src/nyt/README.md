data_dir=$HOME/work/data/nyt2010
http://iesl.cs.umass.edu/riedel/ecml/


# nyt

ls $data_dir
Document.proto  
filtered-freebase-simple-topic-dump-3cols.tsv  1,845,262 line
heldout_relations/           448M    
    testNegative.pb  
    testPositive.pb  
    trainNegative.pb  
    trainPositive.pb
kb_manual/                   2.3G    
    testNewEntities.pb  
    testNewRelations.pb  
    trainEntities.pb  
    trainNegative.pb  
    trainPositive.pb
nyt-2005-2006.backup        3.3G    


 62M 12月  5  2010 heldout_relations.tgz  => heldout_relations/
724M 11月 10  2010 manual-05-06.tgz       => kb_manual/
302M 11月 17  2010 relations.tar.gz       => nyt-2005-2006.backup/


# protobuf

```bash
$ sudo apt install protobuf-compiler
$ protoc --python_out=./ ./Document.proto
```

# instances

data from THU:

$ wc -l RE/test.txt 
172448 
$ wc -l RE/train.txt 
570088 

heldout_relations/           448M    
    testNegative.pb    166004 sentences
    testPositive.pb    6444   sentences     pos + neg: 172448 sentences
    trainNegative.pb   91373  sentences
    trainPositive.pb   34811  sentences     pos + neg: 126184 sentences

kb_manual/                   2.3G    
    testNewEntities.pb     647271 
    testNewRelations.pb    502202
    trainEntities.pb       3487651
    trainNegative.pb       322249
    trainPositive.pb       121867           pos + neg : 444116 sentences

train: 126184 + 444116 = 570300

