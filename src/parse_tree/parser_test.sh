#!/bin/bash


ProjectPath=$HOME/parse_tree
ParserPath=$ProjectPath/stanford-parser-full-2018-02-27
data_dir=$ProjectPath/tmp_dir_test/len200
out_dir=$ProjectPath/tmp_dir_test/len200_lex
mkdir -p $out_dir

java -mx6g \
  -cp $ParserPath/stanford-parser.jar:$ParserPath/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -sentences newline \
  -writeOutputFiles  \
  -outputFormat "typedDependencies" \
  -nthreads 4 \
  -outputFilesDirectory  $out_dir \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  $data_dir/test.0 $data_dir/test.1 $data_dir/test.2 $data_dir/test.3 
