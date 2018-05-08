#!/bin/bash


ProjectPath=$HOME/parse_tree
ParserPath=$ProjectPath/stanford-parser-full-2018-02-27
data_dir=$ProjectPath/tmp_dir_train/len60
out_dir=$ProjectPath/tmp_dir_train/len60_lex
mkdir -p $out_dir

java -mx12g \
  -cp $ParserPath/stanford-parser.jar:$ParserPath/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -sentences newline \
  -writeOutputFiles  \
  -outputFormat "typedDependencies" \
  -nthreads 20 \
  -outputFilesDirectory  $out_dir \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  $data_dir/train.500 $data_dir/train.501 