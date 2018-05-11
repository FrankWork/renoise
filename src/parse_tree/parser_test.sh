#!/bin/bash


ProjectPath=$HOME/parse_tree
ParserPath=$ProjectPath/stanford-parser-full-2018-02-27
data_dir=$ProjectPath/len200
out_dir=$ProjectPath/len200_lex
mkdir -p $out_dir

java -mx35g \
  -cp $ParserPath/stanford-parser.jar:$ParserPath/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -sentences newline \
  -writeOutputFiles  \
  -outputFormat "typedDependencies" \
  -nthreads 10 \
  -outputFilesDirectory  $out_dir \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  $data_dir/txt.0
