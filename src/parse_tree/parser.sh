#!/bin/bash

ProjectPath=$HOME/parse_tree

data_dir=$ProjectPath/len70
out_dir=$ProjectPath/len70_lex
mkdir -p $out_dir

ParserPath=$ProjectPath/stanford-parser-full-2018-02-27

java -mx35g \
  -cp $ParserPath/stanford-parser.jar:$ParserPath/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -sentences newline \
  -writeOutputFiles  \
  -outputFormat "typedDependencies" \
  -nthreads 20 \
  -outputFilesDirectory  $out_dir \
  edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  $data_dir/txt.0 $data_dir/txt.1 $data_dir/txt.2 