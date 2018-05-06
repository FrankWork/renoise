#!/bin/bash

ParserPath=$HOME/bin/stanford-parser-full-2018-02-27

java -mx8g \
  -cp $ParserPath/stanford-parser.jar:$ParserPath/stanford-parser-3.9.1-models.jar \
  edu.stanford.nlp.parser.lexparser.LexicalizedParser \
  -nthreads 20 \
  -sentences newline \
  -outputFormatOptions includePunctuationDependencies \
  -outputFormat "typedDependencies" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz \
  $1 > $2
