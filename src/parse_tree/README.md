stanford parser

```bash
python extract_raw_text.py --input_dir data/RE/ --output_dir ~/tmp/ --max_len 220
```

extract  sentences from train.txt
max_len 216
<=60: 264420 (60,100]: 9887 (100, 200]: 420 >200: 1
ignored: 172
write 2645 files   40g 24th (511.76 wds/sec; 16.05 sents/sec)
write 99 files     40g 24th (175.02 wds/sec; 2.51 sents/sec)
write 5 files      40g 10th (45.39 wds/sec; 0.38 sents/sec)
write 1 files      40g  5th (1.58 wds/sec; 0.01 sents/sec)
extract  sentences from test.txt
max_len 214
<=60: 43988 (60,100]: 1617 (100, 200]: 71 >200: 1
ignored: 0
write 440 files    40g 24th (502.41 wds/sec; 15.73 sents/sec)
write 17 files     40g 24th (157.25 wds/sec; 2.24 sents/sec)
write 1 files      40g 10th (38.89 wds/sec; 0.33 sents/sec)
write 1 files      40g  5th (2.40 wds/sec; 0.01 sents/sec)


```bash
python parser.py --memory 40 --n_threads 24 \
     --length 60 --n_files 2645 \
     --basename train \
     --data_dir tmp \
     --out_dir lex \
     --parser_path stanford-parser-full-2018-02-27
```


```bash
python get_parser_results.py
python align_txt.py
```