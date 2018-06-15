stanford parser

```bash
python extract_raw_text.py --input_dir data/RE/ --output_dir ~/tmp/ --max_len 220
```

extract  sentences from train.txt
max_len 216
<=60: 264420 (60,100]: 9887 (100, 200]: 420 >200: 1
ignored: 172
write 2645 files
write 99 files
write 5 files
write 1 files
extract  sentences from test.txt
max_len 214
<=60: 43988 (60,100]: 1617 (100, 200]: 71 >200: 1
ignored: 0
write 440 files
write 17 files
write 1 files
write 1 files


```bash
python parser.py --memory 40 --n_threads 24 \
     --length 60 --n_files 2645 \
     --basename train \
     --data_dir tmp \
     --out_dir lex \
     --parser_path stanford-parser-full-2018-02-27
```


extract sentences from train.txt
len: 60 100 200 3000
334338 31632 2027 16 sentences
len60   3344 files 
          range(0,500) range(2000,3344) (386.06 wds/sec; 10.93 sents/sec). 12g 20th
          range(500,2000)               (287.55 wds/sec; 8.17 sents/sec).  24g 20th
len100  317 files 35g 20th    (149.06 wds/sec; 2.08 sents/sec)
len200  21 files  50g 10th    (25.93 wds/sec; 0.21 sents/sec)
len3000 1 files   50g 5th     (3.72 wds/sec; 0.02 sents/sec)     

extract sentences from test.txt
55840 5487 365 15
len60   559 files range(0, 559) 6g 4th
len100  55 files                6g 4th  (80.93 wds/sec; 1.13 sents/sec)
len200  4 files                 35g 10th; 50g 10th (21.25 wds/sec; 0.17 sents/sec)
len3000 1 files  max len 225    50g 5th (5.06 wds/sec; 0.02 sents/sec)