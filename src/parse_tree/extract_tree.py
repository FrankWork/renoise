import re

regex = re.compile("\((.+)-(\d+)'*, (.+)-(\d+)'*\)")

def restore_sentence_from_tree(lines):
  words = [0]*300
  n_words = 0
  for line in lines:
    if line != '\n' and line != "":
      m=regex.search(line)
      if not m:
        print(line)
        exit()
      w1, idx1, w2, idx2 = m.groups()
      idx1 = int(idx1)
      idx2 = int(idx2)
      n_words = max(idx1, idx2, n_words)
      if idx1 > 0:
        words[idx1-1] = w1
      if idx2 > 0:
        words[idx2-1] = w2
  tmp = []
  for w in words[:n_words]:
    if w != 0:
      tmp.append(w)
  s = " ".join(tmp)
  s = re.sub(" _ ", "_", s)
  s = re.sub("[':&`_\.,!?=]", " ", s)
  s = re.sub("-", " ", s)
  s = re.sub(" ", " ", s)#'\xa0'
  s = re.sub("LRB|RRB", " ", s)

  s = re.sub(r'\\\*', ' ', s) # replace '\*'
  s = re.sub(' {2,}', ' ', s)
  s = s.strip()

  return s+'\n'

txt  = "the occasion was suitably exceptional : " +\
"a reunion of the 1970s-era sam rivers trio , " +\
"with dave_holland on bass and barry_altschul on drums ."


lex = ""+\
"det(occasion-2, the-1)\n"+\
"nsubj(exceptional-5, occasion-2)\n"+\
"cop(exceptional-5, was-3)\n"+\
"advmod(exceptional-5, suitably-4)\n"+\
"root(ROOT-0, exceptional-5)\n"+\
"det(reunion-8, a-7)\n"+\
"dep(exceptional-5, reunion-8)\n"+\
"case(trio-14, of-9)\n"+\
"det(trio-14, the-10)\n"+\
"amod(trio-14, 1970s-era-11)\n"+\
"compound(trio-14, sam-12)\n"+\
"compound(trio-14, rivers-13)\n"+\
"nmod:of(reunion-8, trio-14)\n"+\
"case(dave_holland-17, with-16)\n"+\
"advcl(exceptional-5, dave_holland-17)\n"+\
"case(bass-19, on-18)\n"+\
"nmod:on(dave_holland-17, bass-19)\n"+\
"cc(dave_holland-17, and-20)\n"+\
"advcl(exceptional-5, barry_altschul-21)\n"+\
"conj:and(dave_holland-17, barry_altschul-21)\n"+\
"case(drums-23, on-22)\n"+\
"nmod:on(barry_altschul-21, drums-23)\n"+\
"\n"
s = restore_sentence_from_tree(lex.split('\n'))
print(s)
# print(txt)
