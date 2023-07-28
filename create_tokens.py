
import glob
import itertools
import json
import os
import re
import sys
import struct
from concurrent.futures import  ProcessPoolExecutor
from tqdm import tqdm
from pathlib import Path
from functools import reduce

import numpy as np

TOKEN_CAPITALIZE = '~1'
TOKEN_UPPERCASE = '~2'

data_dir = "data/TinyStories_all_data"
hist_filename = 'hist.json'

executor = ProcessPoolExecutor(max_workers=4)

def process_text(all_tokens: 'list[str]', text: str):
    preceding_whitespace = True
    prev = 0
    text = (text
            .replace('‚', ',')
            .replace('‘', '\'')
            .replace('’', '\'')
            .replace('“', '"')
            .replace('”', '"')
            .replace('–', '-')
            .replace('—', '-')
            .replace('`', '\'')
            .replace('…', '...')
            .replace('‚', ',')
            )
    while (len(text)):

        if prev == len(text):
            print(json.dumps(all_tokens, indent='    '))
            print(text)
            print(preceding_whitespace)
            print(f'------------|{other}|--------------')
            print(len(set(all_tokens)))
            exit(1)
        prev = len(text)

        m = re.match('\\s+', text)
        if m:
            preceding_whitespace = True
            text = text[len(m.group(0)):]
            continue

        m = re.match('\\w+', text)
        if m:
            word = m.group(0)
            m = re.match('\\d', word)
            if m:
                word = m.group(0)
            if str.isupper(word) and len(word) > 1:
                all_tokens.append(TOKEN_UPPERCASE)
            elif str.isupper(word[0]):
                all_tokens.append(TOKEN_CAPITALIZE)
            all_tokens.append(word.lower())
            text = text[len(word):]
            preceding_whitespace = False
            continue

        m = re.match('[.!?,:;]', text)
        if m:
            char = m.group(0)
            all_tokens.append(char)
            text = text[len(char):]
            preceding_whitespace = True
            continue

        m = re.match('\.\.\.', text)
        if m:
            all_tokens.append('...')
            text = text[len(m.group(0)):]
            preceding_whitespace = True
            continue

        m = re.match('\(\.\.\.\)', text)
        if m:
            all_tokens.append('(<')
            all_tokens.append('...')
            all_tokens.append(')>')
            text = text[len(m.group(0)):]
            preceding_whitespace = True
            continue

        m = re.match('([\'"(])[\\w\']', text)
        if m and preceding_whitespace:
            all_tokens.append(m.group(1) + '<')
            text = text[len(m.group(1)):]
            preceding_whitespace = True
            continue

        m = re.match('([\'")])([\\s.,!?;:"]|$)+', text)
        if m:
            all_tokens.append(m.group(1) + '>')
            text = text[len(m.group(1)):]
            preceding_whitespace = False
            continue

        m = re.match('[$&+§*°/="%]', text)
        if m:
            all_tokens.append(m.group(0).replace('§', '$$').replace('°', '""').replace('"', '"<'))
            text = text[len(m.group(0)):]
            preceding_whitespace = True
            continue

        m = re.match('([\'-]|--)[-\\w]', text)
        if m:
            all_tokens.append(m.group(1))
            text = text[len(m.group(1)):]
            preceding_whitespace = True
            continue

        m = re.match('-+\\s', text)
        if m:
            all_tokens.append('--')
            text = text[len(m.group(0)):]
            preceding_whitespace = True
            continue

        m = re.match('-$', text)
        if m:
            text = text[len(m.group(0)):]
            preceding_whitespace = False
            continue

        all_tokens.append('!!')
        text = text[1:]
        preceding_whitespace = True
        continue

        # m = re.match(r'\W+', text)
        # other = m.group(0)
        # print(json.dumps(list(set(all_tokens)), indent='    '))
        # print(text)
        # print(preceding_whitespace)
        # print(f'------------|{other}|--------------')
        # print(len(set(all_tokens)))
        # exit(1)
        # text = text[len(other):]

    #print(json.dumps(all_tokens, indent='    '))
    #exit()

def process_shard(shard):
    tokenized_filename = shard.replace(".json", ".tokens")
    if (Path(tokenized_filename).exists()):
        return tokenized_filename
    print(f"Started {shard}")
    all_tokens = []
    with open(shard, "r") as f:
        data = json.load(f)
    #wrapper = lambda x: x
    wrapper = tqdm
    for example in wrapper(data):
        text = example["story"]
        process_text(all_tokens, text)

    #print(json.dumps(hist, indent='    '))

    #all_tokens = np.array(all_tokens, dtype=np.uint16)
    with open(tokenized_filename, "w") as f:
        f.write('^'.join(all_tokens))
    print(f"Saved {tokenized_filename}")
    return tokenized_filename


def text_to_tokens():

    # iterate the shards and tokenize all of them one by one
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    # process all the shards in a threadpool
    tokenized_filenames = []
    tokenized_filenames = list(executor.map(process_shard, shard_filenames))

    print(tokenized_filenames)

def create_hist(filename: str):
    print(f"Processing {filename}...")
    hist = {}
    with open(filename, 'r') as fd:
        tokens = fd.read().split('^')
    for token in tqdm(tokens):
        if token in hist:
            hist[token] += 1
        else:
            hist[token] = 1
    return hist

def tokens_histogram():
    tokenized_filenames = sorted(glob.glob(os.path.join(data_dir, "*.tokens")))
    hist = {}
    for src in executor.map(create_hist, tokenized_filenames):
        for token, count in src.items():
            if token in hist:
                hist[token] += count
            else:
                hist[token] = count
    hist = dict(sorted(hist.items(), key=lambda x:x[1], reverse=True))
    print(f"Writing {hist_filename}...")
    with open(hist_filename, 'w') as fd:
        fd.write(json.dumps(hist, indent='    '))

postfixes = [
    '-ing',
    '-s',
    '-ed'
]

allowed_tokens = set([
    '~1',
    '~2',
    '...',
    '\'<', '"<', '(<',
    '\'>', '">', ')>',
    '$$', '""',
    '-', '--',
    '!!',
    '\'',
])

allowed_tokens.update('.!?,:;$&+*/=%')
allowed_tokens.update(postfixes)

sentence_start_tokens = set([
    '~1',
    '~2',
    '\'<', '"<', '(<',
    '$$', '""',
    '-', '--',
    '!!',
    '\'',
])

sentence_start_tokens.update('$&+*/=%')
sentence_start_tokens.update(postfixes)

def split_stem(token: str, hist: 'dict[str, int]'):

    def choose_stem(stem1: str, stem2: str, add_token: str, hist: 'dict[str, int]'):
        if stem1 not in hist:
            stem = stem2
        elif stem2 not in hist:
            stem = stem1
        elif hist[stem1] > hist[stem2]:
            stem = stem1
        else:
            stem = stem2
        if stem in hist:
            return stem, add_token
        return None

    if not re.match('^\\w+$', token):
        return None, None

    m = re.match(r'(.*.[^e])ing$', token)
    if m:
        res = choose_stem(m.group(1), m.group(1) + 'e', '-ing', hist)
        if res: return res

    m = re.match(r'(.*..(s|sh|ch|x|z))es$', token)
    if m:
        res = choose_stem(m.group(1), m.group(1) + 'e', '-s', hist)
        if res: return res

    m = re.match(r'(.*...(?<!.s|sh|ch|.x|.z))s$', token)
    if m:
        res = choose_stem(m.group(1), m.group(1), '-s', hist)
        if res: return res

    m = re.match(r'(.*...)ed$', token)
    if m:
        res = choose_stem(m.group(1), m.group(1) + 'e', '-ed', hist)
        if res: return res

    return None, None


def reduce_tokens():
    mapping = {}
    with open(hist_filename, 'r') as fd:
        hist = json.load(fd)

    for postfix in postfixes:
        hist[postfix] = 0

    for token, count in list(hist.items()):
        if token not in allowed_tokens and not re.match('^\w+$', token, re.ASCII):
            print(f'Token "{token}" is not allowed', file=sys.stderr)
            for c in token:
                print(f'\\u{ord(c):X}', file=sys.stderr)
            hist[token] = 0

    for token, count in list(hist.items()):
        stem, add_token = split_stem(token, hist)
        if stem is not None:
            print(f'Moved token "{token}" with {count} to stem "{stem}" with {hist[stem]} + "{add_token}"')
            hist[stem] += count
            hist[add_token] += count
            hist[token] = 0
            mapping[token] = [add_token, stem]

    hist = dict(sorted(hist.items(), key=lambda x:x[1], reverse=True))
    passed_tokens = list(hist.keys())[:2000]
    nonwords_tokens = list(filter(lambda x: not re.match(r'^\w+$', x), passed_tokens))
    print(f"Done")
    with open('hist2.json', 'w') as fd:
        fd.write(json.dumps(hist, indent='    '))
    with open('mapping.json', 'w') as fd:
        fd.write(json.dumps(mapping, indent='    '))
    with open('passed.json', 'w') as fd:
        fd.write(json.dumps(passed_tokens, indent='    '))
    print(json.dumps(nonwords_tokens, indent='    '), file=sys.stderr)


def create_word(token, postfix_ed, postfix_ing, postfix_s):
    if postfix_s:
        if re.search(r'(s|sh|ch|x|z)$', token):
            token += 'e'
        token += 's'
    if postfix_ed:
        if token[-1] != 'e':
            token += 'e'
        token += 'd'
    if postfix_ing:
        if token[-1] == 'e':
            token = token[0:-1]
        token += 'ing'
    return token


def test_tokens_in_file(filename: str):
    valid_filename = filename.replace(".tokens", ".test.txt")
    bin_filename = filename.replace(".tokens", ".bin")
    markers_filename = filename.replace(".tokens", ".mark")
    if (Path(markers_filename).exists()):
        return
    print(f"Processing {filename}...")
    debug_text: 'list[str]' = []
    deleted: 'list[str]' = []
    with open(filename, 'r') as fd:
        tokens = fd.read().split('^')
    with open('passed.json', 'r') as fd:
        token_ordered_array = json.load(fd)
    with open('mapping.json', 'r') as fd:
        mapping = json.load(fd)
    valid_tokens = set(token_ordered_array)
    token_codes = dict(zip(token_ordered_array, range(len(token_ordered_array))))
    sentence = ''
    sentence_bin = bytearray()
    sentence_ended = False
    sentence_tokens = []
    prev = ''
    capitalize = False
    uppercase = False
    postfix_ed = False
    postfix_ing = False
    postfix_s = False
    all_valid = True

    output = []

    #tokens = tokens[0:200]
    tokens.append('~1')

    for unmapped_token in tqdm(tokens):
        if unmapped_token in mapping:
            mapped_tokens = mapping[unmapped_token]
        else:
            mapped_tokens = (unmapped_token,)

        for token in mapped_tokens:
            this = 'a' if re.match('\w+$', token) else '0' if re.match('\d', token) else token

            # detect end of sentence
            sentence_ended = sentence_ended or (this in '.!?')
            if sentence_ended and ((this in 'a0') or (this in sentence_start_tokens)):
                if all_valid:
                    debug_text.append('+   ' + sentence + '                 ' + '   '.join(sentence_tokens))
                    output.append(bytes(sentence_bin))
                else:
                    debug_text.append('-   ' + sentence + '                 ' + '   '.join(sentence_tokens))
                sentence = ''
                sentence_bin.clear()
                all_valid = True
                prev = ''
                sentence_ended = False
                sentence_tokens = []

            all_valid = all_valid and (token in valid_tokens)

            if token in valid_tokens:
                sentence_tokens.append(token)
            else:
                sentence_tokens.append('«' + token + "»")

            code = token_codes[token] if token in token_codes else 0

            sentence_bin.append(code & 0xFF)
            sentence_bin.append(code >> 8)

            if this == '~1':
                capitalize = True
            elif this == '~2':
                uppercase = True
            elif this == '-s':
                postfix_s = True
            elif this == '-ed':
                postfix_ed = True
            elif this == '-ing':
                postfix_ing = True
            else:
                # adding space
                if prev == '' or this in ('\'>', '">'):
                    pass
                elif prev == '--':
                    sentence += ' '
                elif prev in ('\'<', '"<') or this in '-\'.,!?:':
                    pass
                elif prev in ('\'>', '">') or prev in 'a.,!?:' or this == '--':
                    sentence += ' '
                elif prev in '\'-' or this == '0':
                    pass
                else:
                    sentence += ' '
                prev = this

                if this == 'a':
                    word = create_word(token, postfix_ed, postfix_ing, postfix_s)
                    if capitalize:
                        word = word[0].upper() + word[1:]
                    if uppercase:
                        word = word.upper()
                    sentence += word
                elif this in '0.,\'!?-:' or this in ('"<', '">', '--', '\'>', '\'<'):
                    sentence += token[0]
                elif all_valid:
                    print(f' `{prev}` -> `{this}`')
                    assert(False)
                else:
                    sentence += token
                capitalize = False
                uppercase = False
                postfix_ed = False
                postfix_ing = False
                postfix_s = False

    def reduce_sentence_points(a, b):
        last = a[-1]
        a.append(last + len(b))
        return a

    sentence_points = reduce(reduce_sentence_points, output, [0])

    print(f"Writing {valid_filename}...")
    with open(valid_filename, 'w') as fd:
        fd.write('\n'.join(debug_text))
    print(f"Writing {bin_filename}...")
    with open(bin_filename, 'wb') as fd:
        fd.write(b''.join(output))
    print(f"Writing {markers_filename} ({len(sentence_points) - 1} sentences)...")
    with open(markers_filename, 'w') as fd:
        json.dump(sentence_points, fd)

def test_tokens():
    tokenized_filenames = sorted(glob.glob(os.path.join(data_dir, "*.tokens")))
    list(executor.map(test_tokens_in_file, tokenized_filenames))


def make_dict():
    with open('passed.json', 'r') as fd:
        token_ordered_array = json.load(fd)
    with open('tokenizer2000.bin', 'wb') as fd:
        for token in token_ordered_array:
            bin = token.encode('utf-8')
            fd.write(struct.pack('<I', len(bin)))
            fd.write(bin)


#process_shard('data/TinyStories_all_data/data00.json')
#do_the_job()
#tokens_histogram()
#reduce_tokens()
#test_tokens_in_file('data/TinyStories_all_data/data00.tokens')
#test_tokens()
make_dict()
