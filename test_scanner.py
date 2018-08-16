import collections
import difflib
import pathlib
import dill as pickle
import scanner
import sys
import utils

USE_GROUPS = sys.argv[-1] == 'groups'

Token = collections.namedtuple('Token', 'line col type string virtual')

def reify_token(token):
    return Token(token.line, token.col, token.type, token.string, token.virtual)

def virtual_repr(token):
    if token.virtual:
        return ':' + ('*' * token.virtual)
    return ''

def extra_repr(token):
    if token.type in ['dedent', 'indent']:
        return f':{len(token.string)}'
    if token.type in utils.variable_content_tokens:
        return f':{token.string!r}'
    return ''

def pretty_token(token):
    base = f'{token.line}:{token.col}:{token.type}'
    return f'({base}{extra_repr(token)}{virtual_repr(token)})'

def print_diff(opcodes, expected, actual):
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for t in actual[i1:i2]:
                print(' ', pretty_token(t))
        if tag == 'delete':
            for t in actual[i1:i2]:
                print('-', pretty_token(t))
        if tag == 'insert':
            for t in expected[j1:j2]:
                print('+', pretty_token(t))
        if tag == 'replace':
            for t in actual[i1:i2]:
                print('-', pretty_token(t))
            for t in expected[j1:j2]:
                print('+', pretty_token(t))

def print_diff_groups(opcodes, expected, actual):
    for group in opcodes:
        for i, (tag, i1, i2, j1, j2) in enumerate(group):
            if i == 0:
                print(f'@@ -{i1},{i2-i1} +{j1},{j2-j1} @@')
            if tag == 'equal':
                for t in actual[i1:i2]:
                    print(' ', pretty_token(t))
            if tag == 'delete':
                for t in expected[i1:i2]:
                    print('-', pretty_token(t))
            if tag == 'insert':
                for t in actual[j1:j2]:
                    print('+', pretty_token(t))
            if tag == 'replace':
                for t in expected[i1:i2]:
                    print('-', pretty_token(t))
                for t in actual[j1:j2]:
                    print('+', pretty_token(t))

cwd = pathlib.Path.cwd()

keywords, tokens = open('keywords.json'), open('tokens.json')
regexp = scanner.Regexp.from_files(keywords, tokens)
for path in (cwd / 'examples').glob('*.b'):
    filename = path.relative_to(cwd / 'examples')
    expected_filename = (cwd / 'tests' / filename).with_suffix('.out')
    content = path.open().read()
    tokens = [reify_token(t) for t in scanner.scan(0, regexp, content)]
    if sys.argv[1] == 'load':
        with open(expected_filename, 'rb') as f:
            expected = pickle.load(f)
        sm = difflib.SequenceMatcher(a=expected, b=tokens, autojunk=False)
        if USE_GROUPS:
            opcodes = list(sm.get_grouped_opcodes())
            if opcodes != []:
                print(f'different result for {filename}')
                print_diff_groups(opcodes, expected, tokens)
                break
        else:
            opcodes = list(sm.get_opcodes())
            if opcodes != [('equal', 0, len(tokens), 0, len(tokens))]:
                print(f'different result for {filename}')
                print_diff(opcodes, expected, tokens)
                break
    if sys.argv[1] == 'save':
        with open(expected_filename, 'wb') as f:
            pickle.dump(tokens, f)
        print(f'saved {len(tokens)} tokens for {filename}')
