import collections
import functools
import itertools
import json
import operator
import pprint
import re
import sys

from collections import namedtuple
from itertools import groupby

from utils import *

class Parser:
    """A parser for the Beatle programming language"""

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.index = 0

    def current_token(self):
        return self.tokens[self.index]

    def single_input(self):
        if self.current_token().type == 'newline':




def parse(tokens):
    return Parser(tokens).parse()
