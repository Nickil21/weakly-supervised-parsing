#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import re
import shutil
import sys
import time

import numpy as np
import torch


parser = argparse.ArgumentParser()

# Data path options
parser.add_argument("--tree1", default="data/PROCESSED/english/trees/Yoon_Kim/ptb-test-gold-filtered.txt")
parser.add_argument("--tree2", default="")
parser.add_argument("--length_cutoff", default=150, type=int)


def get_stats(span1, span2):
    tp = 0
    fp = 0
    fn = 0
    for span in span1:
        if span in span2:
            tp += 1
        else:
            fp += 1
    for span in span2:
        if span not in span1:
            fn += 1
    return tp, fp, fn


def get_nonbinary_spans(actions, SHIFT=0, REDUCE=1):
    spans = []
    stack = []
    pointer = 0
    binary_actions = []
    nonbinary_actions = []
    num_shift = 0
    num_reduce = 0
    for action in actions:
        # print(action, stack)
        if action == "SHIFT":
            nonbinary_actions.append(SHIFT)
            stack.append((pointer, pointer))
            pointer += 1
            binary_actions.append(SHIFT)
            num_shift += 1
        elif action[:3] == "NT(":
            stack.append("(")
        elif action == "REDUCE":
            nonbinary_actions.append(REDUCE)
            right = stack.pop()
            left = right
            n = 1
            while stack[-1] != "(":
                left = stack.pop()
                n += 1
            span = (left[0], right[1])
            if left[0] != right[1]:
                spans.append(span)
            stack.pop()
            stack.append(span)
            while n > 1:
                n -= 1
                binary_actions.append(REDUCE)
                num_reduce += 1
        else:
            assert False
    assert len(stack) == 1
    assert num_shift == num_reduce + 1
    return spans, binary_actions, nonbinary_actions


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1) :]:
        if char == "(":
            return True
        elif char == ")":
            return False
    raise IndexError("Bracket possibly not balanced, open bracket not followed by closed bracket")


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1) :]:
        if char == ")":
            break
        assert not (char == "(")
        output.append(char)
    return "".join(output)


def get_tags_tokens_lowercase(line):
    output = []
    line_strip = line.rstrip()
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == "("
        if line_strip[i] == "(" and not (is_next_open_bracket(line_strip, i)):  # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    # print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2  # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def get_nonterminal(line, start_idx):
    assert line[start_idx] == "("  # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1) :]:
        if char == " ":
            break
        assert not (char == "(") and not (char == ")")
        output.append(char)
    return "".join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = len(line_strip) - 1
    while i <= max_idx:
        assert line_strip[i] == "(" or line_strip[i] == ")"
        if line_strip[i] == "(":
            if is_next_open_bracket(line_strip, i):  # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append("NT(" + curr_NT + ")")
                i += 1
                while line_strip[i] != "(":  # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else:  # it's a terminal symbol
                output_actions.append("SHIFT")
                while line_strip[i] != ")":
                    i += 1
                i += 1
                while line_strip[i] != ")" and line_strip[i] != "(":
                    i += 1
        else:
            output_actions.append("REDUCE")
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ")" and line_strip[i] != "(":
                i += 1
    assert i == max_idx
    return output_actions


def main(args):
    corpus_f1 = [0.0, 0.0, 0.0]
    sent_f1 = []
    with torch.no_grad():
        for k, (tree1, tree2) in enumerate(zip(open(args.tree1, "r"), open(args.tree2))):
            tree1 = tree1.strip()
            action1 = get_actions(tree1)
            tags1, sent1, sent_lower1 = get_tags_tokens_lowercase(tree1)
            if len(sent1) > args.length_cutoff or len(sent1) == 1:
                continue
            gold_span1, binary_actions1, nonbinary_actions1 = get_nonbinary_spans(action1)
            tree2 = tree2.strip()
            action2 = get_actions(tree2)
            tags2, sent2, sent_lower2 = get_tags_tokens_lowercase(tree2)
            gold_span2, binary_actions2, nonbinary_actions2 = get_nonbinary_spans(action2)
            pred_span_set = set(gold_span2[:-1])  # the last span in the list is always the
            gold_span_set = set(gold_span1[:-1])  # trival sent-level span so we ignore it
            tp, fp, fn = get_stats(pred_span_set, gold_span_set)
            corpus_f1[0] += tp
            corpus_f1[1] += fp
            corpus_f1[2] += fn
            # Sentence-level F1 is based on the original code from PRPN, i.e.
            # L83-89 from https://github.com/yikangshen/PRPN/test_phrase_grammar.py
            # As pointed out in the README, this code isn't entirely correct since sentence-level F1 could be
            # nonzero for short sentences with only a single sentence-level span.
            # In practice this discrepancy has minimal impact on the final results, but should ideally be
            # fixed nonetheless.
            overlap = pred_span_set.intersection(gold_span_set)
            prec = float(len(overlap)) / (len(pred_span_set) + 1e-8)
            reca = float(len(overlap)) / (len(gold_span_set) + 1e-8)
            if len(gold_span_set) == 0:
                reca = 1.0
                if len(pred_span_set) == 0:
                    prec = 1.0
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            sent_f1.append(f1)
    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    print("Corpus F1: %.2f, Sentence F1: %.2f" % (corpus_f1 * 100, np.mean(np.array(sent_f1)) * 100))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
