#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import util


def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Evaluate the result')
    parser.add_argument('-r', '--result', help='The result path', required=True)
    parser.add_argument('-a', '--answer', help='The standard answer path', required=True)
    args = parser.parse_args()
    return args.result, args.answer


def main():
    result, answer = parse_cmd_args()
    util.evaluate(result, answer)
    return None


if __name__ == '__main__':
    main()