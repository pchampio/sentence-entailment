#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def force_symlink(file1, file2):
    if os.path.exists(file2):
        os.remove(file2)
    os.symlink(file1, file2)


def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1
