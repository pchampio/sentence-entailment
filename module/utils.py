#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import errno

def force_symlink(file1, file2):
    if os.path.exists(file2):
        os.remove(file2)
    os.symlink(file1, file2)
