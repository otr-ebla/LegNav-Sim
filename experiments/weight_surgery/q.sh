#!/bin/bash
# q.sh LOGNAME cmd...   → runs a python experiment, appends DONE sentinel
log=logs/$1.log; shift
PYTHONPATH=. ../../2drlenv/bin/python "$@" > $log 2>&1; echo "DONE rc=$?" >> $log
