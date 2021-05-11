#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A short and simple example experiment with restarts.

The script is fully functional but also emphasises on readability. It
features restarts, timings and recording termination conditions.

To benchmark a different solver, `fmin` must be re-assigned and another
`elif` block added around line 119 to account for the solver-specific
call.

When calling the script, previously assigned variables can be re-assigned
via a ``name=value`` argument without white spaces, where ``value`` is
interpreted as a single python literal. Additionally, ``batch`` is recognized
as argument defining the `current_batch` number and the number of `batches`,
like ``batch=2/8`` runs batch 2 of 8.

Examples, preceeded by "python" in an OS shell and by "run" in an IPython
shell::

    example_experiment2.py budget_multiplier=3  # times dimension

    example_experiment2.py budget_multiplier=1e4 cocopp=None  # omit post-processing
    
    example_experiment2.py budget_multiplier=1e4 suite_name=bbob-biobj

    example_experiment2.py budget_multiplier=1000 batch=1/16

Post-processing with `cocopp` is only invoked in the single-batch case.

Details: ``batch=9/8`` is equivalent to ``batch=1/8``. The first number
is taken modulo to the second.

See the code: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment2.py>`_

See a beginners example experiment: `<https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment_for_beginners.py>`_

"""
from __future__ import division, print_function, unicode_literals
__author__ = "Nikolaus Hansen and ..."
import sys
import time  # output some timings per evaluation
from collections import defaultdict
import os, webbrowser  # to show post-processed results in the browser
import numpy as np  # for median, zeros, random, asarray
import cocoex  # experimentation module
try: import cocopp  # post-processing module
except: pass

### MKL bug fix
def set_num_threads(nt=1, disp=1):
    """see https://github.com/numbbo/coco/issues/1919
    and https://twitter.com/jeremyphoward/status/1185044752753815552
    """
    try: import mkl
    except ImportError: disp and print("mkl is not installed")
    else:
        mkl.set_num_threads(nt)
    nt = str(nt)
    for name in ['OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS',
                 'OMP_NUM_THREADS',
                 'MKL_NUM_THREADS']:
        os.environ[name] = nt
    disp and print("setting mkl threads num to", nt)

if sys.platform.lower() not in ('darwin', 'windows'):
    set_num_threads(1)

import scipy.optimize 
import cma

fmin = cma.fmin2

suite_name = "bbob"  # see cocoex.known_suite_names
budget_multiplier = 1e5  # times dimension, increase to 10, 100, ...
suite_filter_options = ("dimensions: 20")
# for more suite filter options see http://numbbo.github.io/coco-doc/C/#suite-parameters
batches = 1  # number of batches, batch=3/32 works to set both, current_batch and batches
current_batch = 1  # only current_batch modulo batches is relevant
output_folder = 'CMAES'

### possibly modify/overwrite above input parameters from input args
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', 'help', '-help', '--help'):
        print(__doc__)
        raise ValueError("printed help and aborted")
    input_params = cocoex.utilities.args_to_dict(
        sys.argv[1:], globals(), {'batch': 'current_batch/batches'}, print=print)
    globals().update(input_params)  # (re-)assign variables


if batches > 1:
    output_folder += "_batch%03dof%d" % (current_batch, batches)

### prepare
instances = ','.join([str(x) for x in range(1,6) for _ in range(20)])
suite = cocoex.Suite(suite_name, "instances: "+instances, suite_filter_options)
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
stoppings = defaultdict(list)  # dict of lists, key is the problem index
timings = defaultdict(list)  # key is the dimension

### go
print('*** benchmarking %s from %s on suite %s ***'
      % (fmin.__name__, fmin.__module__, suite_name))
time0 = time.time()
for batch_counter, problem in enumerate(suite):  # this loop may take hours or days...
    if batch_counter % batches != current_batch % batches:
        continue
    if not len(timings[problem.dimension]) and len(timings) > 1:
        print("\n   %s %d-D done in %.1e seconds/evaluations"
              % (minimal_print.stime, sorted(timings)[-2],
                 np.median(timings[sorted(timings)[-2]])), end='')
    problem.observe_with(observer)  # generate the data for cocopp post-processing
    problem(np.zeros(problem.dimension))  # making algorithms more comparable
    propose_x0 = problem.initial_solution_proposal  # callable, all zeros in first call
    evalsleft = lambda: int(problem.dimension * budget_multiplier + 1 -
                            max((problem.evaluations, problem.evaluations_constraints)))
    time1 = time.time()
    # apply restarts
    irestart = -1
    while evalsleft() > 0 and not problem.final_target_hit:
        irestart += 1

        xopt, es = fmin(problem, propose_x0, 2, 
            {
                'maxfevals':evalsleft(), 
                'verbose':-9, 
                'termination_callback' : lambda x : problem.final_target_hit
            }, restarts=9, bipop=True)
        stoppings[problem.index].append(es.stop())

    timings[problem.dimension].append((time.time() - time1) / problem.evaluations
                                      if problem.evaluations else 0)
    minimal_print(problem, restarted=irestart, final=problem.index == len(suite) - 1)

### print timings and final message
print("\n   %s %d-D done in %.1e seconds/evaluations"
      % (minimal_print.stime, sorted(timings)[-1], np.median(timings[sorted(timings)[-1]])))
if batches > 1:
    print("*** Batch %d of %d batches finished in %s."
          " Make sure to run *all* batches (via current_batch or batch=#/#) ***"
          % (current_batch, batches, cocoex.utilities.ascetime(time.time() - time0)))
else:
    print("*** Full experiment done in %s ***"
          % cocoex.utilities.ascetime(time.time() - time0))

print("Timing summary:\n"
      "  dimension  median seconds/evaluations\n"
      "  -------------------------------------")
for dimension in sorted(timings):
    print("    %3d       %.1e" % (dimension, np.median(timings[dimension])))
print("  -------------------------------------")
