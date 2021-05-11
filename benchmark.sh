#!/usr/bin/env bash
parallel python cmaes.py ::: batch={1..32}/32
