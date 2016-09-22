#!/bin/bash

targetdir="dailymail-ensemble/"

scp -r bdhingra@saskia:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_2l_h256_d.2/ $targetdir
scp -r bdhingra@saskia:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_2l_h384_d.2/ $targetdir
scp -r bdhingra@marten:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_2l_h384_d.1/ $targetdir
scp -r bdhingra@marten:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_2l_h384_d.3/ $targetdir
scp -r bdhingra@jan:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_3l_h256_d.2/ $targetdir
scp -r bdhingra@jan:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_3l_h384_d.2/ $targetdir
scp -r bdhingra@lysbet:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_3l_h256_d.1/ $targetdir
scp -r bdhingra@lysbet:/usr1/public/bdhingra/semantics_project/exp_dailymail/mul_3l_h384_d.1/ $targetdir

sh scripts/update_config.py
