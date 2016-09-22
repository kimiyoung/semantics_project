#!/bin/bash

targetdir="cbtcn-ensemble/"

scp -r bdhingra@floris:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_2l_h128_d.2_w2v1_w128/ $targetdir
scp -r bdhingra@floris:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_2l_h128_d.5_w2v1_w128/ $targetdir
scp -r bdhingra@floris:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_2l_h128_d.3_w2v1_w128/ $targetdir
scp -r bdhingra@floris:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_3l_h128_d.5_w2v1_w128/ $targetdir
scp -r bdhingra@floris:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_3l_h128_d.3_w2v1_w128/ $targetdir
scp -r bdhingra@arnout:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_2l_h256_d.2_w2v1_w128/ $targetdir
scp -r bdhingra@arnout:/usr1/public/bdhingra/semantics_project/exp_cbtcn/mul_2l_h256_d.5_w2v1_w128/ $targetdir
