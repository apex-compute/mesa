# SPDX-License-Identifier: MIT
# Apex late algebraic rules, run after FMA fusion.
import argparse
import sys

a, b, c, d, e, f, g, h, i, t = 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 't'
vertex = 'info->stage == MESA_SHADER_VERTEX'

late = [
   # mix(a, b, 1 - t) = a + (1 - t)(b - a) = b - t(b - a), one operation fewer.
   (('~ffma', ('fadd(is_used_once)', 1.0, ('fneg', t)), ('fadd', b, ('fneg', a)), a),
    ('ffma', ('fneg', t), ('fadd', b, ('fneg', a)), b)),

   # A final addition joins the innermost product of an FMA chain
   # (matrix-vector products plus a translation). Mesa's late rules do this
   # outside vertex stages for the sake of position invariance; here the
   # position slice is marked no-reassociate first, so other vertex outputs
   # take it too.
   (('~fadd', ('ffma(is_used_once)', a, b, ('ffma(is_used_once)', c, d, ('ffma', e, f, ('fmul(is_used_once)', 'g(is_not_const_and_not_fsign)', 'h(is_not_const_and_not_fsign)')))), 'i(is_not_const)'),
    ('ffma', a, b, ('ffma', c, d, ('ffma', e, f, ('ffma', g, h, i)))), vertex),
   (('~fadd', ('ffma(is_used_once)', a, b, ('ffma', c, d, ('fmul(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)'))), 'g(is_not_const)'),
    ('ffma', a, b, ('ffma', c, d, ('ffma', e, f, g))), vertex),
   (('~fadd', ('ffma(is_used_once)', a, b, ('fmul(is_used_once)', 'c(is_not_const_and_not_fsign)', 'd(is_not_const_and_not_fsign)')), 'e(is_not_const)'),
    ('ffma', a, b, ('ffma', c, d, e)), vertex),
   (('~fadd', ('fneg', ('ffma(is_used_once)', a, b, ('ffma', c, d, ('fmul(is_used_once)', 'e(is_not_const_and_not_fsign)', 'f(is_not_const_and_not_fsign)')))), 'g(is_not_const)'),
    ('ffma', ('fneg', a), b, ('ffma', ('fneg', c), d, ('ffma', ('fneg', e), f, g))), vertex),
]


def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('-p', '--import-path', required=True)
   args = parser.parse_args()
   sys.path.insert(0, args.import_path)
   import nir_algebraic  # pylint: disable=import-error
   print('#include "apex.h"')
   print(nir_algebraic.AlgebraicPass('apex_nir_opt_late', late).render())


if __name__ == '__main__':
   main()
