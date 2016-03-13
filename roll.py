#!/usr/bin/env python

import logging
import re
import sys
import readline
from random import randint

logFormatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logFormatter)

def format_roll(s, n=1, mod=0, drop_low=0, drop_high=0):
    fmt = ''
    if n > 1:
        fmt += str(n)
    fmt += 'd' + str(s)
    if drop_low > 0:
        fmt += "-L"
        if drop_low > 1:
            fmt += str(drop_low)
    if drop_high > 0:
        fmt += "-H"
        if drop_high > 1:
            fmt += str(drop_high)
    if mod != 0:
        fmt += '%+i' % (mod,)
    return fmt

def roll_die(s, mod=0, adv=0):
    '''Roll a single S-sided die. '''
    roll = randint(1, s) + mod
    logger.debug('%s rolled: %s', format_roll(s, 1, mod), roll)
    return roll

def roll_dice(s, n=1, mod=0, drop_low=0, drop_high=0):
    '''Roll n s-sided dice, then add modifier.

    (The modifier feature of this function currently isn't used by the
    script.)

    '''
    s = int(s)
    n = int(n)
    mod = int(mod)
    drop_low = int(drop_low)
    drop_high = int(drop_high)
    if n < 1:
        raise ValueError('Must roll at least one die.')
    if s < 2:
        raise ValueError('Dice must have at least 2 sides')
    if drop_low < 0 or drop_high < 0:
        raise ValueError('Cannot drop negative number of dice.')
    if drop_low + drop_high >= n:
        raise ValueError('Dropping too many dice; must keep at least one die.')
    n = int(n)
    rolls = [ roll_die(s) for i in xrange(n) ]
    dropped_low_rolls = sorted(rolls)[:drop_low]
    dropped_high_rolls = sorted(rolls, reverse=True)[:drop_high]
    kept_rolls = list(rolls)
    # Cannot use setdiff because of possible repeated values
    for drop in dropped_low_rolls + dropped_high_rolls:
        kept_rolls.remove(drop)
    total = mod + sum(kept_rolls)
    # TODO: Special reporting for natural 1 and natural 20 (only when a single d20 roll is returned)
    natural = ''
    if len(kept_rolls) == 1 and s == 20:
        if kept_rolls[0] == 1:
            natural = " (NATURAL 1)"
        elif kept_rolls[0] == 20:
            natural = " (NATURAL 20)"
    if n > 1:
        paren_stmts = [ ('Individual rolls', kept_rolls), ]
        if dropped_low_rolls:
            paren_stmts.append( ('Dropped low', dropped_low_rolls) )
        if dropped_high_rolls:
            paren_stmts.append( ('Dropped high', dropped_high_rolls) )
        if dropped_low_rolls or dropped_high_rolls:
            paren_stmts.append( ('Original rolls', rolls) )
        paren_stmt = "; ".join("%s: %s" % (k, repr(v)) for k,v in paren_stmts)
        logger.info('%s rolled: %s%s\n(%s)', format_roll(s, n, mod, drop_low, drop_high), total, natural, paren_stmt)
    else:
        logger.info('%s rolled: %s%s', format_roll(s, n, mod, drop_low, drop_high), total, natural)
    return total

def _roll_matchgroup(m):
    sides = int(m.group(2))
    n = int(m.group(1) or 1)
    dropspec = m.group(3) or ''
    mod = int(m.group(4) or 0)
    drop_low, drop_high = 0,0
    drops = re.findall('-([HL])(\\d*)', dropspec)
    for (dtype, dnum) in drops:
        if dnum == '':
            dnum = 1
        else:
            dnum = int(dnum)
        if dtype == 'L':
            drop_low += dnum
        else:
            drop_high += dnum
    return str(roll_dice(sides, n, mod, drop_low, drop_high))

def roll(expr):
    try:
        logger.info("Rolling %s", expr)
        # Multi dice rolls, e.g. "4d6"
        subbed = re.sub('(\\d+)?\\s*d\\s*(\\d+)((?:-[HL]\\d*)+)?([+-]\\d+)?',
                           lambda m: _roll_matchgroup(m),
                           expr)
        logger.debug("Expression with dice rolls substituted: %s", repr(subbed))
        return int(eval(subbed))
    except Exception as exc:
        raise Exception("Cannot parse expression %s: %s" % (repr(expr), exc))

def read_roll(handle=sys.stdin):
    return raw_input("Enter roll> ")

if __name__ == '__main__':
    expr = " ".join(sys.argv[1:])
    if re.search("\\S", expr):
        try:
            result = roll(expr)
            logger.info("Total roll: %s", result)
        except Exception as exc:
            logger.error("Error while rolling: %s", repr(exc))
            sys.exit(1)
    else:
        try:
            while True:
                try:
                    expr = read_roll()
                    if expr in ('exit', 'quit', 'q'):
                        break
                    if re.search("\\S", expr):
                        try:
                            result = roll(expr)
                            logger.info("Total roll: %s", result)
                        except Exception as exc:
                            logger.error("Error while rolling: %s", repr(exc))
                except KeyboardInterrupt:
                    print "\n",
        except EOFError:
            # Print a newline before exiting
            print "\n",
