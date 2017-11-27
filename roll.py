#!/usr/bin/env python

import attr
import logging
import re
import sys
import readline
import operator
from numbers import Number
from random import randint
from pyparsing import Regex, oneOf, Optional, Group, Combine, Literal, CaselessLiteral, ZeroOrMore, StringStart, StringEnd, opAssoc, infixNotation, ParseException, Empty, pyparsing_common, ParseResults, White, Suppress
logFormatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logFormatter)

@attr.s
class IntegerValidator(object):
    min_val = attr.ib(default='-inf', convert=float)
    max_val = attr.ib(default='+inf', convert=float)
    handle_float = attr.ib(default='exception')
    @handle_float.validator
    def validate_handle_float(self, attribute, value):
        assert value in ('exception', 'truncate', 'round')

    def __call__(self, value):
        try:
            xf = float(value)
        except ValueError:
            raise ValueError('{} does not look like a number'.format(value))
        if not xf.is_integer():
            if self.handle_float == 'exception':
                raise ValueError('{} is not an integer'.format(value))
            elif self.handle_float == 'truncate':
                x = int(xf)
            else:
                x = round(xf)
        else:
            x = int(xf)
        if self.min_val is not None:
            assert x >= self.min_val
        if self.max_val is not None:
            assert x <= self.max_val
        return x

def normalize_die_type(x):
    if x == 'F':
        return x
    elif x == '%':
        return 100
    else:
        try:
            return IntegerValidator(min_val=2, handle_float='exception')(x)
        except Exception:
            raise ValueError('Invalid die type: d{}'.format(x))

def normalize_dice_count(x):
    xf = float(x)
    x = int(x)
    if not xf.is_integer():
        raise ValueError('dice count must be an integer, not {}'.format(xf))
    if x < 1:
        raise ValueError("dice count must be positive; {} is invalid".format(x))
    return x

def ImplicitToken(x):
    '''Like pyparsing.Empty, but yields one or more tokens instead of nothing.'''
    return Empty().setParseAction(lambda toks: x)

# TODO: Look at http://infohost.nmt.edu/tcc/help/pubs/pyparsing/web/classes.html#class-ParserElement

# Implementing the syntax described here: https://www.critdice.com/roll-advanced-dice
# https://stackoverflow.com/a/23956778/125921

# https://stackoverflow.com/a/46583691/125921

var_name = pyparsing_common.identifier.copy().setResultsName('varname')
real_num = pyparsing_common.fnumber.copy()
positive_int = pyparsing_common.integer.copy().setParseAction(lambda toks: [ IntegerValidator(min_val=1)(toks[0]) ])

drop_type = oneOf('K k X x -H -L')
drop_spec = Group(drop_type.setResultsName('type') +
                    (positive_int | ImplicitToken(1)).setResultsName('count')
).setResultsName('drop')

pos_int_implicit_one = (positive_int | ImplicitToken(1))

comparator_type = oneOf('<= < >= > ≤ ≥ =')

reroll_type = Combine(oneOf('R r') ^ ( oneOf('! !!') + Optional('p')))
reroll_spec = Group(
    reroll_type.setResultsName('type') +
    Optional(
        (comparator_type | ImplicitToken('=')).setResultsName('operator') + \
        positive_int.setResultsName('value')
    )
).setResultsName('reroll')

count_spec = Group(
    Group(
        comparator_type.setResultsName('operator') + \
        positive_int.setResultsName('value')
    ).setResultsName('success_condition') +
    Optional(
        Literal('f') +
        Group(
            comparator_type.setResultsName('operator') + \
            positive_int.setResultsName('value')
        ).setResultsName('failure_condition')
    )
).setResultsName('count_successes')

roll_spec = Group(
    (positive_int | ImplicitToken(1)).setResultsName('dice_count') +
    CaselessLiteral('d') +
    (positive_int | oneOf('% F')).setResultsName('die_type') +
    Optional(reroll_spec ^ drop_spec) +
    Optional(count_spec)
).setResultsName('roll')

expr_parser = infixNotation(
    baseExpr=(roll_spec | positive_int | real_num | var_name),
    opList=[
        (oneOf('** ^').setResultsName('operator', True), 2, opAssoc.RIGHT),
        (oneOf('* / × ÷').setResultsName('operator', True), 2, opAssoc.LEFT),
        (oneOf('+ -').setResultsName('operator', True), 2, opAssoc.LEFT),
    ]
).setResultsName('expr')

assignment_parser = var_name + Literal('=').setResultsName('assignment') + expr_parser

def roll_die(sides=6):
    '''Roll a single die.

Supports any valid integer number of sides as well as 'F' for a fate
die, which can return -1, 0, or 1 with equal probability.

    '''
    if sides == 'F':
        # Fate die = 1d3-2
        return roll_die(3) - 2
    else:
        return randint(1, int(sides))

class DieRolled(int):
    '''Subclass of int that allows a string suffix.

    This is meant for recording the result of rolling a die. The
    suffix is purely cosmetic, for the purposes of string conversion.
    It can be used to indicate a die roll that has been re-rolled or
    exploded, or to indicate a critical hit/miss.

    '''
    def __new__(cls, value, formatter='{}'):
        newval = super(DieRolled, cls).__new__(cls, value)
        newval.formatter = formatter
        return newval

    def __str__(self):
        return self.formatter.format(super().__str__())

    def __repr__(self):
        if self.formatter != '{}':
            return 'DieRolled(value={value!r}, formatter={formatter!r})'.format(
                value=int(self),
                formatter=self.formatter,
            )
        else:
            return 'DieRolled({value!r})'.format(value=int(self))

def validate_dice_roll_list(instance, attribute, value):
    for x in value:
        # Not using positive_int here because 0 is a valid roll for
        # penetrating dice, and -1 and 0 are valid for fate dice
        pyparsing_common.signed_integer.parseString(str(x))
def format_dice_roll_list(rolls, always_list=False):
    if len(rolls) == 0:
        raise ValueError('Need at least one die rolled')
    elif len(rolls) == 1 and not always_list:
        return str(rolls[0])
    else:
        return '[' + ",".join(map(str, rolls)) + ']'

@attr.s
class DiceRolled(object):
    '''Class representing the result of rolling one or more similar dice.'''
    dice_results = attr.ib(convert=list, validator = validate_dice_roll_list)
    dropped_results = attr.ib(default=attr.Factory(list), convert=list,
                              validator = validate_dice_roll_list)
    roll_desc = attr.ib(default='', convert=str)
    success_count = attr.ib(default=None)
    @success_count.validator
    def validate_success_count(self, attribute, value):
        if value is not None:
            self.success_count = int(value)

    def __attrs_post_init__(self):
        if len(self.dice_results) < 1:
            raise ValueError('Need at least one non-dropped roll')

    def total(self):
        if self.success_count is not None:
            return int(self.success_count)
        else:
            return sum(self.dice_results)

    def __str__(self):
        if self.roll_desc:
            prefix = '{roll} rolled'.format(roll=self.roll_desc)
        else:
            prefix = 'Rolled'
        if self.dropped_results:
            drop = ' (dropped {dropped})'.format(dropped = format_dice_roll_list(self.dropped_results))
        else:
            drop = ''
        if self.success_count is not None:
            tot = ', Total successes: ' + str(self.total())
        elif len(self.dice_results) > 1:
            tot = ', Total: ' + str(self.total())
        else:
            tot = ''
        return '{prefix}: {results}{drop}{tot}'.format(
            prefix=prefix,
            results=format_dice_roll_list(self.dice_results),
            drop=drop,
            tot=tot,
        )

    def __int__(self):
        return self.total()

def validate_by_parser(parser):
    def private_validator(instance, attribute, value):
        parser.parseString(str(value), True)
    return private_validator

@attr.s
class Comparator(object):
    cmp_dict = {
        '<=': operator.le,
        '<': operator.lt,
        '>=': operator.ge,
        '>': operator.gt,
        '≤': operator.le,
        '≥': operator.ge,
        '=': operator.eq,
    }
    operator = attr.ib(convert=str, validator=validate_by_parser(comparator_type))
    value = attr.ib(convert=int, validator=validate_by_parser(positive_int))

    def __str__(self):
        return '{op}{val}'.format(op=self.operator, val=self.value)

    def compare(self, x):
        '''Return True if x satisfies the comparator.

        In other words, x is placed on the left-hand side of the
        comparison, the Comparator's value is placed on the right hand
        side, and the truth value of the resulting test is returned.

        '''
        return self.cmp_dict[self.operator](x, self.value)

@attr.s
class RerollSpec(object):
    type = attr.ib(convert=str, validator=validate_by_parser(reroll_type))
    operator = attr.ib(default=None)
    value = attr.ib(default=None)

    def __attrs_post_init__(self):
        if (self.operator is None) != (self.value is None):
            raise ValueError('Operator and value must be provided together')

    def __str__(self):
        result = self.type
        if self.operator is not None:
            result += self.operator + str(self.value)
        return result

    def roll_die(self, sides):
        '''Roll a single die, following specified re-rolling rules.

        Returns a list of rolls, since some types of re-rolling
        collect the result of multiple die rolls.

        '''
        if sides == 'F':
            raise ValueError("Re-rolling/exploding is incompatible with Fate dice")
        sides = int(sides)

        if self.value is None:
            if self.type in ('R', 'r'):
                cmpr = Comparator('=', 1)
            else:
                cmpr = Comparator('=', sides)
        else:
            cmpr = Comparator(self.operator, self.value)

        if self.type == 'r':
            # Single reroll
            roll = roll_die(sides)
            if cmpr.compare(roll):
                roll = DieRolled(roll_die(sides), '{}' + self.type)
            return [ roll ]
        elif self.type == 'R':
            # Indefinite reroll
            roll = roll_die(sides)
            while cmpr.compare(roll):
                roll = DieRolled(roll_die(sides), '{}' + self.type)
            return [ roll ]
        elif self.type in ['!', '!!', '!p', '!!p']:
            # Explode/penetrate/compound
            all_rolls = [ roll_die(sides) ]
            while cmpr.compare(all_rolls[-1]):
                all_rolls.append(roll_die(sides))
            # If we never re-rolled, no need to do anything special
            if len(all_rolls) == 1:
                return all_rolls
            # For penetration, subtract 1 from all rolls except the first
            if self.type.endswith('p'):
                for i in range(1, len(all_rolls)):
                    all_rolls[i] -= 1
            # For compounding, return the sum
            if self.type.startswith('!!'):
                total = sum(all_rolls)
                return [ DieRolled(total, '{}' + self.type) ]
            else:
                for i in range(0, len(all_rolls)-1):
                    all_rolls[i] = DieRolled(all_rolls[i], '{}' + self.type)
                return all_rolls

@attr.s
class DropSpec(object):
    type = attr.ib(convert=str, validator=validate_by_parser(drop_type))
    count = attr.ib(default=1, convert=int, validator=validate_by_parser(positive_int))

    def __str__(self):
        if self.count > 1:
            return self.type + str(self.count)
        else:
            return self.type

    def drop_rolls(self, rolls):
        '''Drop the appripriate rolls from a list of rolls.

        Returns a 2-tuple of roll lists. The first list is the kept
        rolls, and the second list is the dropped rolls.

        The order of the rolls is not preserved.

        '''
        if not isinstance(rolls, list):
            rolls = list(rolls)
        keeping = self.type in ('K', 'k')
        # if keeping:
        #     num_to_keep = self.count
        # else:
        #     num_to_keep = len(rolls) - self.count
        # if num_to_keep == 0:
        #     raise ValueError('Not enough rolls: would drop all rolls')
        # elif num_to_keep == len(rolls):
        #     raise ValueError('Keeping too many rolls: would not drop any rolls')
        rolls.sort()
        if self.type in ('K', 'X', '-H'):
            rolls.reverse()
        (head, tail) = rolls[:self.count], rolls[self.count:]
        if keeping:
            (kept, dropped) = (head, tail)
        else:
            (kept, dropped) = (tail, head)
        return (kept, dropped)

@attr.s
class DiceRoller(object):
    die_type = attr.ib(convert = normalize_die_type)
    dice_count = attr.ib(default=1, convert=normalize_dice_count)
    reroll_spec = attr.ib(default=None)
    @reroll_spec.validator
    def validate_reroll_spec(self, attribute, value):
        if value is not None:
            assert isinstance(value, RerollSpec)
    drop_spec = attr.ib(default=None)
    @drop_spec.validator
    def validate_drop_spec(self, attribute, value):
        if value is not None:
            assert isinstance(value, DropSpec)
    success_comparator = attr.ib(default=None)
    failure_comparator = attr.ib(default=None)
    @success_comparator.validator
    @failure_comparator.validator
    def validate_comparator(self, attribute, value):
        if value is not None:
            assert isinstance(value, Comparator)

    def __attrs_post_init__(self):
        if self.reroll_spec is not None and self.drop_spec is not None:
            raise ValueError('Reroll and drop specs are mutually exclusive')
        if self.success_comparator is None and self.failure_comparator is not None:
            raise ValueError('Cannot use a failure condition without a success condition')

    def __str__(self):
        return '{count}d{type}{reroll}{drop}{success}{fail}'.format(
            count = self.dice_count if self.dice_count > 1 else '',
            type = self.die_type,
            reroll = self.reroll_spec or '',
            drop = self.drop_spec or '',
            success = self.success_comparator or '',
            fail = ('f' + str(self.failure_comparator)) if self.failure_comparator else '',
        )

    def roll(self):
        '''Roll dice according to specifications. Returns a DiceRolled object.'''
        all_rolls = []
        if self.reroll_spec:
            for i in range(self.dice_count):
                all_rolls.extend(self.reroll_spec.roll_die(self.die_type))
        else:
            for i in range(self.dice_count):
                all_rolls.append(roll_die(self.die_type))
        if self.drop_spec:
            (dice_results, dropped_results) = self.drop_spec.drop_rolls(all_rolls)
        else:
            (dice_results, dropped_results) = (all_rolls, [])
        if self.success_comparator is not None:
            success_count = 0
            for roll in dice_results:
                if self.success_comparator.compare(roll):
                    success_count += 1
            if self.failure_comparator is not None:
                for roll in dice_results:
                    if self.failure_comparator.compare(roll):
                        success_count -= 1
        else:
            success_count = None
        return DiceRolled(
            dice_results=dice_results,
            dropped_results=dropped_results,
            roll_desc=str(self),
            success_count=success_count,
        )

def make_dice_roller(expr):
    if isinstance(expr, str):
        expr = roll_spec.parseString(expr, True)['expr']
    assert expr.getName() == 'roll'
    expr = expr.asDict()

    dtype = normalize_die_type(expr['die_type'])
    dcount = normalize_dice_count(expr['dice_count'])
    constructor_args = {
        'die_type': dtype,
        'dice_count': dcount,
        'reroll_spec': None,
        'drop_spec': None,
        'success_comparator': None,
        'failure_comparator': None,
    }

    rrdict = None
    if 'reroll' in expr:
        rrdict = expr['reroll']
        constructor_args['reroll_spec'] = RerollSpec(**rrdict)

    if 'drop' in expr:
        ddict = expr['drop']
        constructor_args['drop_spec'] = DropSpec(**ddict)

    if 'count_successes' in expr:
        csdict = expr['count_successes']
        constructor_args['success_comparator'] = Comparator(**csdict['success_condition'])
        if 'failure_condition' in csdict:
            constructor_args['failure_comparator'] = Comparator(**csdict['failure_condition'])
    return DiceRoller(**constructor_args)

# examples = [
#     '1+1',
#     '1 + 1 + x',
#     '3d8',
#     '2e3 * 4d6 + 2',
#     '2d20k',
#     '3d20x2',
#     '4d4rK3',
#     '4d4R4',
#     '4d4R>=3',
#     '4d4r>=3',
#     '4d4!1',
#     '4d4!<3',
#     '4d4!p',
#     '2D20K+10',
#     '2D20k+10',
#     '10d6X4',
#     '4d8r + 6',
#     '20d6R≤2',
#     '6d10!≥8+6',
#     '10d4!p',
#     '20d6≥6',
#     '8d12≥10f≤2',
#     # '4d20R<=2!>=19Xx21>=20f<=5*2+3',  # Pretty much every possible feature
# ]

# example_results = {}
# for x in examples:
#     try:
#         example_results[x] = parse_roll(x)
#     except ParseException as ex:
#         example_results[x] = ex
# example_results

# rs = RerollSpec('!!p', '=', 6)
# rs.roll_die(6)

# ds = DropSpec('K', 2)
# ds.drop_rolls([1,2,3,4,5])
# ds = DropSpec('x', 2)
# ds.drop_rolls([1,2,3,4,5])

# parse_roll = lambda x: expr_parser.parseString(x)[0]
# exprstring = 'x + 1 + (2 + (3 + 4))'
# expr = parse_roll(exprstring)

# r = parse_roll('x + 1 - 2 * y * 4d4 + 2d20K1>=20f<=5')[0]

op_dict = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '×': operator.mul,
    '/': operator.truediv,
    '÷': operator.truediv,
    '**': operator.pow,
    '^': operator.pow,
}

def normalize_expr(expr):
    if isinstance(expr, str):
        return expr_parser.parseString(expr)['expr']
    try:
        if 'expr' in expr:
            return expr['expr']
    except TypeError:
        pass
    return expr

def _eval_expr_internal(expr, env={}, print_rolls=True, recursed_vars=set()):
    if isinstance(expr, Number):
        # Numeric literal
        return expr
    elif isinstance(expr, str):
        # variable name
        if expr in recursed_vars:
            raise ValueError('Recursive variable definition detected for {!r}'.format(expr))
        elif expr in env:
            var_value = env[expr]
            parsed = normalize_expr(var_value)
            return _eval_expr_internal(parsed, env, print_rolls,
                                       recursed_vars = recursed_vars.union([expr]))
        else:
            raise ValueError('Expression referenced undefined variable {!r}'.format(expr))
    elif 'operator' in expr:
        # Compound expression
        operands = expr[::2]
        operators = expr[1::2]
        assert len(operands) == len(operators) + 1
        values = [ _eval_expr_internal(x, env, print_rolls, recursed_vars)
                   for x in operands ]
        result = values[0]
        for (op, nextval) in zip(operators, values[1:]):
            opfun = op_dict[op]
            result = opfun(result, nextval)
        # Corece integral floats to ints
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return result
    else:
        # roll specification
        roller = make_dice_roller(expr)
        result = roller.roll()
        if print_rolls:
            print(result, file=sys.stderr)
        return int(result)

def eval_expr(expr, env={}, print_rolls=True):
    expr = normalize_expr(expr)
    return _eval_expr_internal(expr, env, print_rolls)

def _expr_as_str_internal(expr, env={}, recursed_vars = set()):
    if isinstance(expr, Number):
        # Numeric literal
        return str(expr)
    elif isinstance(expr, str):
        # variable name
        if expr in recursed_vars:
            raise ValueError('Recursive variable definition detected for {!r}'.format(expr))
        elif expr in env:
            var_value = env[expr]
            parsed = normalize_expr(var_value)
            return _expr_as_str_internal(parsed, env, recursed_vars = recursed_vars.union([expr]))
        else:
            return expr
    elif 'operator' in expr:
        # Compound expression
        operands = expr[::2]
        operators = expr[1::2]
        assert len(operands) == len(operators) + 1
        values = [ _expr_as_str_internal(x, env, recursed_vars)
                   for x in operands ]
        result = str(values[0])
        for (op, nextval) in zip(operators, values[1:]):
            result += ' {} {}'.format(op, nextval)
        return '(' + result + ')'
    else:
        # roll specification
        return str(make_dice_roller(expr))

def expr_as_str(expr, env={}):
    expr = normalize_expr(expr)
    expr = _expr_as_str_internal(expr, env)
    if expr.startswith('(') and expr.endswith(')'):
        expr = expr[1:-1]
    return expr

def read_roll(handle=sys.stdin):
    return input("Enter roll> ")

special_command_parser = (
    oneOf('h help ?').setResultsName('help') |
    oneOf('q quit exit').setResultsName('quit') |
    oneOf('v vars').setResultsName('vars') |
    (oneOf('d del delete').setResultsName('delete').leaveWhitespace() + Suppress(White()) + var_name)
)

def var_name_allowed(vname):
    '''Disallow variable names like 'help' and 'quit'.'''
    try:
        special_command_parser.parseString(vname, True)
        return False
    except ParseException:
        return True

line_parser = (special_command_parser ^ (assignment_parser | expr_parser))

def print_interactive_help():
    print('\n' + '''
To make a roll, type in the roll in dice notation, e.g. '4d4 + 4'. All
dice notation forms listed in
https://www.critdice.com/roll-advanced-dice and
http://rpg.greenimp.co.uk/dice-roller/ should be supported.
Expressions can include addition, subtraction, multiplication,
division, and exponentiation.

To assign a variable, use 'VAR = VALUE'. For example 'health_potion =
4d4+4'. Subsequent roll expressions can refer to this variable, whose
value will be substituted in to the expression.

If a variable's value includes any dice rolls, those dice will be
rolled (and produce a different value) every each time the variable is
used.

To delete a variable, type 'del VAR'.

To show the values of all currently assigned variables, type 'vars'.

To show this help text, type 'help'.

To quit, type 'quit'.
    '''.strip() + '\n', file=sys.stdout)

def print_vars(env):
    if len(env):
        logger.info('Currently defined variables:')
        for k in sorted(env.keys()):
            print('{} = {!r}'.format(k, env[k]), file=sys.stderr)
    else:
        logger.info('No vars are currently defined.')

if __name__ == '__main__':
    expr_string = " ".join(sys.argv[1:])
    if re.search("\\S", expr_string):
        try:
            # Note: using expr_parser instead of line_parser, because
            # on the command line only roll expressions are valid.
            expr = expr_parser.parseString(expr_string, True)
            result = eval_expr(expr)
            logger.info("Total roll for {expr!r}: {result}".format(
                expr=expr_as_str(expr),
                result=result
            ))
        except Exception as exc:
            logger.error("Error while rolling: %s", repr(exc))
            raise exc
            sys.exit(1)
    else:
        env = {}
        while True:
            try:
                expr_string = read_roll()
                if not re.search("\\S", expr_string):
                    continue
                parsed = line_parser.parseString(expr_string, True)
                if 'help' in parsed:
                    print_interactive_help()
                elif 'quit' in parsed:
                    logger.info('Quitting.')
                    break
                elif 'vars' in parsed:
                    print_vars(env)
                elif 'delete' in parsed:
                    vname = parsed['varname']
                    if vname in env:
                        logger.info('Deleting saved value for {var!r}.'.format(var=vname))
                        del env[vname]
                    else:
                        logger.error('Variable {var!r} is not defined.'.format(var=vname))
                elif re.search("\\S", expr_string):
                    if 'assignment' in parsed:
                        # We have an assignment operation
                        vname = parsed['varname']
                        if var_name_allowed(vname):
                            env[vname] = expr_as_str(parsed['expr'])
                            logger.info('Saving {var} as {expr!r}'.format(
                                var=vname, expr=env[vname],
                            ))
                        else:
                            logger.error('You cannot use {!r} as a variable name because it is a special command.'.format(vname))
                    else:
                        # Just an expression to evaluate
                        result = eval_expr(parsed['expr'], env)
                        logger.info('Total roll for {expr!r}: {result}'.format(
                            expr=expr_as_str(parsed, env),
                            result=result,
                        ))
                print('', file=sys.stderr)
            except KeyboardInterrupt:
                print('', file=sys.stderr)
            except EOFError:
                logger.info('Quitting.')
                break
            except Exception as exc:
                logger.error('Error while evaluating {expr!r}: {ex!r}'.format(
                    expr=expr_string, ex=exc,
                ))
