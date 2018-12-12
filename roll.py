#!/usr/bin/env python

import attr
import logging
import re
import sys
import readline
import operator
import traceback
from numbers import Number
from random import SystemRandom
from pyparsing import ParserElement, Token, Regex, oneOf, Optional, Group, Combine, Literal, CaselessLiteral, ZeroOrMore, StringStart, StringEnd, opAssoc, infixNotation, ParseException, Empty, pyparsing_common, ParseResults, White, Suppress

from typing import Union, List, Any, Tuple, Sequence, Dict, Callable, Set, TextIO
from typing import Optional as OptionalType

try:
    import colorama
    colorama.init()
    from colors import color
except ImportError:
    # Fall back to no color
    def color(s: str, *args, **kwargs):
        '''Fake color function that does nothing.

        Used when the colors module cannot be imported.'''
        return s

EXPR_COLOR = "green"
RESULT_COLOR = "red"
DETAIL_COLOR = "yellow"

logFormatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(logging.StreamHandler())
for handler in logger.handlers:
    handler.setFormatter(logFormatter)

sysrand = SystemRandom()
randint = sysrand.randint

def int_limit_converter(x: Any) -> OptionalType[int]:
    if x is None:
        return None
    else:
        return int(x)

@attr.s
class IntegerValidator(object):
    min_val: OptionalType[int] = attr.ib(default = None, converter = int_limit_converter)
    max_val: OptionalType[int] = attr.ib(default = None, converter = int_limit_converter)
    handle_float: str = attr.ib(default = 'exception')
    @handle_float.validator
    def validate_handle_float(self, attribute, value):
        assert value in ('exception', 'truncate', 'round')
    value_name: str = attr.ib(default = "value")

    def __call__(self, value: Any) -> int:
        xf: float
        x: int
        try:
            xf = float(value)
        except ValueError:
            raise ValueError('{} {} does not look like a number'.format(self.value_name, value))
        if not xf.is_integer():
            if self.handle_float == 'exception':
                raise ValueError('{} {} is not an integer'.format(self.value_name, value))
            elif self.handle_float == 'truncate':
                x = int(xf)
            else:
                x = round(xf)
        else:
            x = int(xf)
        if self.min_val is not None and x < self.min_val:
                raise ValueError('{} {} is too small; must be at least {}'.format(self.value_name, value, self.min_val))
        if self.max_val is not None and x > self.max_val:
                raise ValueError('{} {} is too large; must be at most {}'.format(self.value_name, value, self.max_val))
        return x

die_face_num_validator = IntegerValidator(
    min_val = 2, handle_float = 'exception',
    value_name = 'die type',
)

DieFaceType = Union[int, str]
def is_fate_face(x: DieFaceType) -> bool:
    if isinstance(x, int):
        return False
    else:
        x = str(x).upper()
        return x in ('F', 'F.1', 'F.2')

def normalize_die_type(x: DieFaceType) -> DieFaceType:
    if is_fate_face(x):
        return str(x).upper()
    elif x == '%':
        return 100
    else:
        return die_face_num_validator(x)

dice_count_validator = IntegerValidator(
    min_val = 1, handle_float = 'exception',
    value_name = 'dice count'
)

# Just a named function wrapper for dice_count_validator
def normalize_dice_count(x: Any) -> int:
    return dice_count_validator(x)

def ImplicitToken(x) -> ParserElement:
    '''Like pyparsing.Empty, but yields one or more tokens instead of nothing.'''
    return Empty().setParseAction(lambda toks: x)

# TODO: Look at http://infohost.nmt.edu/tcc/help/pubs/pyparsing/web/classes.html#class-ParserElement

# Implementing the syntax described here: https://www.critdice.com/roll-advanced-dice
# https://stackoverflow.com/a/23956778/125921

# https://stackoverflow.com/a/46583691/125921

var_name: ParserElement = pyparsing_common.identifier.copy().setResultsName('varname')
real_num: ParserElement = pyparsing_common.fnumber.copy()
positive_int: ParserElement = pyparsing_common.integer.copy().setParseAction(lambda toks: [ IntegerValidator(min_val=1)(toks[0]) ])

drop_type: ParserElement = oneOf('K k X x -H -L')
drop_spec: ParserElement = Group(
    drop_type.setResultsName('type') +
    (positive_int | ImplicitToken(1)).setResultsName('count')
).setResultsName('drop')

pos_int_implicit_one: ParserElement = (positive_int | ImplicitToken(1))

comparator_type: ParserElement = oneOf('<= < >= > ≤ ≥ =')

reroll_type: ParserElement = Combine(oneOf('R r') ^ ( oneOf('! !!') + Optional('p')))
reroll_spec: ParserElement = Group(
    reroll_type.setResultsName('type') +
    Optional(
        (comparator_type | ImplicitToken('=')).setResultsName('operator') + \
        positive_int.setResultsName('value')
    )
).setResultsName('reroll')

count_spec: ParserElement = Group(
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

roll_spec: ParserElement = Group(
    (positive_int | ImplicitToken(1)).setResultsName('dice_count') +
    CaselessLiteral('d') +
    (positive_int | oneOf('% F F.1 F.2')).setResultsName('die_type') +
    Optional(reroll_spec ^ drop_spec) +
    Optional(count_spec)
).setResultsName('roll')

expr_parser: ParserElement = infixNotation(
    baseExpr=(roll_spec ^ positive_int ^ real_num ^ var_name),
    opList=[
        (oneOf('** ^').setResultsName('operator', True), 2, opAssoc.RIGHT),
        (oneOf('* / × ÷').setResultsName('operator', True), 2, opAssoc.LEFT),
        (oneOf('+ -').setResultsName('operator', True), 2, opAssoc.LEFT),
    ]
).setResultsName('expr')

assignment_parser: ParserElement = var_name + Literal('=').setResultsName('assignment') + expr_parser

def roll_die(sides: DieFaceType = 6) -> int:
    '''Roll a single die.

Supports any valid integer number of sides as well as 'F' for a fate
die, which can return -1, 0, or 1 with equal probability.

    '''
    if sides in ('F', 'F.2'):
        # Fate die = 1d3-2
        return roll_die(3) - 2
    elif sides == 'F.1':
        d6 = roll_die(6)
        if d6 == 1:
            return -1
        elif d6 == 6:
            return 1
        else:
            return 0
    else:
        return randint(1, int(sides))

class DieRolled(int):
    '''Subclass of int that allows a string suffix.

    This is meant for recording the result of rolling a die. The
    suffix is purely cosmetic, for the purposes of string conversion.
    It can be used to indicate a die roll that has been re-rolled or
    exploded, or to indicate a critical hit/miss.

    '''
    formatter: str

    def __new__(cls: type, value: int, formatter: str = '{}') -> 'DieRolled':
        newval = super(DieRolled, cls).__new__(cls, value) # type: ignore
        newval.formatter = formatter
        return newval

    def __str__(self) -> str:
        return self.formatter.format(super().__str__())

    def __repr__(self) -> str:
        if self.formatter != '{}':
            return 'DieRolled(value={value!r}, formatter={formatter!r})'.format(
                value=int(self),
                formatter=self.formatter,
            )
        else:
            return 'DieRolled({value!r})'.format(value=int(self))

def normalize_dice_roll_list(value: List[Any]) -> List[int]:
    result = []
    for x in value:
        if isinstance(x, int):
            result.append(x)
        else:
            result.append(int(x))
    return result

def format_dice_roll_list(rolls: List[int], always_list: bool = False) -> str:
    if len(rolls) == 0:
        raise ValueError('Need at least one die rolled')
    elif len(rolls) == 1 and not always_list:
        return color(str(rolls[0]), DETAIL_COLOR)
    else:
        return '[' + color(" ".join(map(str, rolls)), DETAIL_COLOR) + ']'

def int_or_none(x: OptionalType[Any]) -> OptionalType[int]:
    if x is None:
        return None
    else:
        return int(x)

@attr.s
class DiceRolled(object):
    '''Class representing the result of rolling one or more similar dice.'''
    dice_results: List[int] = attr.ib(converter = normalize_dice_roll_list)
    @dice_results.validator
    def validate_dice_results(self, attribute, value):
        if len(value) == 0:
            raise ValueError('Need at least one non-dropped roll')
    dropped_results: List[int] = attr.ib(
        default = attr.Factory(list),
        converter = normalize_dice_roll_list)
    roll_desc: str = attr.ib(default = '', converter = str)
    success_count: OptionalType[int] = attr.ib(default = None, converter = int_or_none)

    def total(self) -> int:
        if self.success_count is not None:
            return int(self.success_count)
        else:
            return sum(self.dice_results)

    def __str__(self) -> str:
        if self.roll_desc:
            prefix = '{roll} rolled'.format(roll=color(self.roll_desc, EXPR_COLOR))
        else:
            prefix = 'Rolled'
        if self.dropped_results:
            drop = ' (dropped {dropped})'.format(dropped = format_dice_roll_list(self.dropped_results))
        else:
            drop = ''
        if self.success_count is not None:
            tot = ', Total successes: ' + color(str(self.total()), DETAIL_COLOR)
        elif len(self.dice_results) > 1:
            tot = ', Total: ' + color(str(self.total()), DETAIL_COLOR)
        else:
            tot = ''
        return '{prefix}: {results}{drop}{tot}'.format(
            prefix=prefix,
            results=format_dice_roll_list(self.dice_results),
            drop=drop,
            tot=tot,
        )

    def __int__(self) -> int:
        return self.total()

    def __float__(self) -> float:
        return float(self.total())

def validate_by_parser(parser):
    '''Return a validator that validates anything parser can parse.'''
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
    operator: str = attr.ib(converter = str,
                            validator = validate_by_parser(comparator_type))
    value: int = attr.ib(converter = int,
                         validator = validate_by_parser(positive_int))

    def __str__(self) -> str:
        return '{op}{val}'.format(op=self.operator, val=self.value)

    def compare(self, x) -> bool:
        '''Return True if x satisfies the comparator.

        In other words, x is placed on the left-hand side of the
        comparison, the Comparator's value is placed on the right hand
        side, and the truth value of the resulting test is returned.

        '''
        return self.cmp_dict[self.operator](x, self.value)

@attr.s
class RerollSpec(object):
    # Yes, it has to be called type
    type: str = attr.ib(converter = str, validator=validate_by_parser(reroll_type))
    operator: OptionalType[str] = attr.ib(default = None)
    value: OptionalType[int] = attr.ib(default = None)

    def __attrs_post_init__(self):
        if (self.operator is None) != (self.value is None):
            raise ValueError('Operator and value must be provided together')

    def __str__(self) -> str:
        result = self.type
        if self.operator is not None:
            result += self.operator + str(self.value)
        return result

    def roll_die(self, sides: DieFaceType) -> List[int]:
        '''Roll a single die, following specified re-rolling rules.

        Returns a list of rolls, since some types of re-rolling
        collect the result of multiple die rolls.

        '''
        if is_fate_face(sides):
            raise ValueError("Re-rolling/exploding is incompatible with Fate dice")
        sides = int(sides)

        cmpr: Comparator
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
            all_rolls: List[int] = [ roll_die(sides) ]
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
        else:
            raise Exception('Unknown reroll type: {}'.format(self.type))

@attr.s
class DropSpec(object):
    # Yes, it has to be called type
    type: str = attr.ib(converter = str, validator=validate_by_parser(drop_type))
    count: int = attr.ib(default = 1, converter = int, validator=validate_by_parser(positive_int))

    def __str__(self) -> str:
        if self.count > 1:
            return self.type + str(self.count)
        else:
            return self.type

    def drop_rolls(self, rolls: List[int]) -> Tuple[List[int], List[int]]:
        '''Drop the appripriate rolls from a list of rolls.

        Returns a 2-tuple of roll lists. The first list is the kept
        rolls, and the second list is the dropped rolls.

        The order of the rolls is not preserved. (TODO FIX THIS)

        '''
        if not isinstance(rolls, list):
            rolls = list(rolls)
        keeping = self.type in ('K', 'k')
        if keeping:
            num_to_keep = self.count
        else:
            num_to_keep = len(rolls) - self.count
        if num_to_keep == 0:
            raise ValueError('Not enough rolls: would drop all rolls')
        elif num_to_keep == len(rolls):
            raise ValueError('Keeping too many rolls: would not drop any rolls')
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
    die_type: DieFaceType = attr.ib(converter = normalize_die_type)
    dice_count: int = attr.ib(default = 1, converter = normalize_dice_count)
    reroll_spec: OptionalType[RerollSpec] = attr.ib(default = None)
    @reroll_spec.validator
    def validate_reroll_spec(self, attribute, value):
        if value is not None:
            assert isinstance(value, RerollSpec)
    drop_spec: OptionalType[DropSpec] = attr.ib(default = None)
    @drop_spec.validator
    def validate_drop_spec(self, attribute, value):
        if value is not None:
            assert isinstance(value, DropSpec)
    success_comparator: OptionalType[Comparator] = attr.ib(default = None)
    failure_comparator: OptionalType[Comparator] = attr.ib(default = None)
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

    def __str__(self) -> str:
        return '{count}d{type}{reroll}{drop}{success}{fail}'.format(
            count = self.dice_count if self.dice_count > 1 else '',
            type = self.die_type,
            reroll = self.reroll_spec or '',
            drop = self.drop_spec or '',
            success = self.success_comparator or '',
            fail = ('f' + str(self.failure_comparator)) if self.failure_comparator else '',
        )

    def roll(self) -> DiceRolled:
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
        success_count: OptionalType[int]
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

def make_dice_roller(expr: Union[str,ParseResults]) -> DiceRoller:
    if isinstance(expr, str):
        expr = roll_spec.parseString(expr, True)['roll']
    assert expr.getName() == 'roll'
    expr = expr.asDict()

    dtype = normalize_die_type(expr['die_type'])
    dcount = normalize_dice_count(expr['dice_count'])
    constructor_args: Dict[str, Any] = {
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

op_dict: Dict[str, Callable] = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '×': operator.mul,
    '/': operator.truediv,
    '÷': operator.truediv,
    '**': operator.pow,
    '^': operator.pow,
}

NumericType = Union[float,int]
ExprType = Union[NumericType, str, ParseResults]

def normalize_expr(expr: ExprType) -> ParseResults:
    if isinstance(expr, str):
        return expr_parser.parseString(expr)['expr']
    elif isinstance(expr, Number):
        return expr
    else:
        assert isinstance(expr, ParseResults)
        return expr['expr']

def _eval_expr_internal(
        expr: ExprType,
        env: Dict[str, str] = {},
        print_rolls: bool = True,
        recursed_vars: Set[str] = set()) -> NumericType:
    if isinstance(expr, float) or isinstance(expr, int):
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
    else:
        assert isinstance(expr, ParseResults)
        if 'operator' in expr:
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
            return result
        else:
            # roll specification
            roller = make_dice_roller(expr)
            rolled = roller.roll()
            if print_rolls:
                print(rolled)
            return int(rolled)

def eval_expr(expr: ExprType,
              env: Dict[str,str] = {},
              print_rolls: bool = True) -> NumericType:
    expr = normalize_expr(expr)
    return _eval_expr_internal(expr, env, print_rolls)

def _expr_as_str_internal(expr: ExprType,
                          env: Dict[str,str] = {},
                          recursed_vars: Set[str] = set()) -> str:
    if isinstance(expr, float) or isinstance(expr, int):
        return '{:g}'.format(expr)
    elif isinstance(expr, str):
        # variable name
        if expr in recursed_vars:
            raise ValueError('Recursive variable definition detected for {!r}'.format(expr))
        elif expr in env:
            var_value = env[expr]
            parsed = normalize_expr(var_value)
            return _expr_as_str_internal(parsed, env, recursed_vars = recursed_vars.union([expr]))
        # Not a variable name, just a string
        else:
            return expr
    else:
        assert isinstance(expr, ParseResults)
        if 'operator' in expr:
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

def expr_as_str(expr: ExprType, env: Dict[str,str]  = {}) -> str:
    expr = normalize_expr(expr)
    expr = _expr_as_str_internal(expr, env)
    if expr.startswith('(') and expr.endswith(')'):
        expr = expr[1:-1]
    return expr

def read_roll(handle: TextIO = sys.stdin) -> str:
    if handle == sys.stdin:
        return input("Enter roll> ")
    else:
        return handle.readline()[:-1]

special_command_parser: ParserElement = (
    oneOf('h help ?').setResultsName('help') |
    oneOf('q quit exit').setResultsName('quit') |
    oneOf('v vars').setResultsName('vars') |
    (oneOf('d del delete').setResultsName('delete').leaveWhitespace() + Suppress(White()) + var_name)
)

def var_name_allowed(vname: str) -> bool:
    '''Disallow variable names like 'help' and 'quit'.'''
    parsers = [ special_command_parser, roll_spec ]
    for parser in [ special_command_parser, roll_spec ]:
        try:
            parser.parseString(vname, True)
            return False
        except ParseException:
            pass
    # If the variable name didn't parse as anything else, it's valid
    return True

line_parser: ParserElement = (
    special_command_parser ^
    (assignment_parser | expr_parser)
)

def print_interactive_help() -> None:
    print('\n' + '''

To make a roll, type in the roll in dice notation, e.g. '4d4 + 4'.
Nearly all dice notation forms listed in the following references should be supported:

- http://rpg.greenimp.co.uk/dice-roller/
- https://www.critdice.com/roll-advanced-dice

Expressions can include numeric constants, addition, subtraction,
multiplication, division, and exponentiation.

To assign a variable, use 'VAR = VALUE'. For example 'health_potion =
2d4+2'. Subsequent roll expressions (and other variables) can refer to
this variable, whose value will be substituted in to the expression.

If a variable's value includes any dice rolls, those dice will be
rolled (and produce a different value) every time the variable is
used.

Special commands:

- To show the values of all currently assigned variables, type 'vars'.
- To delete a previously defined variable, type 'del VAR'.
- To show this help text, type 'help'.
- To quit, type 'quit'.

    '''.strip() + '\n', file=sys.stdout)

def print_vars(env: Dict[str,str]) -> None:
    if len(env):
        print('Currently defined variables:')
        for k in sorted(env.keys()):
            print('{} = {!r}'.format(k, env[k]))
    else:
        print('No variables are currently defined.')

if __name__ == '__main__':
    expr_string = " ".join(sys.argv[1:])
    if re.search("\\S", expr_string):
        try:
            # Note: using expr_parser instead of line_parser, because
            # on the command line only roll expressions are valid.
            expr = expr_parser.parseString(expr_string, True)
            result = eval_expr(expr)
            print('Result: {result} (rolled {expr})'.format(
                expr=color(expr_as_str(expr), EXPR_COLOR),
                result=color("{:g}".format(result), RESULT_COLOR),
            ))
        except Exception as exc:
            logger.error("Error while rolling: %s", repr(exc))
            raise exc
            sys.exit(1)
    else:
        env: Dict[str, str] = {}
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
                        print('Deleting saved value for "{var}".'.format(var=color(vname, RESULT_COLOR)))
                        del env[vname]
                    else:
                        logger.error('Variable "{var}" is not defined.'.format(var=color(vname, RESULT_COLOR)))
                elif re.search("\\S", expr_string):
                    if 'assignment' in parsed:
                        # We have an assignment operation
                        vname = parsed['varname']
                        if var_name_allowed(vname):
                            env[vname] = expr_as_str(parsed['expr'])
                            print('Saving "{var}" as "{expr}"'.format(
                                var=color(vname, RESULT_COLOR),
                                expr=color(env[vname], EXPR_COLOR),
                            ))
                        else:
                            logger.error('You cannot use {!r} as a variable name.'.format(vname))
                    else:
                        # Just an expression to evaluate
                        result = eval_expr(parsed['expr'], env)
                        print('Result: {result} (rolled {expr})'.format(
                            expr=color(expr_as_str(parsed, env), EXPR_COLOR),
                            result=color("{:g}".format(result), RESULT_COLOR),
                        ))
                print('')
            except KeyboardInterrupt:
                print('')
            except EOFError:
                print('')
                logger.info('Quitting.')
                break
            except Exception as exc:
                logger.error('Error while evaluating {expr!r}:\n{tb}'.format(
                    expr=expr_string,
                    tb=traceback.format_exc(),
                ))
