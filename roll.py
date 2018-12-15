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
from copy import copy

from arpeggio import ParserPython, RegExMatch, Optional, ZeroOrMore, OneOrMore, OrderedChoice, Sequence, Combine, Not, EOF, PTNodeVisitor, visit_parse_tree, ParseTreeNode, SemanticActionResults

from typing import Union, List, Any, Tuple, Dict, Callable, Set, TextIO
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

# Implementing the syntax described here: https://www.critdice.com/roll-advanced-dice
# https://stackoverflow.com/a/23956778/125921

# Whitespace parsing
def Whitespace(): return RegExMatch(r'\s+')
def OpWS(): return Optional(Whitespace)

# Number parsing
def Digits(): return RegExMatch('[0-9]+')
def NonzeroDigits():
    '''Digits with at least one nonzero number.'''
    return RegExMatch('0*[1-9][0-9]*')
def Sign(): return ['+', '-']
def Integer(): return Optional(Sign), Digits
def PositiveInteger(): return Optional('+'), Digits
def FloatingPoint():
    return (
        Optional(Sign),
        [
            # e.g. '1.', '1.0'
            (Digits, '.', Optional(Digits)),
            # e.g. '.1'
            ('.', Digits),
        ]
    )
def Scientific():
    return ([FloatingPoint, Integer], RegExMatch('[eE]'), Integer)
def Number(): return Combine([Scientific, FloatingPoint, Integer])

def ReservedWord():
    '''Matches identifiers that aren't allowed as variable names.'''
    command_word_parsers = []
    for cmd_type in Command():
        cmd_parser = cmd_type()
        if isinstance(cmd_parser, tuple):
            command_word_parsers.append(cmd_parser[0])
        else:
            command_word_parsers.append(cmd_parser)
    return([
        # Starts with a roll expression
        RollExpr,
        # Matches a command word exactly
        (command_word_parsers, [RegExMatch('[^A-Za-z0-9_]'), EOF]),
    ])

# Valid variable name parser (should disallow names like 'help', 'quit', or 'd4r')
def Identifier(): return (
        Not(ReservedWord),
        RegExMatch(r'[A-Za-z_][A-Za-z0-9_]*')
)

def MyNum(): return (
        Not('0'),
        RegExMatch('[0-9]+'),
)

# Roll expression parsing
def PercentileFace(): return '%'
def DieFace(): return [NonzeroDigits, PercentileFace, RegExMatch(r'F(\.[12])?')]
def BasicRollExpr():
    return (
        Optional(NonzeroDigits),
        RegExMatch('[dD]'),
        DieFace,
    )
def DropSpec(): return 'K k X x -H -L'.split(' '), Optional(NonzeroDigits)
def CompareOp(): return '<= < >= > ≤ ≥ ='.split(' ')
def Comparison(): return CompareOp, Integer
def RerollType(): return Combine(['r', 'R', ('!', Optional('!'), Optional('p'))])
def RerollSpec():
    return  (
        RerollType,
        Optional(
            Optional(CompareOp),
            Integer,
        ),
    )
def CountSpec():
    return (
        Comparison,
        Optional('f', Comparison),
    )

def RollExpr():
    return (
        BasicRollExpr,
        Optional([DropSpec, RerollSpec]),
        Optional(CountSpec),
    )

# Arithmetic expression parsing
def PrimaryTerm(): return [RollExpr, Number, Identifier]
def TermOrGroup(): return [PrimaryTerm, ParenExpr]
def Exponent(): return ['**', '^'], OpWS, TermOrGroup
def ExponentExpr(): return TermOrGroup, ZeroOrMore(OpWS, Exponent)
def Mul(): return ['*', '×'], OpWS, ExponentExpr
def Div(): return ['/', '÷'], OpWS, ExponentExpr
def ProductExpr(): return ExponentExpr, ZeroOrMore(OpWS, [Mul, Div])
def Add(): return '+', OpWS, ProductExpr
def Sub(): return '-', OpWS, ProductExpr
def SumExpr(): return ProductExpr, ZeroOrMore(OpWS, [Add, Sub])
def ParenExpr(): return Optional(Sign), '(', OpWS, SumExpr, OpWS, ')'
def Expression():
    # Wrapped in a tuple to force a separate entry in the parse tree
    return (SumExpr,)

# For parsing vars/expressions without evaluating them. The Combine()
# hides the child nodes from a visitor.
def UnevaluatedExpression(): return Combine(Expression)
def UnevaluatedVariable(): return Combine(Identifier)

# Variable assignment
def VarAssignment(): return (
        UnevaluatedVariable,
        OpWS, '=', OpWS,
        UnevaluatedExpression
)

# Commands
def DeleteCommand(): return (
        Combine(['delete', 'del', 'd']),
        Whitespace,
        UnevaluatedVariable,
)
def HelpCommand(): return Combine(['help', 'h', '?'])
def QuitCommand(): return Combine(['quit', 'exit', 'q'])
def ListVarsCommand(): return Combine(['variables', 'vars', 'v'])
def Command(): return [ ListVarsCommand, DeleteCommand, HelpCommand, QuitCommand, ]

def InputParser(): return Optional([Command, VarAssignment, Expression, Whitespace])

def FullParserPython(language_def, *args, **kwargs):
    '''Like ParserPython, but auto-adds EOF to the end of the parser.'''
    def TempFullParser(): return (language_def, EOF)
    return ParserPython(TempFullParser, *args, **kwargs)

expr_parser = FullParserPython(Expression, skipws = False, memoization = True)
input_parser = FullParserPython(InputParser, skipws = False, memoization = True)

def test_parse(rule, text):
    if isinstance(text, str):
        return FullParserPython(rule, skipws=False, memoization = True).parse(text)
    else:
        return [ test_parse(rule, x) for x in text ]


def eval_infix(terms: List[float],
               operators: List[Callable[[float,float],float]],
               associativity: str = 'l') -> float:
    '''Evaluate an infix expression with N terms and N-1 operators.'''
    assert associativity in ['l', 'r']
    assert len(terms) == len(operators) + 1, 'Need one more term than operator'
    if len(terms) == 1:
        return terms[0]
    elif associativity == 'l':
        value = terms[0]
        for op, term in zip(operators, terms[1:]):
            value = op(value, term)
        return value
    elif associativity == 'r':
        value = terms[-1]
        for op, term in zip(reversed(operators), reversed(terms[:-1])):
            value = op(term, value)
        return value
    else:
        raise ValueError(f'Invalid associativity: {associativity!r}')

class UndefinedVariableError(KeyError):
    pass

def print_vars(env: Dict[str,str]) -> None:
    if len(env):
        print('Currently defined variables:')
        for k in sorted(env.keys()):
            print('{} = {!r}'.format(k, env[k]))
    else:
        print('No variables are currently defined.')

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

DieFaceType = Union[int, str]
def roll_die(face: DieFaceType = 6) -> int:
    '''Roll a single die.

Supports any valid integer number of sides as well as 'F', 'F.1', and
'F.2' for a Face die, which can return -1, 0, or 1.

    '''
    if face in ('F', 'F.2'):
        # Fate die = 1d3-2
        return roll_die(3) - 2
    elif face == 'F.1':
        d6 = roll_die(6)
        if d6 == 1:
            return -1
        elif d6 == 6:
            return 1
        else:
            return 0
    else:
        face = int(face)
        if face < 2:
            raise ValueError(f"Can't roll a {face}-sided die")
        return randint(1, face)

def roll_die_with_rerolls(face: int, reroll_condition: Callable, reroll_limit = None) -> List[int]:
    '''Roll a single die, and maybe reroll it several times.

    After each roll, 'reroll_condition' is called on the result, and
    if it returns True, the die is rolled again. All rolls are
    collected in a list, and the list is returned as soon as the
    condition returns False.

    If reroll_limit is provided, it limits the maximum number of
    rerolls. Note that the total number of rolls can be one more than
    the reroll limit, since the first roll is not considered a reroll.

    '''
    all_rolls = [roll_die(face)]
    while reroll_condition(all_rolls[-1]):
        if reroll_limit is not None and len(all_rolls) > reroll_limit:
            break
        all_rolls.append(roll_die(face))
    return all_rolls

class DieRolled(int):
    '''Subclass of int that allows an alternate string representation.

    This is meant for recording the result of rolling a die. The
    formatter argument should include '{}' anywhere that the integer
    value should be substituted into the string representation.
    (However, it can also override the string representation entirely
    by not including '{}'.) The string representation has no effect on
    the numeric value. It can be used to indicate a die roll that has
    been re-rolled or exploded, or to indicate a critical hit/miss.

    '''
    formatter: str

    def __new__(cls: type, value: int, formatter: str = '{}') -> 'DieRolled':
        # https://github.com/python/typeshed/issues/2686
        newval = super(DieRolled, cls).__new__(cls, value) # type: ignore
        newval.formatter = formatter
        return newval

    def __str__(self) -> str:
        return self.formatter.format(super().__str__())

    def __repr__(self) -> str:
        if self.formatter != '{}':
            return f'DieRolled(value={int(self)!r}, formatter={self.formatter!r})'
        else:
            return f'DieRolled({int(self)!r})'

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

def str_or_none(x: OptionalType[Any]) -> OptionalType[str]:
    if x is None:
        return None
    else:
        return str(x)

@attr.s
class DiceRolled(object):
    '''Class representing the result of rolling one or more similar dice.'''
    dice_results: List[int] = attr.ib()
    @dice_results.validator
    def validate_dice_results(self, attribute, value):
        if len(value) == 0:
            raise ValueError('Need at least one non-dropped roll')
    dropped_results: OptionalType[List[int]] = attr.ib(default = None)
    roll_text: OptionalType[str] = attr.ib(
        default = None, converter = str_or_none)
    success_count: OptionalType[int] = attr.ib(
        default = None, converter = int_or_none)

    def total(self) -> int:
        if self.success_count is not None:
            return int(self.success_count)
        else:
            return sum(self.dice_results)

    def __str__(self) -> str:
        results = format_dice_roll_list(self.dice_results)
        if self.roll_text:
            prefix = '{text} rolled'.format(text=color(self.roll_text, EXPR_COLOR))
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
        return f'{prefix}: {results}{drop}{tot}'

    def __int__(self) -> int:
        return self.total()

    def __float__(self) -> float:
        return float(self.total())

cmp_dict = {
    '<=': operator.le,
    '<': operator.lt,
    '>=': operator.ge,
    '>': operator.gt,
    '≤': operator.le,
    '≥': operator.ge,
    '=': operator.eq,
}

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
    operator: str = attr.ib(converter = str)
    @operator.validator
    def validate_operator(self, attribute, value):
        if not value in self.cmp_dict:
            raise ValueError(f'Unknown comparison operator: {value!r}')
    value: int = attr.ib(converter = int)

    def __str__(self) -> str:
        return '{op}{val}'.format(op=self.operator, val=self.value)

    def compare(self, x: float) -> bool:
        '''Return True if x satisfies the comparator.

        In other words, x is placed on the left-hand side of the
        comparison, the Comparator's value is placed on the right hand
        side, and the truth value of the resulting test is returned.

        '''
        return self.cmp_dict[self.operator](x, self.value)

    def __call__(self, x: float) -> bool:
        '''Calls Comparator.compare() on x.

        This allows the Comparator to be used as a callable.'''
        return self.compare(x)

def roll_dice(roll_desc: Dict) -> DiceRolled:
    '''Roll dice based on a roll description.

    See InputHandler.visit_RollExpr(), which generates roll
    descriptions. This function assumes the roll description is
    already validated.

    Returns a tuple of two lists. The first list is the kept rolls,
    and the second list is the dropped rolls.

    '''
    die_face: DieFaceType = roll_desc['die_face']
    dice_count: int = roll_desc['dice_count']
    kept_rolls: List[int] = []
    dropped_rolls: OptionalType[List[int]] = None
    success_count: Optional[int] = None
    if 'reroll_type' in roll_desc:
        die_face = int(die_face) # No fate dice
        reroll_type: str = roll_desc['reroll_type']
        reroll_limit = 1 if reroll_type == 'r' else None
        reroll_desc: Dict = roll_desc['reroll_desc']
        reroll_comparator = Comparator(
            operator = reroll_desc['comparator'],
            value = reroll_desc['target'],
        )
        for i in range(dice_count):
            current_rolls = roll_die_with_rerolls(die_face, reroll_comparator, reroll_limit)
            if len(current_rolls) == 1:
                # If no rerolls happened, then just add the single
                # roll as is.
                kept_rolls.append(current_rolls[0])
            elif reroll_type in ['r', 'R']:
                # Keep only the last roll, and mark it as rerolled
                kept_rolls.append(DieRolled(current_rolls[-1], '{}' + reroll_type))
            elif reroll_type in ['!', '!!', '!p', '!!p']:
                if reroll_type.endswith('p'):
                    # For penetration, subtract 1 from all rolls
                    # except the first
                    for i in range(1, len(current_rolls)):
                        current_rolls[i] -= 1
                if reroll_type.startswith('!!'):
                    # For compounding, return the sum, marked as a
                    # compounded roll.
                    kept_rolls.append(DieRolled(sum(current_rolls),
                                               '{}' + reroll_type))
                else:
                    # For exploding, add each individual roll to the
                    # list. Mark each roll except the last as
                    # rerolled.
                    for i in range(0, len(current_rolls) - 1):
                        current_rolls[i] = DieRolled(current_rolls[i], '{}' + reroll_type)
                    kept_rolls.extend(current_rolls)
            else:
                raise ValueError(f'Unknown reroll type: {reroll_type}')
    else:
        # Roll the requested number of dice
        all_rolls = [ roll_die(die_face) for i in range(dice_count) ]
        if 'drop_type' in roll_desc:
            keep_count: int = roll_desc['keep_count']
            keep_high: bool = roll_desc['keep_high']
            # We just need to keep the highest/lowest N rolls. The
            # extra complexity here is just to preserve the original
            # order of those rolls.
            rolls_to_keep = sorted(all_rolls, reverse = keep_high)[:keep_count]
            kept_rolls = []
            for roll in rolls_to_keep:
                kept_rolls.append(all_rolls.pop(all_rolls.index(roll)))
            # Remaining rolls are dropped
            dropped_rolls = all_rolls
        else:
            kept_rolls = all_rolls
    # Now we have a list of kept rolls
    if 'count_success' in roll_desc:
        die_face = int(die_face) # No fate dice
        success_desc = roll_desc['count_success']
        success_test = Comparator(
            operator = success_desc['comparator'],
            value = success_desc['target'],
        )
        success_count = sum(success_test(x) for x in kept_rolls)
        if 'count_failure' in roll_desc:
            failure_desc = roll_desc['count_failure']
            failure_test = Comparator(
                operator = failure_desc['comparator'],
                value = failure_desc['target'],
            )
            # Make sure the two conditions don't overlap
            for i in range(1, die_face + 1):
                if success_test(i) and failure_test(i):
                    raise ValueError(f"Can't use overlapping success and failure conditions: {str(success_test)!r}, {str(failure_test)!r}")
            success_count -= sum(failure_test(x) for x in kept_rolls)
    temp_args = dict(
        dice_results = kept_rolls,
        dropped_results = dropped_rolls,
        success_count = success_count,
        roll_text = roll_desc['roll_text'],
    )
    return DiceRolled(
        dice_results = kept_rolls,
        dropped_results = dropped_rolls,
        success_count = success_count,
        roll_text = roll_desc['roll_text'],
    )

class ExpressionStringifier(PTNodeVisitor):
    def __init__(self, **kwargs):
        self.env: Dict[str, str] = kwargs.pop('env', {})
        self.recursed_vars: Set[str] = kwargs.pop('recursed_vars', set())
        self.expr_parser = kwargs.pop('expr_parser', expr_parser)
        super().__init__(**kwargs)
    def visit__default__(self, node, children):
        if children:
            return ''.join(children)
        else:
            return node.value
    def visit_Identifier(self, node, children):
        '''Interpolate variable.'''
        var_name = node.value
        if var_name in self.recursed_vars:
            raise ValueError(f'Recursive variable definition detected for {var_name!r}')
        try:
            var_expression = self.env[var_name]
        except KeyError as ex:
            raise UndefinedVariableError(*ex.args)
        recursive_visitor = copy(self)
        recursive_visitor.recursed_vars = self.recursed_vars.union([var_name])
        return self.expr_parser.parse(var_expression).visit(recursive_visitor)

class QuitRequested(BaseException):
    pass

class InputHandler(PTNodeVisitor):
    def __init__(self, **kwargs):
        self.expr_stringifier = ExpressionStringifier(**kwargs)
        self.env: Dict[str, str] = kwargs.pop('env', {})
        self.recursed_vars: Set[str] = kwargs.pop('recursed_vars', set())
        self.expr_parser = kwargs.pop('expr_parser', expr_parser)
        self.print_results = kwargs.pop('print_results', True)
        self.print_rolls = kwargs.pop('print_rolls', True)
        super().__init__(**kwargs)

    def visit_Whitespace(self, node, children):
        '''Remove whitespace nodes'''
        return None
    def visit_Number(self, node, children):
        '''Return the numeric value.

        Uses int if possible, otherwise float.'''
        try:
            return int(node.value)
        except ValueError:
            return float(node.value)
    def visit_NonzeroDigits(self, node, children):
        return int(node.flat_str())
    def visit_Integer(self, node, children):
        return int(node.flat_str())
    def visit_PercentileFace(self, node, children):
        return 100
    def visit_BasicRollExpr(self, node, children):
        die_face = children[-1]
        if isinstance(die_face, int) and die_face < 2:
            raise ValueError(f"Invalid roll: Can't roll a {die_face}-sided die")
        return {
            'dice_count': children[0] if len(children) == 3 else 1,
            'die_face': die_face,
        }
    def visit_DropSpec(self, node, children):
        return {
            'drop_type': children[0],
            'drop_or_keep_count': children[1] if len(children) > 1 else 1,
        }
    def visit_RerollSpec(self, node, children):
        if len(children) == 1:
            return {
                'reroll_type': children[0],
                # The default reroll condition depends on other parts
                # of the roll expression, so it will be "filled in"
                # later.
            }
        elif len(children) == 2:
            return {
                'reroll_type': children[0],
                'reroll_desc': {
                    'comparator': '=',
                    'target': children[1],
                },
            }
        elif len(children) == 3:
            return {
                'reroll_type': children[0],
                'reroll_desc': {
                    'comparator': children[1],
                    'target': children[2],
                },
            }
        else:
            raise ValueError("Invalid reroll specification")
    def visit_Comparison(self, node, children):
        return {
            'comparator': children[0],
            'target': children[1],
        }
    def visit_CountSpec(self, node, children):
        result = { 'count_success': children[0], }
        if len(children) > 1:
            result['count_failure'] = children[1]
        return result
    def visit_RollExpr(self, node, children):
        # Collect all child dicts into one
        roll_desc = {
            'roll_text': node.flat_str(),
        }
        for child in children:
            roll_desc.update(child)
        logger.debug(f'Initial roll description: {roll_desc!r}')
        # Perform some validation that can only be done once the
        # entire roll description is collected.
        if not isinstance(roll_desc['die_face'], int):
            if 'reroll_type' in roll_desc:
                raise ValueError('Can only reroll/explode numeric dice, not Fate dice')
            if 'count_success' in roll_desc:
                raise ValueError('Can only count successes on numeric dice, not Fate dice')
        # Fill in implicit reroll type
        if 'reroll_type' in roll_desc and not 'reroll_desc' in roll_desc:
            rrtype = roll_desc['reroll_type']
            if rrtype in ['r', 'R']:
                roll_desc['reroll_desc'] = {
                    'comparator': '=',
                    'target': 1,
                }
            else:
                roll_desc['reroll_desc'] = {
                    'comparator': '=',
                    'target': roll_desc['die_face'],
                }
        # Validate drop spec and determine exactly how many dice to
        # drop/keep
        if 'drop_type' in roll_desc:
            dtype = roll_desc['drop_type']
            keeping = dtype in ['K', 'k']
            if keeping:
                roll_desc['keep_count'] = roll_desc['drop_or_keep_count']
            else:
                roll_desc['keep_count'] = roll_desc['dice_count'] - roll_desc['drop_or_keep_count']
            if roll_desc['keep_count'] < 1:
                drop_count = roll_desc['dice_count'] - roll_desc['keep_count']
                raise ValueError(f"Can't drop {drop_count} dice out of {roll_desc['dice_count']}")
            if roll_desc['keep_count'] >= roll_desc['dice_count']:
                raise ValueError(f"Can't keep {roll_desc['keep_count']} dice out of {roll_desc['dice_count']}")
            # Keeping high rolls is the same as dropping low rolls
            roll_desc['keep_high'] = dtype in ['K', 'x', '-L']
        # Validate count spec
        elif 'count_failure' in roll_desc and not 'count_success' in roll_desc:
            # The parser shouldn't allow this, but just in case
            raise ValueError("Can't have a failure condition without a success condition")
        logger.debug(f'Final roll description: {roll_desc!r}')
        result = roll_dice(roll_desc)
        if self.print_rolls:
            print(str(result))
        return int(result)
    def visit_Identifier(self, node, children):
        '''Interpolate variable.'''
        var_name = node.value
        if var_name in self.recursed_vars:
            raise ValueError(f'Recursive variable definition detected for {var_name!r}')
        try:
            var_expression = self.env[var_name]
        except KeyError as ex:
            raise UndefinedVariableError(*ex.args)
        recursive_visitor = copy(self)
        recursive_visitor.recursed_vars = self.recursed_vars.union([var_name])
        # Don't print the results of evaluating variables
        recursive_visitor.print_results = False
        if self.debug:
            self.dprint(f'Evaluating variable {var_name} with expression {var_expression!r}')
        return self.expr_parser.parse(var_expression).visit(recursive_visitor)
    def visit_Expression(self, node, children):
        if self.print_results:
            expr_full_text = node.visit(self.expr_stringifier)
            print('Result: {result} (rolled {expr})'.format(
                expr=color(expr_full_text, EXPR_COLOR),
                result=color(f'{children[0]:g}', RESULT_COLOR),
            ))
        return children[0]
    # Each of these returns a tuple of (operator, value)
    def visit_Add(self, node, children):
        return (operator.add, children[-1])
    def visit_Sub(self, node, children):
        return (operator.sub, children[-1])
    def visit_Mul(self, node, children):
        return (operator.mul, children[-1])
    def visit_Div(self, node, children):
        return (operator.truediv, children[-1])
    def visit_Exponent(self, node, children):
        return (operator.pow, children[-1])
    # Each of these receives a first child that is a number and the
    # remaining children are tuples of (operator, number)
    def visit_SumExpr(self, node, children):
        values = [children[0]]
        ops = []
        for (op, val) in children[1:]:
            values.append(val)
            ops.append(op)
        if self.debug:
            self.dprint(f'Sum: values: {values!r}; ops: {ops!r}')
        return eval_infix(values, ops, 'l')
    def visit_ProductExpr(self, node, children):
        values = [children[0]]
        ops = []
        for (op, val) in children[1:]:
            values.append(val)
            ops.append(op)
        if self.debug:
            self.dprint(f'Product: values: {values!r}; ops: {ops!r}')
        return eval_infix(values, ops, 'l')
    def visit_ExponentExpr(self, node, children):
        values = [children[0]]
        ops = []
        for (op, val) in children[1:]:
            values.append(val)
            ops.append(op)
        if self.debug:
            self.dprint(f'Exponent: values: {values!r}; ops: {ops!r}')
        return eval_infix(values, ops, 'l')
    def visit_Sign(self, node, children):
        if node.value == '-':
            return -1
        else:
            return 1
    def visit_ParenExpr(self, node, children):
        assert len(children) > 0
        # Multiply the sign (if present) and the value inside the
        # parens
        return functools.reduce(operator.mul, children)
    def visit_VarAssignment(self, node, children):
        logger.debug(f'Doing variable assignment: {node.flat_str()}')
        var_name, var_value = children
        print('Saving "{var}" as "{expr}"'.format(
            var=color(var_name, RESULT_COLOR),
            expr=color(var_value, EXPR_COLOR),
        ))
        self.env[var_name] = var_value
    def visit_ListVarsCommand(self, node, children):
        print_vars(self.env)
    def visit_DeleteCommand(self, node, children):
        var_name = children[-1]
        print('Deleting saved value for "{var}".'.format(
            var=color(var_name, RESULT_COLOR)))
        try:
            self.env.pop(var_name)
        except KeyError as ex:
            raise UndefinedVariableError(*ex.args)
    def visit_HelpCommand(self, node, children):
        print_interactive_help()
    def visit_QuitCommand(self, node, children):
        raise QuitRequested()

# def handle_input(expr: str, **kwargs) -> float:
#     return input_parser.parse(expr).visit(InputHandler(**kwargs))

# handle_input('help')
# handle_input('2+2 * 2 ** 2')
# env = {}
# handle_input('y = 2 + 2', env = env)
# handle_input('x = y + 2', env = env)
# handle_input('2 + x', env = env)
# handle_input('del x', env = env)
# handle_input('vars', env = env)
# handle_input('2 + x', env = env)
# handle_input('d4 = 5', env = env)

def read_input(handle: TextIO = sys.stdin) -> str:
    if handle == sys.stdin:
        return input("Enter roll> ")
    else:
        return handle.readline()[:-1]

if __name__ == '__main__':
    expr_string = " ".join(sys.argv[1:])
    if re.search("\\S", expr_string):
        try:
            expr_parser.parse(expr_string).visit(InputHandler())
        except Exception as exc:
            logger.error("Error while rolling: %s", repr(exc))
            raise exc
            sys.exit(1)
    else:
        env: Dict[str, str] = {}
        handler = InputHandler(env = env)
        while True:
            try:
                input_string = read_input()
                input_parser.parse(input_string).visit(handler)
            except KeyboardInterrupt:
                print('')
            except (EOFError, QuitRequested):
                print('')
                logger.info('Quitting.')
                break
            except Exception as exc:
                logger.error('Error while evaluating {expr!r}:\n{tb}'.format(
                    expr=expr_string,
                    tb=traceback.format_exc(),
                ))
