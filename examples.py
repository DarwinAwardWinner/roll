test_parse(Number, '+6.0223e23')
test_parse(RollExpr, [
    '4d4',
    '2d20K',
    '8d6x1',
    '8d4!p<=1',
    '8d4r4',
    '8d6r1>3f<3',
])
test_parse(Expression, [
    'x+1',
    '4d4+4',
    '2*2',
    '(2*2)',
    '2d20K + d6 + (2 * 2 ^ 2)',
])
test_parse(VarAssignment, [
    'x= 5',
    'int = d20 + 7',
])
test_parse(InputParser, [
    '4d4',
    '2d20K',
    '8d6x1',
    '8d4!p<=1',
    '8d4r4',
    '8d6r1>3f<3',
    'x+1',
    '4d4+4',
    '2*2',
    '(2*2)',
    '2d20K + d6 + (2 * 2 ^ 2)',
    'x= 5',
    'int = d20 + 7',
    'del x',
    'delete x',
    'help',
    'quit',
    'v',
])


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
# ]

# example_results = {}
# for x in examples:
#     try:
#         example_results[x] = parse_roll(x)
#     except ParseException as ex:
#         example_results[x] = ex
# example_results
