# roll.py: A command-line dice-rolling simulator for RPGs

This was a weekend hobby project inspired by [Critical Role][1] and
other D&D shows and podcasts. While a dice simulator is no substitute
for physically rolling real dice, it might be useful for anyone who's
on the go or otherwise unable to access their dice and a flat surface
to roll them on (or for new players who want to try out RPGs without
having to spend money on dice first). It's also great for when you
finally get to cast that [*Meteor
Swarm*](https://www.dndbeyond.com/spells/meteor-swarm) spell and don't
want to make everyone wait while you roll 40d6 for the damage.

It supports almost everything in the ["Standard Notation"][2] section
of the Wikipedia page on dice notation, as well as the syntax in this
online [RPG Dice Roller][3] and this [Android App][4].

[1]: http://geekandsundry.com/shows/critical-role/
[2]: https://en.wikipedia.org/wiki/Dice_notation#Standard_notation
[3]: http://rpg.greenimp.co.uk/dice-roller/
[4]: https://www.critdice.com/roll-advanced-dice

## Usage

There are two ways to use this script. Either run it with any number
of arguments, which will be concatenated and rolled, or run it with no
arguments to enter interactive mode, where you can type a roll on each
line and hit enter to roll it. In either case, the last line indicates
the total roll, and all the lines before it indicate the individual
dice rolls that led to it. Examples:

```
# Single-roll command-line mode
$ ./roll.py 2d4+2
2d4 rolled: [1 4], Total: 5
Result: 7 (rolled 2d4+2)

# Interactive mode
$ ./roll.py
# Advantage roll (drop lowest) in D&D, with a modifier
Enter roll> 2d20-L+2
2d20-L rolled: 15 (dropped 11)
Result: 17 (rolled 2d20-L+2)

# Some more basic rolls
Enter roll> 6d6+6
6d6 rolled: [5 2 6 1 1 5], Total: 20
Result: 26 (rolled 6d6+6)
Enter roll> d100
d100 rolled: 35
Result: 35 (rolled d100)

# Complex arithmetic expressions and non-standard dice are allowed
Enter roll> d4 + 2d6 + 4 - 3d8 + 6d7/2
d4 rolled: 3
2d6 rolled: [6 6], Total: 12
3d8 rolled: [1 3 2], Total: 6
6d7 rolled: [1 2 5 7 5 1], Total: 21
Result: 23.5 (rolled d4 + 2d6 + 4 - 3d8 + 6d7/2)

# Any arithmetic expression is allowed, even if it doesn't roll any dice
Enter roll> 4 + 6^2 / 3
Result: 16 (rolled 4 + 6^2 / 3)

# You can save specific rolls as named variables
Enter roll> health_potion = 2d4 + 2
Saving "health_potion" as "2d4 + 2"

Enter roll> health_potion
2d4 rolled: [3 4], Total: 7
Result: 9 (rolled 2d4 + 2)

# Variables can reference other variables
Enter roll> greater_health_potion = health_potion + health_potion
Saving "greater_health_potion" as "health_potion + health_potion"

Enter roll> greater_health_potion
2d4 rolled: [3 3], Total: 6
2d4 rolled: [1 3], Total: 4
Result: 14 (rolled 2d4 + 2 + 2d4 + 2)

# Show currently defined variables
Enter roll> vars
Currently defined variables:
greater_health_potion = 'health_potion + health_potion'
health_potion = '2d4 + 2'

# Delete a variable
Enter roll> del greater_health_potion
Deleting saved value for "greater_health_potion".

# Reroll ones once (rerolled dice are marked with 'r')
Enter roll> 4d4r
4d4r rolled: [2 3r 4 4r], Total: 13
Result: 13 (rolled 4d4r)

# Reroll ones indefinitely (rerolls marked with 'R')
Enter roll> 4d4R
4d4R rolled: [2R 3R 4R 3], Total: 12
Result: 12 (rolled 4d4R)

# Exploding dice (marked with '!')
Enter roll> 6d6!
6d6! rolled: [4 2 4 4 5 6! 1], Total: 26
Result: 26 (rolled 6d6!)

# Explode any roll greater than 3
Enter roll> 6d6!>3
6d6!>3 rolled: [2 4! 5! 4! 2 6! 5! 6! 6! 6! 3 4! 2 4! 4! 1 2], Total: 66
Result: 66 (rolled 6d6!>3)

# Penetrating dice
Enter roll> 6d6!p
6d6!p rolled: [2 4 6!p 2 6!p 0 1 2], Total: 23
Result: 23 (rolled 6d6!p)

# Fate dice
Enter roll> 4dF+2
4dF rolled: [1 0 1 -1], Total: 1
Result: 3 (rolled 4dF+2)

# Enter 'help' for other non-roll commands, 'quit' to quit
Enter roll> quit

2018-12-15 09:56:33 INFO: Quitting.
```

As you can see, it not only reports the total for each roll, but all
the individual dice rolls as well, so you can understand how it came
up with the total. This also lets you figure out whether you rolled a
natural 20 or natural 1 on the die for critical hits and misses, and
if your damage roll involves multiple different kinds of damage, it
lets you figure out how much of each damage type was dealt.
