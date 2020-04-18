# Takes a hand and board combination and evaluates it #

from treys import Card
from treys import Deck
from treys import Evaluator

board = [
    Card.new('Ah'),
    Card.new('Kd'),
    Card.new('Jc')
]
hand = [
   Card.new('Qs'),
   Card.new('Th')
]

print(Card.print_pretty_cards(board + hand))
