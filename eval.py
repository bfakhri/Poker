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

# Hand strength is valued on a scale of 1 to 7462, where 1 is a Royal Flush and 7462 is unsuited 7-5-4-3-2
evaluator = Evaluator()
print(evaluator.evaluate(board, hand))
