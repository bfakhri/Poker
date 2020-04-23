# Automatic Poker Hand Evaluator

System to provide useful information during virtual poker games automatically. 

A card classification model trained on an automatically generated synthetic dataset detects the cards present in an image. When setup, it detects cards in a player's hand and community cards (for Texas Hold'em poker). The player's hand and the community cards are evaluated for strength using the treys poker hand evaluator: [Github](https://github.com/ihendley/treys).

## Install
```
python3 -m venv virtualenv

source virtualenv/bin/activate

pip install -r requirements.txt
```

## Training the Model
```
python train_model.py

tensorboard --logdir=./logs/
```

To monitor training, point your browser to 127.0.0.1:6006. A decent model takes about a day or more to train on an Nvidia GTX 1080 ti. 

## Playing with the Trained Model

To use, run ``` python3 board_eval_cv.py ```, two windows will appear and your mouse controls the smaller one. 

Move the mouse so that your two cards (your "hand") appear in view. Use the arrow keys to adjust the size of the window to snuggly fit your cards. Press ENTER when finished adjusting. 

The do the same for the second window, this is should capture the "community" or "board" cards. 

The windows should look like this: 

![alt text](https://raw.githubusercontent.com/bfakhri/Poker/master/docs/hand_board.png) 

The card classification model will infer the cards in your hand and on the board and evaluate both with the treys poker hand evaluation library. The output will look something like this: 

![alt text](https://raw.githubusercontent.com/bfakhri/Poker/master/docs/model_and_eval_output.png) 
