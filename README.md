# Automatic Poker Hand Evaluator
Evaluates Poker Hands with the Treys Evaluator: [Github](https://github.com/ihendley/treys).

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
![alt text](https://github.com/bfakhri/tf/blob/master/docs/windows.png) 

The card classification model will infer the cards in your hand and on the board and evaluate both with the treys poker hand evaluation library. The output will look something like this: 
![alt text](https://github.com/bfakhri/tf/blob/master/docs/model_and_hand_output.png) 
