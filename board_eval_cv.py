###############################################################################
# Sets up windows to capture screenshot of player hand and community board    # 
# The hand/board images are sent to a model to infer the cards in the image   #
# The inferred cards are evaluated for strength and displayed for the user    #
###############################################################################

import numpy as np
import cv2
from PIL import Image
import mss
import mss.tools
import pynput
import time
import threading
import tensorflow as tf
import utils
from treys import Card
from treys import Evaluator

from pynput import mouse
from pynput import keyboard

class ScreenScraper:
    def __init__(self):
        # Card detection model
        model_dir = './saved_models/'
        model_path, step = utils.find_latest_model(model_dir)
        self.model = tf.keras.models.load_model(model_path)

        # Switches
        self.move_hand = True
        self.move_board = False 

        # Card Game Params
        self.hand_evaluator = Evaluator()
        self.num_distinct_hands = 7462

        # Window Params
        self.min_width = 10
        self.min_height = 10
        self.max_width = 2000
        self.max_height = 2000
        self.first_windows = True

        # Format is [x,y,w,h]
        self.hand_window = [0,0,100,80]
        self.board_window = [0,0,268,80]

        # Start watcher thread
        watch = threading.Thread(target=self.window_watcher)
        watch.start()

        # Setup input listeners:
        self.key_listener = keyboard.Listener(on_press=self.key_on_press)
        self.key_listener.start()
        self.listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move)
        self.listener.start()

        # Fix windows
        self.init_windows()


    def init_windows(self):
        # Set switches
        self.move_hand = True
        self.move_board = False 

        # Gets the position of the user's cards
        print('Move mouse to top left-hand corner of hand')
        print('Use the arrow keys to adjust window')
        print('PRESS ENTER WHEN DONE')
        while(self.move_hand):
            print('Waiting on hand...')
            time.sleep(1)

        # Gets the position of the board
        self.move_hand = False
        self.move_board = True 
        print('Move mouse to top left-hand corner of board')
        print('Use the arrow keys to adjust window')
        print('PRESS ENTER WHEN DONE')
        while(self.move_board):
            print('Waiting on board...')
            time.sleep(1)
        self.move_board = False 

    def window_watcher(self):
        while(True):
            self.draw_windows()
            time.sleep(0.01)

    def draw_windows(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor_hand = {"top": self.hand_window[1], "left": self.hand_window[0], "width": self.hand_window[2], "height": self.hand_window[3]}
            monitor_board = {"top": self.board_window[1], "left": self.board_window[0], "width": self.board_window[2], "height": self.board_window[3]}
            # Grab the data
            try:
                sct_img_hand = np.asarray(sct.grab(monitor_hand))
                sct_img_board = np.asarray(sct.grab(monitor_board))
            except mss.exception.ScreenShotError:
                print('Screentshot Failed')
                sct_img_hand = np.zeros((200,400, 3))
                sct_img_board = np.zeros((200,400, 3)) 
            cv2.imshow('hand', sct_img_hand)
            cv2.imshow('board', sct_img_board)
            if(self.first_windows):
                cv2.moveWindow('hand', 600, 100)
                cv2.moveWindow('board', 600, 400)
                self.first_windows = False

            # Get Predictions from Models
            pred_hand = tf.squeeze(self.model(sct_img_hand[np.newaxis,...,0:3]/255.0))
            pred_board = tf.squeeze(self.model(sct_img_board[np.newaxis,...,0:3]/255.0))
            pred_hand_cards = utils.cards_from_preds(pred_hand, 2)
            pred_board_cards = utils.cards_from_preds(pred_board, 3)
            hand_str = Card.print_pretty_cards(pred_hand_cards)
            board_str = Card.print_pretty_cards(pred_board_cards)
            hand_rank = self.hand_evaluator.evaluate(pred_board_cards, pred_hand_cards)
            strength_str = 'Strength: {0}, Win: {1:3.2f}%'.format(hand_rank, 100.0*hand_rank/self.num_distinct_hands)
            print('Hand: ',  hand_str, '  Board: ', board_str, strength_str)
            cv2.waitKey(1)

    def on_move(self, x, y):
        if(self.move_hand):
            self.hand_window[0:2] = [x,y]
        if(self.move_board):
            self.board_window[0:2] = [x,y]


    def on_click(self, x, y, button, pressed):
        pass

    def key_on_press(self, key):
        ''' Edit window size based on arrow keys '''
        delta = 4
        if(key == keyboard.Key.enter):
            print('Done moving:')
            self.move_hand = False
            self.move_board = False
        if(self.move_hand):
            if(key == keyboard.Key.up):
                self.hand_window[3] = np.clip(self.hand_window[3]+delta, self.min_height, self.max_height)
            if(key == keyboard.Key.down):
                self.hand_window[3] = np.clip(self.hand_window[3]-delta, self.min_height, self.max_height)
            if(key == keyboard.Key.left):
                self.hand_window[2] = np.clip(self.hand_window[2]-delta, self.min_height, self.max_height)
            if(key == keyboard.Key.right):
                self.hand_window[2] = np.clip(self.hand_window[2]+delta, self.min_height, self.max_height)
        if(self.move_board):
            if(key == keyboard.Key.up):
                self.board_window[3] = np.clip(self.board_window[3]+delta, self.min_height, self.max_height)
            if(key == keyboard.Key.down):
                self.board_window[3] = np.clip(self.board_window[3]-delta, self.min_height, self.max_height)
            if(key == keyboard.Key.left):
                self.board_window[2] = np.clip(self.board_window[2]-delta, self.min_height, self.max_height)
            if(key == keyboard.Key.right):
                self.board_window[2] = np.clip(self.board_window[2]+delta, self.min_height, self.max_height)




if __name__ == '__main__':
    ss = ScreenScraper()
    s = input('')
