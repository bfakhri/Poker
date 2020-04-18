################################################################
# Captures a portion of the screen                             #
# Based on the mss library: https://python-mss.readthedocs.io/ #
################################################################

import numpy as np
import cv2
from PIL import Image
import mss
import mss.tools
import pynput
import time

from pynput import mouse
from pynput import keyboard


class ScreenScraper:
    def __init__(self):
        # Setup input listeners:
        self.key_listener = keyboard.Listener(on_press=self.key_on_press)
        self.key_listener.start()
        self.listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move)
        self.listener.start()

        # Switches
        self.move_hand = True
        self.move_board = False 

        # Window Params
        self.min_width = 10
        self.min_height = 10
        self.max_width = 2000
        self.max_height = 2000
        self.first_windows = True

        # Format is [x,y,w,h]
        self.hand_window = [0,0,self.min_width,self.min_height]
        self.board_window = [0,0,self.min_width,self.min_height]

        # Fix windows
        self.init_windows()

    def init_windows(self):
        # Set switches
        self.move_hand = True
        self.move_board = False 
        # Gets the position of the user's cards
        print('Move mouse to top left-hand corner of hand then click')
        print('Use the arrow keys to adjust window')
        while(self.move_hand):
            print('Waiting on hand...')
            time.sleep(1)
        # Gets the position of the board
        self.move_hand = False
        self.move_board = True 
        print('Move mouse to top left-hand corner of board then click')
        print('Use the arrow keys to adjust window')
        while(self.move_board):
            print('Waiting on board...')
            time.sleep(1)
        self.move_board = False 

    def draw_windows(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor_hand = {"top": self.hand_window[1], "left": self.hand_window[0], "width": self.hand_window[2], "height": self.hand_window[3]}
            monitor_board = {"top": self.board_window[1], "left": self.board_window[0], "width": self.board_window[2], "height": self.board_window[3]}
            # Grab the data
            sct_img_hand = np.asarray(sct.grab(monitor_hand))
            sct_img_board = np.asarray(sct.grab(monitor_board))
            cv2.imshow('hand', sct_img_hand)
            cv2.imshow('board', sct_img_board)
            if(self.first_windows):
                cv2.moveWindow('hand', 600, 100)
                cv2.moveWindow('board', 600, 400)
                self.first_windows = False
            cv2.waitKey(-1)

    def on_move(self, x, y):
        if(self.move_hand):
            self.hand_window[0:2] = [x,y]
        if(self.move_board):
            self.board_window[0:2] = [x,y]


    def on_click(self, x, y, button, pressed):
        print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
        if not pressed:
            print('Done moving:')
            self.move_hand = False
            self.move_board = False
            # Stop the listener
            #return False

    def key_on_press(self, key):
        if(key == keyboard.Key.enter):
            print('Done moving:')
            self.move_hand = False
            self.move_board = False
        if(self.move_hand):
            if(key == keyboard.Key.up):
                self.hand_window[3] = np.clip(self.hand_window[3]+1, self.min_height, self.max_height)
            if(key == keyboard.Key.down):
                self.hand_window[3] = np.clip(self.hand_window[3]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.left):
                self.hand_window[2] = np.clip(self.hand_window[2]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.right):
                self.hand_window[2] = np.clip(self.hand_window[2]+1, self.min_height, self.max_height)
        if(self.move_board):
            if(key == keyboard.Key.up):
                self.board_window[3] = np.clip(self.board_window[3]+1, self.min_height, self.max_height)
            if(key == keyboard.Key.down):
                self.board_window[3] = np.clip(self.board_window[3]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.left):
                self.board_window[2] = np.clip(self.board_window[2]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.right):
                self.board_window[2] = np.clip(self.board_window[2]+1, self.min_height, self.max_height)

        self.draw_windows()
        print(self.board_window)




if __name__ == '__main__':
    ss = ScreenScraper()
    s = input('')
