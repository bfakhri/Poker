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

        # Min Window Params
        self.min_width = 20
        self.min_height = 20
        self.max_width = 2000
        self.max_height = 2000

        # Format is [x,y,w,h]
        self.hand_window = [0,0,self.min_width,self.min_height]
        self.board_window = [0,0,self.min_width,self.min_height]
        # Gets the position of the user's cards
        print('Move mouse to top left-hand corner of hand')
        # Gets the position of the board
        # Sets up scrapers...
        return None

    def draw_windows(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": self.hand_window[1], "left": self.hand_window[0], "width": self.hand_window[2], "height": self.hand_window[3]}
            print(self.hand_window)
            # Grab the data
            sct_img = np.asarray(sct.grab(monitor))
            cv2.imshow('hand', sct_img)
            cv2.waitKey(-1)

    def on_move(self, x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))
        if(self.move_hand):
            self.hand_window[0:2] = [x,y]


    def on_click(self, x, y, button, pressed):
        print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
        if not pressed:
            print('Done...?')
            # Stop the listener
            #return False

    def key_on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(key.char))
        except AttributeError:
            print('special key {0} pressed'.format(key))
            if(key == keyboard.Key.up):
                self.hand_window[3] = np.clip(self.hand_window[3]+1, self.min_height, self.max_height)
            if(key == keyboard.Key.down):
                self.hand_window[3] = np.clip(self.hand_window[3]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.left):
                self.hand_window[2] = np.clip(self.hand_window[2]-1, self.min_height, self.max_height)
            if(key == keyboard.Key.right):
                self.hand_window[2] = np.clip(self.hand_window[2]+1, self.min_height, self.max_height)

        self.draw_windows()




if __name__ == '__main__':
    ss = ScreenScraper()
    s = input('')
