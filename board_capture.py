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

with mss.mss() as sct:
    # The screen part to capture
    monitor = {"top": 160, "left": 160, "width": 160, "height": 135}
    output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

    # Grab the data
    sct_img = sct.grab(monitor)

    # Save to the picture file
    mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
    print(output)






class ScreenScraper:
    def __init__(self):
        # Setup input listeners:
        self.key_listener = keyboard.Listener(on_press=self.key_on_press)
        self.key_listener.start()
        self.listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move)
        self.listener.start()

        # Format is [x,y,w,h]
        self.hand_window = [0,0,0,0]
        self.board_window = [0,0,0,0]
        # Gets the position of the user's cards
        print('Move mouse to top left-hand corner of hand')
        # Gets the position of the board
        # Sets up scrapers...
        return None

    def draw_windows(self):
        with mss.mss() as sct:
            # The screen part to capture
            monitor = {"top": self.hand_window[0], "left": self.hand_window[1], "width": 160, "height": 135}
            print(self.hand_window)
            # Grab the data
            sct_img = np.asarray(sct.grab(monitor))
            cv2.imshow('hand', sct_img)
            cv2.waitKey(-1)

    def on_move(self, x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))

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
            print(key)
            if(key == 'Key.up'):
                self.hand_window[3] += 1
            if(key == 'Key.down'):
                self.hand_window[3] -= 1
            if(key == 'Key.left'):
                self.hand_window[2] -= 1
            if(key == 'Key.right'):
                self.hand_window[2] += 1

        self.draw_windows()




if __name__ == '__main__':
    ss = ScreenScraper()
    s = input('')
