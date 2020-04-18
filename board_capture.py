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

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    if not pressed:
        # Stop listener
        return False

# Collect events until released
listener = mouse.Listener(on_click=on_click)
listener.start()

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
        # Gets the position of the user's cards
        print('Move mouse to top left-hand corner of hand')
        # Gets the position of the board
        # Sets up scrapers...
        return None


if __name__ == '__main__':
    ss = ScreenScraper()
    input('')
