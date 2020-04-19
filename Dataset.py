# Defines the dataset
import cv2
import numpy as np

class CardDataset:
    def __init__(self):
        img_dir = './images/'
        self.suits = ['s','h','d','c']
        self.suit_colors = {'s': (1,1,1),'h': (1,1,255),'d': (255,1,1),'c': (2,130,2)}
        self.ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        self.fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]

        self.img_blank_card = cv2.imread(img_dir+'blank_card.png')
        self.imgs_suit = {}
        for suit in self.suits:
            self.imgs_suit[suit] = cv2.imread(img_dir+'suit_'+suit+'.png')

    def make_card(self, suit, rank, randomize=True):
        # Get the base images
        img_card = self.img_blank_card.copy()
        img_suit = self.imgs_suit[suit].copy()

        # Params
        if(randomize):
            suit_x_offs = np.random.randint(1,4) 
            suit_y_offs = np.random.randint(1,4) 
            rank_x_offs = np.random.randint(4,9) 
            rank_y_offs = np.random.randint(5,8) 
            font = self.fonts[np.random.randint(0,len(self.fonts)-1)]
            font_scale = np.random.rand(1)*0.2 + 0.55 
            thickness = np.random.rand(1)*0.2 + 0.9
            suit_scale = np.random.rand(2)*0.5 + 0.9
            img_suit = cv2.resize(img_suit, (0,0), fx=suit_scale[0], fy=suit_scale[1])
        else:
            suit_x_offs = 3 
            suit_y_offs = 3 
            rank_x_offs = 5 
            rank_y_offs = 6
            font = self.fonts[0]
            font_scale = 0.65
            thickness = 1
        rank_x_offs *= len(rank)
        color = self.suit_colors[suit]

        # Place suit on blank card
        img_card[suit_y_offs:suit_y_offs+img_suit.shape[0], suit_x_offs:suit_x_offs+img_suit.shape[1], :] = img_suit
        img_card[img_card.shape[0]-img_suit.shape[0]-suit_y_offs:img_card.shape[0]-suit_y_offs, img_card.shape[1]-img_suit.shape[1]-suit_x_offs:img_card.shape[1]-suit_x_offs, :] = img_suit
        # Place rank on card
        rank_coords = (int(img_card.shape[1]/2-rank_x_offs), int(img_card.shape[0]/2+rank_y_offs))
        print(rank, rank_coords, font, font_scale, color, thickness, cv2.LINE_AA)
        img_card = cv2.putText(img_card, rank, rank_coords, font, font_scale, color, thickness, cv2.LINE_AA)


        # Place rank on blank card

        return img_card

    def make_all_cards(self):
        cards = []
        for suit in self.suits:
            for rank in self.ranks:
                print('Making: ', rank+suit)
                card = self.make_card(suit, rank)
                cards.append(card)
                cv2.imshow('Card', card)
                cv2.waitKey(-1)

    #kdef show_cards(cards):
    #k    for idx,card in enumerate(cards):
    #k        cv2.imshow('
    #k    

if(__name__ == '__main__'):
    ds = CardDataset()
    ds.make_all_cards()
