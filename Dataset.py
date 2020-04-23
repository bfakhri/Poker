# Defines the dataset
import cv2
import numpy as np
from treys import Card
from treys import Deck

class CardDataset:
    def __init__(self):
        img_dir = './images/'
        self.suits = ['s','h','d','c']
        self.suit_colors = {'s': (1,1,1),'h': (1,1,255),'d': (255,1,1),'c': (2,130,2)}
        self.ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        self.ranks_trey = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
        self.fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]

        self.img_blank_card = cv2.imread(img_dir+'blank_card.png')
        self.imgs_suit = {}
        for suit in self.suits:
            self.imgs_suit[suit] = cv2.imread(img_dir+'suit_'+suit+'.png')

        self.backgrounds = ['felt_base.png', 'felt_base_cr.png', 'hard_base.png']
        self.img_backgrounds = []
        for bg in self.backgrounds:
            self.img_backgrounds.append(cv2.imread(img_dir+bg))

    def flatidx_to_card(self, flat_idxs):
        ''' 
        Converts a flat idx (like what is given by the model) into a 'treys' card
        Inputs: (idx) -> [cards]
        '''
        suit_idxs, rank_idxs = np.unravel_index(flat_idxs, (len(self.suits), len(self.ranks)))
        cards = []
        for i in range(len(flat_idxs)):
            rank_str = self.ranks_trey[rank_idxs[i]]
            suit_str = self.suits[suit_idxs[i]]
            cards.append(Card.new(rank_str+suit_str))

        return cards


    def make_card(self, suit, rank, randomize=True, font=None, thickness=None):
        # Get the base images
        img_card = self.img_blank_card.copy()
        img_suit = self.imgs_suit[suit].copy()

        # Params
        if(randomize):
            suit_x_offs = np.random.randint(1,4) 
            suit_y_offs = np.random.randint(1,4) 
            rank_x_offs = np.random.randint(4,9) 
            rank_y_offs = np.random.randint(5,8) 
            if(font is None):
                font = self.fonts[np.random.randint(0,len(self.fonts))]
            else: 
                font = self.fonts[font]
            font_scale = np.random.rand(1)*0.2 + 0.55 
            suit_scale = np.random.rand(2)*0.5 + 0.75
            img_suit = cv2.resize(img_suit, (0,0), fx=suit_scale[0], fy=suit_scale[1])
        else:
            suit_x_offs = 3 
            suit_y_offs = 3 
            rank_x_offs = 5 
            rank_y_offs = 6
            if(font is None):
                font = self.fonts[0]
            else: 
                font = self.fonts[font]
            font_scale = [0.65]
        rank_x_offs *= len(rank)
        color = self.suit_colors[suit]

        if(thickness is None):
            thickness = 1

        # Place suit on blank card
        img_card[suit_y_offs:suit_y_offs+img_suit.shape[0], suit_x_offs:suit_x_offs+img_suit.shape[1], :] = img_suit
        img_card[img_card.shape[0]-img_suit.shape[0]-suit_y_offs:img_card.shape[0]-suit_y_offs, img_card.shape[1]-img_suit.shape[1]-suit_x_offs:img_card.shape[1]-suit_x_offs, :] = img_suit
        # Place rank on card
        rank_coords = (int(img_card.shape[1]/2-rank_x_offs), int(img_card.shape[0]/2+rank_y_offs))
        img_card = cv2.putText(img_card, rank, rank_coords, font, font_scale[0], color, thickness, cv2.LINE_AA)

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

    def make_all_card_fonts(self):
        cards = []
        for suit in self.suits:
            for rank in self.ranks:
                for font in self.fonts:
                    print('Making: ', rank+suit+str(font))
                    card = self.make_card(suit, rank, font=font)
                    cards.append(card)
                    cv2.imshow('Card', card)
                    cv2.waitKey(-1)

    def make_all_card_thicknesses(self):
        cards = []
        for i in range(20):
            thick = i
            print('Making: ', thick)
            card = self.make_card('s', '2', randomize=False, font=0, thickness=thick)
            cards.append(card)
            cv2.imshow('Card', card)
            cv2.waitKey(-1)

    def make_collection(self, background=None, min_cards=0, max_cards=5, font=None, cutoff=False):
        ''' Makes a collection of cards and the labels '''
        # Spacing between cards
        x_spacing = np.random.randint(1,20) 
        # Get base image
        if(background is None):
            img_bg = self.img_backgrounds[np.random.randint(0,len(self.img_backgrounds))].copy()
        else:
            img_bg = background.copy()

        # Place cards on base image
        num_cards = np.random.randint(min_cards, max_cards) 
        x = x_spacing
        y = np.random.randint(10,30) 
        label = np.zeros((len(self.suits), len(self.ranks)))
        for c in range(num_cards):
            card_id = np.random.randint(0, label.size-1)
            while(label.flat[card_id] == 1):
                card_id = np.random.randint(0, label.size-1)
            label.flat[card_id] = 1
            suit_idx, rank_idx = np.unravel_index(card_id, (4,13))
            card = self.make_card(self.suits[suit_idx], self.ranks[rank_idx], randomize=True, font=font)
            img_bg[y:y+card.shape[0], x:x+card.shape[1], :] = card
            x += card.shape[1] + x_spacing

        if(cutoff):
            #img_bg = img_bg[0:int(card.shape[0]), x:x+card.shape[1], :] = card
            img_bg = img_bg[0:y+int(0.7*card.shape[0]), :, :] 

        # Reshape labels to flat
        label = np.reshape(label, (-1))

        return img_bg, label

    def show_collections(self, N=100000):
        for i in range(N):
            collection, label = self.make_collection()
            print(collection.shape, label)
            cv2.imshow('Collection', collection)
            cv2.waitKey(-1)
            

    def batch_collection(self, bs, font=None):
        ''' Generates a batch of card collections and labels '''
        collections = []
        labels = []
        bg = self.img_backgrounds[np.random.randint(0,len(self.img_backgrounds))].copy()
        for i in range(bs):
            coll, lab = self.make_collection(background=bg, font=font, cutoff=(i%2==0))
            collections.append(coll)
            labels.append(lab)

        batch_collections = np.stack(collections, axis=0)
        batch_labels = np.stack(labels, axis=0)

        return batch_collections, batch_labels


        

if(__name__ == '__main__'):
    ds = CardDataset()
    #ds.make_all_cards()
    #ds.show_collections()
    #ds.make_all_card_fonts()
    ds.make_all_card_thicknesses()
