# Defines the dataset
import cv2

class CardDataset:
    def __init__(self):
        img_dir = './images/'
        self.suits = ['s','h','d','c']
        self.ranks = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
        self.fonts = ['FONT_HERSHEY_SIMPLEX', 'FONT_HERSHEY_PLAIN', 'FONT_HERSHEY_DUPLEX', 'FONT_HERSHEY_COMPLEX', 'FONT_HERSHEY_TRIPLEX', 'FONT_HERSHEY_COMPLEX_SMALL', 'FONT_HERSHEY_SCRIPT_SIMPLEX', 'FONT_HERSHEY_SCRIPT_COMPLEX']

        self.img_blank_card = cv2.imread(img_dir+'blank_card.png')
        self.imgs_suit = {}
        for suit in self.suits:
            self.imgs_suit[suit] = cv2.imread(img_dir+'suit_'+suit+'.png')

    def make_card(self, suit, rank):
        # Params
        x_offs = 3 
        y_offs = 3 
        font = self.fonts[0]
        font_scale = 1

        # Get the base images
        img_card = self.img_blank_card.copy()
        img_suit = self.imgs_suit[suit].copy()
        # Place suit on blank card
        img_card[y_offs:y_offs+img_suit.shape[0], x_offs:x_offs+img_suit.shape[1], :] = img_suit
        img_card[img_card.shape[0]-img_suit.shape[0]-y_offs:img_card.shape[0]-y_offs, img_card.shape[1]-img_suit.shape[1]-x_offs:img_card.shape[1]-x_offs, :] = img_suit
        ## Place rank on card
        #rank_coords = (img_card[0]/2, img_card[1]/2)
        #cv2.putText(img_card, rank, rank_coords, font, font_scale)


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
                cv2.waitKey(10)

    #kdef show_cards(cards):
    #k    for idx,card in enumerate(cards):
    #k        cv2.imshow('
    #k    

if(__name__ == '__main__'):
    ds = CardDataset()
    ds.make_all_cards()
