from matplotlib import pyplot as plt
import numpy as np

class Letters():
    def __init__(self, hsize, vsize):
        """
        Creates the frame where a single letter will be drawn
        """
        self.frame = np.zeros(shape=(hsize*vsize), dtype=float)
        self.hsize = hsize
        self.vsize = vsize

    def _check_pixel(self, pixel):
        if pixel < 0 or pixel > self.hsize*self.vsize:
            raise ValueError('Pixel {} does not exist.'.format(pixel))

    def _draw(self, begin, number, direction):
        """
        Converts the specified number of pixels into white along the specified direction.
        Directions: 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'
        Returns the number of the pixel where it stopped drawing.
        """
        wind_rose = {'n': -self.hsize, 'ne': -self.hsize+1, 'e': 1,
                     'se': self.hsize+1, 's': self.hsize, 'sw': self.hsize-1,
                     'w': -1, 'nw': -self.hsize-1}
        if direction not in wind_rose.keys():
            raise ValueError('Please insert a valid direction.')
        pixel = begin
        for i in range(number):
            self._check_pixel(pixel)
            self.frame[pixel] = 1.
            if i!=number-1: pixel = pixel + wind_rose[direction]
        return pixel

    def reset(self):
        self.frame[:] = 0.

    def get_frame(self):
        return np.reshape(self.frame, (self.hsize, self.vsize))

    def save_as_fig(self, name):
        self.fig = plt.figure(figsize=(6,6))   
        plt.axis('off')
        plt.imshow(self.frame.reshape(self.hsize,self.vsize), cmap='Greys_r')
        plt.savefig('figs/letter_{}.png'.format(name))
        plt.close()

    def draw_letter(self, letter):
        letters = {'a', 'e', 'm', 'n'}
        if letter not in letters:
            raise ValuerError('The specified letter cannot be drawn.')

        if letter == 'a':
            pix_init = int(3*self.hsize / 2)
            pixel = pix_init - 1
            i=0
            while (pixel-1)%self.hsize != 0: #left diagonal
                pixel = self._draw(pixel, number=2, direction='sw')
                pixel = self._draw(pixel, number=2, direction='s')
            pixel = pix_init
            while (pixel+2)%self.hsize != 0: #right diagonal
                pixel = self._draw(pixel, number=2, direction='se')
                pixel = self._draw(pixel, number=2, direction='s')
            self._draw(begin=self.hsize*10 + 8, number=11, direction='e')
            self._draw(begin=self.hsize*11 + 8, number=11, direction='e')

        elif letter == 'e':
            self._draw(begin=3 + 2*self.hsize, number=22, direction='e') #first horizontal bar
            self._draw(begin=3 + 11*self.hsize, number=18, direction='e') #second horizontal bar
            self._draw(begin=3 + 24*self.hsize, number=22, direction='e') #third horizontal bar
            self._draw(begin=3 + 2*self.hsize, number=23, direction='s') #vertical bar

        elif letter == 'm':
            pixel = self._draw(begin=self.vsize*self.hsize - 3*self.hsize + 5, number=22, direction='n')
            pixel = self._draw(begin=pixel, number=9, direction='se') 
            pixel = self._draw(begin=pixel, number=9, direction='ne') 
            pixel = self._draw(begin=pixel, number=22, direction='s') 

        elif letter == 'n':
            pixel = self._draw(begin=self.vsize*self.hsize - 3*self.hsize + 3, number=22, direction='n')
            pixel = self._draw(begin=pixel, number=22, direction='se') 
            pixel = self._draw(begin=pixel, number=22, direction='n') 
