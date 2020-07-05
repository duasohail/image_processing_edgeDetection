import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import skimage.io as io

from copy import deepcopy


#show image in rgb channels


pic1 = io.imread('img1.png')

red_ch= deepcopy(pic1)
green_ch =deepcopy(pic1)
blue_ch=deepcopy(pic1)

red_ch [: , : , 1]=0
red_ch [: , : , 2]=0

green_ch [: , : , 0]=0
green_ch [: , : , 2]=0

blue_ch [: , : , 0]=0
blue_ch [: , : , 1]=0

fig , ax =plt.subplots( ncols=2 , nrows=2)

ax[0,0].imshow(pic1)
ax[0,0].set_title('original')

ax[0,1].imshow(red_ch)
ax[0,1].set_title('show red channel')

ax[1,0].imshow(green_ch)
ax[1,0].set_title('show green channel')

ax[1,1].imshow(blue_ch)
ax[1,1].set_title('show blue channel')

plt.show()

#here start code of robert edge detection

import skimage.filters as sk_filters
image= io.imread('img1.png' , as_gray=True)
res =sk_filters.roberts(image)
plt.imshow(res , cmap='Blues')
plt.title('roberts edge detection')
plt.show()

#here start code of sobel edge detection
res1 =sk_filters.sobel(image)
plt.imshow(res1 , cmap='hot')
plt.title('sobels edge detection')
plt.show()

#here start code of scharr edge detection
res2 =sk_filters.scharr(image)
plt.imshow(res2 , cmap='Wistia')
plt.title('scharrs edge detection')
plt.show()


#here start code of pewitt edge detection
res3 =sk_filters.prewitt(image)
plt.imshow(res3 , cmap='cool')
plt.title('prewitts edge detection')
plt.show()