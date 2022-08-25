

import numpy as np
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show


# the function that I'm going to plot
def f_y(alpha,y):
 return (1/a*(1-a))*(a+(1-a)*y-np.power(y,1-a))
 
alpha = arange(0,30,1)
y = arange(1,2,0.1)
X,Y = meshgrid(alpha, y) # grid of point
Z = f_y(X, Y) # evaluation of the function on the grid

im = imshow(Z,cmap=cm.RdBu) # drawing the function
# adding the Contour lines with labels
cset = contour(Z,arange(-3,2,0.1),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$D$')
show()