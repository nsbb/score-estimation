import math
import numpy as np
import cv2 as cv
from math import sqrt
​
def mean_intensity(img):
    sum_intensity = 0
    sum_intensity = [pixel for row in img for pixel in row]
    mean_intensity = int(sum(sum_intensity) / img.size)
    return mean_intensity
    
def var_abs(img_name):
    img = cv.imread(img_name,0)
    mean = mean_intensity(img)
    var_intensity = 0
    var_intensity = [abs(pixel-mean) for row in img for pixel in row]
    var_intensity = int(sum(var_intensity) / img.size)
    return var_intensity
​
def var_sqr(img_name):
    img = cv.imread(img_name,0)
    mean = mean_intensity(img)
    var_intensity = 0
    var_intensity = [(pixel-mean)**2 for row in img for pixel in row]
    var_intensity = int(sum(var_intensity) / img.size)
    return sqrt(var_intensity)
​
def grd_abs(img_name):
    img = cv.imread(img_name,0)
    grd_intensity = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]-1):
            i1, i2 = int(img[i][j]), int(img[i][j+1])
            diff = abs(i2-i1)
            grd_intensity += diff
    return grd_intensity
​
def grd_sqr(img_name):
    img = cv.imread(img_name,0)
    grd_intensity = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]-1):
            i1, i2 = int(img[i][j]), int(img[i][j+1])
            diff = abs(i2-i1)
            grd_intensity += (diff)**2
    return sqrt(grd_intensity)
​
def correlation(img_name):
    img = cv.imread(img_name,0)
    cor1 = 0
    cor2 = 0
    for i in range(img.shape[0]-1):
        for j in range(img.shape[1]):
            i1, i2 = int(img[i][j]), int(img[i+1][j])
            cor1 += i1*i2
    for i in range(img.shape[0]-2):
        for j in range(img.shape[1]):
            i1, i2 = int(img[i][j]), int(img[i+2][j])
            cor2 += i1*i1
    return cor1-cor2
​
def entropy(img_name):
    img = cv.imread(img_name,0)
    ent_intensity = [int(pixel*math.log(pixel,2)) for row in img for pixel in row if int(pixel) is not 0]
    ent_intensity = -sum(ent_intensity)
    return ent_intensity 
​
def gcf(img_name):
    img = cv.imread(img_name,0)
    r = 2.2
    C = 0
    w = 0
    GCF = 0
    L = [[0 for _ in range(img.shape[1])] for _ in range(img.shape[0])]
    Lc = [[0 for _ in range(img.shape[1])] for _ in range(img.shape[0])]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            L[i][j] = 100*sqrt((img[i][j]/255)*r)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if i==0 and j==0: Lc[i][j]=(abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j+1]))/2
            elif i==0 and j==img.shape[1]-1: Lc[i][j]=(abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j-1]))/2
            elif i==img.shape[0]-1 and j==0: Lc[i][j]=(abs(L[i][j]-L[i-1][j])+abs(L[i][j]-L[i][j+1]))/2
            elif i==img.shape[0]-1 and j==img.shape[1]-1: Lc[i][j]=(abs(L[i][j]-L[i-1][j])+abs(L[i][j]-L[i][j-1]))/2
            elif i==0 and j!=0 and j!=img.shape[1]-1: Lc[i][j]=(abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j-1])+abs(L[i][j]-L[i][j+1])/3)
            elif j==0 and i!=0 and i!=img.shape[0]-1: Lc[i][j]=(abs(L[i][j]-L[i-1][j])+abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j+1])/3)
            elif j==img.shape[1]-1 and i!=0 and i!=img.shape[0]-1: Lc[i][j]=abs((L[i][j]-L[i-1][j])+abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j-1])/3)
            elif i==img.shape[0]-1 and j!=0 and j!=img.shape[1]-1: Lc[i][j]=abs((L[i][j]-L[i-1][j])+abs(L[i][j]-L[i][j-1])+abs(L[i][j]-L[i][j+1])/3)
            else: Lc[i][j]=(abs(L[i][j]-L[i-1][j])+abs(L[i][j]-L[i+1][j])+abs(L[i][j]-L[i][j-1])+abs(L[i][j]-L[i][j+1]))/4
            C+=Lc[i][j]
    C=C/(img.shape[0]*img.shape[1])
    for i in range(9):
        w = (-0.406385*(i/9)+0.334573)*(i/9)+0.0877526
        GCF += w*C
    return GCF