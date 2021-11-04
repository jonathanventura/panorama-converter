from OpenGL.GL import *
from OpenGL.GLUT import *
import time


import numpy as np
import os

from rendertools import *

import cv2
from cv2 import imwrite

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--height',type=int,default=1024)
    parser.add_argument('--width',type=int,default=2048)
    args = parser.parse_args()
    
    glutInit()
    glutInitDisplayMode(GLUT_RGBA|GLUT_3_2_CORE_PROFILE|GLUT_DOUBLE)
    glutInitWindowSize(args.width,args.height)
    glutInitWindowPosition(0,0)
    window = glutCreateWindow('window')
    glutHideWindow(window)
    
    mesh = Cylinder(bottom=-2,top=2,radius=1)
    
    renderer = Renderer([mesh],width=1,height=args.height,cubemappath=args.input)
    
    thetas = np.linspace(-np.pi,np.pi,args.width,endpoint=False)
    
    fovy = 2.*np.arctan(2)*180./np.pi
    eye = np.array([0,0,0])
    up = np.array([0,1,0])

    proj_matrix = perspective(fovy, 1./args.height, 0.1, 1000.0)
    
    image = np.zeros((args.height,args.width,3),dtype='uint8')
    for i,theta in enumerate(thetas):
        target = np.array([np.sin(theta),0,-np.cos(theta)])
        view_matrix = lookAt(eye,target,up)
        mvp_matrix = proj_matrix@view_matrix
        column = renderer.render(mvp_matrix)
        image[:,i:i+1] = column

    image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    imwrite(args.output,image_bgr)

