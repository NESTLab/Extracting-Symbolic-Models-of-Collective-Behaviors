from random import random
from math import pi, cos, sin, atan2, sqrt, radians, degrees
import pygame
import sys
from utils.models import *
import torch
from torch_geometric.data import Data

################################
## SIMULATION PARAMETERS
################################
MAXSPEED = 3
MINSPEED = 0
MAXFORCE = 0.5
HEIGHT = 400
WIDTH = 400
N_BOIDS = 30

################################
## VISUALIZATION PARAMETERS
################################
FPS = 30
VIS_SCALE = 10

################################
## EXPERIMENTATION PARAMETERS
################################
SECONDS = 15

LOG = False

class Flock:
    def __init__(self, n, screen, VISUALIZE=False):
        self.flock = [BOID(i) for i in range(n)]
        self.screen = screen
        self.VISUALIZE = VISUALIZE
    def run(self, st):
        output = ""
        count = 0
        if LOG:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        for b in self.flock:
            b.run(self.flock)
            output += str(st) + "," + str(count) + "," + \
                      str(b.pos[0]) + "," + str(b.pos[1]) + "," + \
                      str(b.vel[0]) + "," + str(b.vel[1]) + "," + \
                      str(b.rawForce[0]) + "," + str(b.rawForce[1]) + "\n"
            count += 1
        if self.VISUALIZE:
            self.draw()
        if LOG:
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        return output
    def draw(self):
        if self.VISUALIZE:
            self.screen.fill((255, 255, 255))
            for b in self.flock:
                b.draw(self.screen)
class BOID:
    def __init__(self, i):
        self.i = i
        self.pos = [random()*WIDTH, random()*HEIGHT]
        angle = random() * 2 * pi
        speed = random()*MAXSPEED
        self.vel = [cos(angle), sin(angle)]
        self.acc = [0, 0]
        self.rawForce = [0, 0]

    def run(self, boids):
        self.flock(boids)
        self.update()
        self.borders()
    def limitVel(self):
        mag = sqrt(pow(self.vel[0], 2) + pow(self.vel[1], 2))
        ang = atan2(self.vel[1], self.vel[0])
        if mag > MAXSPEED:
            mag = MAXSPEED
        if mag < MINSPEED:
            mag = MINSPEED
        self.vel[0] = mag*cos(ang)
        self.vel[1] = mag*sin(ang)
    def flock(self, boids):
        global LOG
        if LOG:
            print("Boid " + str(self.i) + ": ")
        force = [0, 0]
        visRange = 150
        desiredSep = 50
        count = 0
        gnnData = [[1.]]
        edgeData = [[], []]
        att = []
        for b in boids:
            if b is not self:
                dis = sqrt(pow(self.pos[0]-b.pos[0], 2) + pow(self.pos[1]-b.pos[1],2))
                if 0 < dis <= visRange:

                    count += 1

                    x1 = self.pos[0]
                    x2 = b.pos[0]
                    y1 = self.pos[1]
                    y2 = b.pos[1]
                    xd1 = self.vel[0]
                    xd2 = b.vel[0]
                    yd1 = self.vel[1]
                    yd2 = b.vel[1]
                    s1 = sqrt(xd1 * xd1 + yd1 * yd1)
                    dS = desiredSep

                    # Conversion to local frame
                    lFX 	= (x2-x1)*(xd1/s1)-(y2-y1)*(yd1/s1)
                    lFY 	= (x2-x1)*(yd1/s1)+(y2-y1)*(xd1/s1)
                    lFXD 	= xd2-xd1
                    lFYD 	= yd2-yd1
                    d 		= sqrt(pow(lFX, 2) + pow(lFY, 2))
                    s 		= sqrt(pow(lFXD, 2) + pow(lFYD, 2))
                    invd = 1.0 / d
                    invs = 0
                    if s != 0:
                        invs = 1.0 / s
                    lFnX = lFX * invd
                    lFnY = lFY * invd
                    lFnXD = lFXD * invs
                    lFnYD = lFYD * invs

                    # NORMALIZE
                    lFX /= visRange
                    lFY /= visRange
                    d /= visRange
                    dS /= visRange
                    lFXD /= (2*MAXSPEED)
                    lFYD /= (2*MAXSPEED)

                    S =  1.5
                    A =  3.0
                    C =  2.0

                    coh = [0, 0]
                    sep = [0, 0]
                    ali = [0, 0]

                    coh[0] = lFX/d	*(float(C))
                    sep[0] = lFX/d	*(float(S))	 		*(dS/d)
                    ali[0] = lFXD	*(float(A))

                    coh[1] = lFY/d 	*(float(C))
                    sep[1] = lFX/d	*(float(S))	 		*(dS/d)
                    ali[1] = lFYD 	*(float(A))

                    LOG = False
                    if LOG:
                        print("-------------------")
                        csMag = sqrt(pow(coh[0]-sep[0], 2) + pow(coh[1]-sep[1], 2))
                        aMag = sqrt(pow(ali[0], 2) + pow(ali[1], 2))
                        print(csMag)
                        print(aMag)
                        print("-------------------")
                    LOG = False

                    force[0] += coh[0]-sep[0]+ali[0]
                    force[1] += coh[1]-sep[1]+ali[1]

        if count > 0:
            force[0] /= count
            force[1] /= count
            self.rawForce[0] = force[0]
            self.rawForce[1] = force[1]
            LOG = False
            if LOG:
                print("==================")
                print(self.rawForce)
                print("==================")
            LOG = False
        self.acc[0] += force[0]
        self.acc[1] += force[1]
    def update(self):

        myang = -atan2(self.vel[1], self.vel[0])

        newX = (self.acc[0]*cos(myang)) - (self.acc[1]*sin(myang))
        newY = (self.acc[0]*sin(myang)) + (self.acc[1]*cos(myang))
        self.vel[0] += newX
        self.vel[1] += newY

        self.limitVel()

        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        self.acc = [0, 0]
    def borders(self):
        if self.pos[0] < -2: self.pos[0] = WIDTH + 2
        if self.pos[1] < -2: self.pos[1] = HEIGHT + 2
        if self.pos[0] > WIDTH + 2: self.pos[0] = -2
        if self.pos[1] > HEIGHT + 2: self.pos[1] = -2
    def draw(self, scr):
        scale = VIS_SCALE
        x = self.pos[0]
        y = self.pos[1]
        xa = self.vel[0]
        ya = self.vel[1]
        t = atan2(ya, xa)/pi * 180
        pygame.draw.polygon(scr, (0, 0, 0), [self.rotatePoint((x, y), (x+1*scale, y+0), t),
                                            self.rotatePoint((x, y), (x-.5*scale, y+.4*scale), t),
                                            self.rotatePoint((x, y), (x-.2*scale, y+0), t),
                                            self.rotatePoint((x, y), (x-.5*scale, y-.4*scale), t)], 1)
        mag = sqrt(pow(xa, 2) + pow(ya, 2))
        pygame.draw.line(scr, (0, 0, 0), (x, y), self.rotatePoint((x, y), (x+mag/2, y), t))
    def rotatePoint(self, centerPoint,point,angle):
        angle = radians(angle)
        temp_point = point[0]-centerPoint[0] , point[1]-centerPoint[1]
        temp_point = ( temp_point[0]*cos(angle)-temp_point[1]*sin(angle) , temp_point[0]*sin(angle)+temp_point[1]*cos(angle))
        temp_point = temp_point[0]+centerPoint[0] , temp_point[1]+centerPoint[1]
        return temp_point