import pygame
import numpy as np
import global_val as Gva

class Enemy:
    def __init__(self):
        self.speed = 0
        self.health = 0
        self.rect = pygame.rect
        self.x = 0.0
        self.y = 0.0
        self.mode=1

        self.y_minus = False
        self.x_minus = False
        self.y_plus = False
        self.x_plus = False

    def set(self, speed, health, x, y,mode=1):
        self.speed = speed
        self.health = health
        self.x = x
        self.y = y
        self.rect = pygame.Rect((0, 0, 20, 20))
        self.rect.center = (x, y)
        self.mode=mode

    def control(self):
        for EV in Gva.event_list:
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_w:
                    self.y_minus = True
                if EV.key == pygame.K_a:
                    self.x_minus = True
                if EV.key == pygame.K_s:
                    self.y_plus = True
                if EV.key == pygame.K_d:
                    self.x_plus = True

            if EV.type == pygame.KEYUP:
                if EV.key == pygame.K_w:
                    self.y_minus = False
                if EV.key == pygame.K_a:
                    self.x_minus = False
                if EV.key == pygame.K_s:
                    self.y_plus = False
                if EV.key == pygame.K_d:
                    self.x_plus = False

        if self.y_minus is True:
            self.y = self.y - 1 * self.speed
        if self.x_minus is True:
            self.x = self.x - 1 * self.speed
        if self.y_plus is True:
            self.y = self.y + 1 * self.speed
        if self.x_plus is True:
            self.x = self.x + 1 * self.speed

        self.rect.center = (self.x, self.y)

    def track(self,x,y):
        if self.mode is 0:
            dir_x=np.sign(x-self.x)
            dir_y=np.sign(y-self.y)
            self.x=self.x+dir_x*self.speed
            self.y=self.y+dir_y*self.speed

            self.rect.center =(self.x,self.y)
        if self.mode is 1:
            dir_x = np.sign(x - self.x)
            dir_y = np.sign(y - self.y)
            choice=np.random.randint(0,2)
            if choice is 0:
                self.x = self.x + dir_x * self.speed
            if choice is 1:
                self.y = self.y + dir_y * self.speed
            self.rect.center = (self.x, self.y)
        if self.mode is 2:
            dir_x = np.sign(x - self.x)
            dir_y = np.sign(y - self.y)
            choice = np.random.randint(0, 2)
            if choice is 0:
                self.x = self.x + dir_x * self.speed
            if choice is 1:
                self.y = self.y + dir_y * self.speed
            self.rect.center = (self.x, self.y)
        if self.mode is 3:
            dir_x = np.sign(x - self.x)
            dir_y = np.sign(y - self.y)
            choice=np.random.randint(0,10)
            if choice is 0:
                self.x = self.x + dir_x * self.speed
            else:
                self.y = self.y + dir_y * self.speed
            self.rect.center = (self.x, self.y)
        if self.mode is 4:
            dir_x = np.sign(x - self.x)
            dir_y = np.sign(y - self.y)
            choice=np.random.randint(0,10)
            if choice is 0:
                self.y = self.y + dir_y * self.speed
            else:
                self.x = self.x + dir_x * self.speed
            self.rect.center = (self.x, self.y)
        if self.mode is 5:
            choice=np.random.randint(0,3)
            dir_x = np.sign(x - self.x)
            dir_y = np.sign(y - self.y)
            if choice is not 0:
                self.x = self.x + dir_x * self.speed
                self.y = self.y + dir_y * self.speed
            else:
                self.x = self.x - dir_x * self.speed
                self.y = self.y - dir_y * self.speed
            self.rect.center = (self.x, self.y)
    def show(self):
        pygame.draw.rect(Gva.surface, [255, 0, 0], self.rect)

def enemy_creator(number):
    enemy_list=[]
    assert isinstance(Gva.surface, pygame.Surface)
    h=Gva.surface.get_height()
    w=Gva.surface.get_width()
    x = np.random.random((number))*w
    y = np.random.random((number))*h
    s = 20
    for i in range(number):
        enemyI=Enemy()
        m = np.random.randint(0, 6)
        enemyI.set(s,20,x[i],y[i],0)
        enemy_list.append(enemyI)
    return enemy_list