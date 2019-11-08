import pygame
import global_val as Gva


class Player:

    def __init__(self):
        self.speed = 0
        self.health = 0
        self.rect = pygame.Rect
        self.x = 0.0
        self.y = 0.0

        self.y_minus = False
        self.x_minus = False
        self.y_plus = False
        self.x_plus = False

    def set(self, speed, health, x, y):
        self.speed = speed
        self.health = health
        self.x = x
        self.y = y
        self.rect = pygame.Rect((0, 0, 20, 20))
        self.rect.center = (x, y)

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

    def set_xy(self,x,y):
        self.x=x
        self.y=y
        self.rect.center = (x, y)

    def show(self):
        pygame.draw.rect(Gva.surface, [0, 0, 128], self.rect)
