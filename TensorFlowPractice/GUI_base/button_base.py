import pygame
import enum
from tool.color import *


class Status(enum.Enum):
    UP = 1
    DOWN = 2


class ButtonBase():

    def __init__(self, b_surface: pygame.Surface, rect=pygame.Rect((0, 0), (128, 128)), down_color=gray,
                 up_color=silver,
                 status=Status.UP):
        self.surface = b_surface
        self.rect = rect
        self.down_color = down_color
        self.up_color = up_color
        self.color = up_color
        self.status = status
        self.even_list = []

    def function_down(self):
        print('default')

    def function_up(self):
        print('default')

    def use(self):
        click_left = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos) and click_left is 1:
            while click_left:
                pygame.event.get()
                click_left = pygame.mouse.get_pressed()[0]
            if self.status == Status.DOWN:
                self.status = Status.UP
            elif self.status == Status.UP:
                self.status = Status.DOWN

        if self.status == Status.DOWN:
            self.color = black
            self.function_down()
        else:
            self.color = self.up_color
            self.function_up()
        self.surface.fill(black, self.rect)
        rect_inner=pygame.Rect((self.rect.x+3,self.rect.y+3),(self.rect.w-6,self.rect.h-6))
        self.surface.fill(self.color, rect_inner)


