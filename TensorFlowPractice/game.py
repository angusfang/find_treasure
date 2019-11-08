import pygame
import global_val as Gvar
from player import Player
from treasure import Treasure
from enemy import Enemy
from enemy import enemy_creator
from treasure import treasure_creator
import numpy as np
from GUI_base.text_box_base import TextBoxBase


class Game:

    # get treasure

    def init(self):
        pygame.init()
        self.surface = pygame.display.set_mode((1200, 700))
        Gvar.surface = self.surface
        self.surfRect = self.surface.get_rect()
        self.player1 = Player()
        self.treasurelist = treasure_creator(1, is_rand=False)
        self.player1.set(10, 5, 900, 600)
        # self.enemylist = enemy_creator(2)
        self.command = ''
        self.text_box = TextBoxBase(Gvar.surface)
        self.add_draw_rect_color_list = []
        Gvar.surface.fill([255, 255, 255])

    def render(self):
        Gvar.event_list = pygame.event.get()

        for EV in Gvar.event_list:
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_ESCAPE:
                    # quit()
                    self.command = 'exit'
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_o:
                    # quit()
                    self.command = 'set_epsilon 1.1'
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_p:
                    # quit()
                    self.command = 'set_epsilon 0.1'
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_i:
                    # quit()
                    self.command = 'manual'
            if EV.type == pygame.KEYDOWN:
                if EV.key == pygame.K_SPACE:
                    # quit()
                    self.command = 'setting'
        Gvar.command=self.command

        # Gvar.surface.fill([255, 255, 255])
        self.text_box.use()
        # self.player1.control()
        xn, yn = self.boundary_limit(self.player1.x, self.player1.y, self.player1.speed)
        self.player1.set_xy(xn, yn)
        for rect_color_I in self.add_draw_rect_color_list:
            pygame.draw.rect(Gvar.surface, rect_color_I[1], rect_color_I[0])
        self.add_draw_rect_color_list.clear()
        self.player1.show()

        for treasureI in self.treasurelist:
            treasureI.show()

        # for enemyI in self.enemylist:
        #     enemyI.track(self.player1.x, self.player1.y)
        #     enemyI.show()

        # if pygame.time.get_ticks()//1%100==0:
        #     enemyI=enemy_creator(1)[0]
        #     enemylist.append(enemyI)



        pygame.display.flip()

    def get_infomation(self):
        player = [self.player1.x/1200, self.player1.y/700]
        # treasure =[self.treasure1.x,self.treasure1.y]
        treasure = []
        for treasureI in self.treasurelist:
            treasure.append([treasureI.x, treasureI.y])
        # enemy = []
        # for enemyI in self.enemylist:
        #     enemy.append([enemyI.x, enemyI.y])
        return np.array(player)

    def reward_judgment(self):

        # enemy_rect_list = []
        # for enemyI in self.enemylist:
        #     enemy_rect_list.append(enemyI.rect)
        # reward = len(self.player1.rect.collidelistall(enemy_rect_list))
        treasure_rect_list = []
        distance=0
        done =False
        for treasureI in self.treasurelist:
            distanceI=(self.player1.x-treasureI.x)*(self.player1.x-treasureI.x)+(self.player1.y-treasureI.y)*(self.player1.y-treasureI.y)
            distanceI=np.sqrt(distanceI)
            distance=distance+distanceI
        reward = -distance/100

        if reward == 1:
            done = True
            reward=1
        return reward , done

    def boundary_limit(self, x, y,s):
        xn = x
        yn = y
        collection=False
        if x < 0:
            xn = 0+50
        if x > self.surfRect.w:
            xn = self.surfRect.w-50
        if y < 0:
            yn = 0+50
        if y > self.surfRect.h:
            yn = self.surfRect.h-50
        return xn, yn


if __name__ == '__main__':

    game1 = Game()
    game1.init()
    i = 1
    while 1:
        i = i + 1
        # command=int(input())
        import math

        game1.player1.set_xy(500 + math.sin(i) * 200, 500 + math.cos(i) * 200)
        game1.render()
        print(game1.player1.x)
