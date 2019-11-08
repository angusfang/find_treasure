from game import Game
from RL_brain_k import DeepQNetwork
import numpy as np
import math as ma
import pygame
import global_val as Gvar

if __name__ == "__main__":
    # maze game
    game = Game()
    game.init()
    Gvar.command = 'manual'
    game.command = Gvar.command
    # number of observation * 2dir
    n_feature = (1) * 2 + 1
    # number of action is 4 dir
    n_action = 4

    RL = DeepQNetwork(4, n_feature,
                      learning_rate=0.01,
                      reward_decay=1.0,
                      e_greedy=0.7,
                      replace_target_iter=3,
                      memory_size=40000,
                      # output_graph=True
                      )

    while True:
        step = 0
        game.init()
        game.command = Gvar.command
        game.render()
        # observation = game.get_infomation()
        # observation2= np.append(observation,[episode_step/500])
        for episode in range(int(1e20)):
            episode_step = 0
            # initial observation
            if game.command == 'exit':
                RL.plot_cost()
            game.init()
            game.command = Gvar.command
            game.render()
            observation = game.get_infomation()
            observation2 = np.append(observation, [episode_step/500])

            while True:
                try:

                    RL.memory_counter
                except:
                    pass
                else:
                    if RL.memory_counter > 39998:
                        # RL.output_memory()
                        pass


                game.command = Gvar.command
                episode_step = episode_step + 1
                # fresh env

                game.render()

                observation = game.get_infomation()
                observation2 = np.append(observation, [episode_step/500])
                game.text_box.text_list.append(str(observation2) + ':input:observation2')
                action, all_action_value = RL.choose_action(observation2)
                game.text_box.text_list.append(str(action) + ':output:action')
                # RL choose action based on observation
                if game.command == 'set_epsilon 1.1':
                    print('set epsilon 0.6')
                    RL.epsilon = 1.1
                    game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                    game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                    game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                    game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                    game.text_box.text_list.append(str(action) + ':action')
                    game.surface.fill([255, 255, 255])
                    game.render()
                if game.command == 'set_epsilon 0.1':
                    print('set epsilon 0.1')
                    RL.epsilon = 0.6
                    game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                    game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                    game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                    game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                    game.text_box.text_list.append(str(action) + ':action')
                    game.surface.fill([255, 255, 255])
                    game.render()

                # draw predictvalue
                size = 20
                # to see predict
                for i in range(4):
                    col = ((ma.tanh(all_action_value[0][i]) + 1) * 255 / 2)
                    # print('i:',i,',col:',col)
                    rect1 = pygame.Rect([0, i], [size // 2, size // 2])
                    if i is 0:
                        rect1.center = [game.player1.x + 1 * size // 2, game.player1.y]
                    if i is 1:
                        rect1.center = [game.player1.x - 1 * size // 2, game.player1.y]
                    if i is 2:
                        rect1.center = [game.player1.x, game.player1.y + 1 * size // 2]
                    if i is 3:
                        rect1.center = [game.player1.x, game.player1.y - 1 * size // 2]
                    game.add_draw_rect_color_list.append([rect1, [col, col, 0]])

                if game.command == 'manual':
                    print('set', game.command)
                    right_key = False
                    while right_key == False:
                        event = pygame.event.wait()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_f:
                                Gvar.command = 'free'
                                game.command = 'free'
                                right_key = True
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_d:
                                action = 0
                                right_key = True

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_a:
                                action = 1
                                right_key = True
                                # game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                                # game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                                # game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                                # game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                                # game.text_box.text_list.append(str(action) + ':action')
                                # game.surface.fill([255, 255, 255])
                                # game.render()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_s:
                                action = 2
                                right_key = True
                                # game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                                # game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                                # game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                                # game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                                # game.text_box.text_list.append(str(action) + ':action')
                                # game.surface.fill([255, 255, 255])
                                # game.render()
                        if event.type == pygame.KEYDOWN:

                            if event.key == pygame.K_w:
                                action = 3
                                right_key = True
                                # game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                                # game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                                # game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                                # game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                                # game.text_box.text_list.append(str(action) + ':action')
                                # game.surface.fill([255, 255, 255])
                                # game.render()

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_u:
                                right_key = True

                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_o:
                                Gvar.command = 'set_epsilon 1.1'
                                game.command = Gvar.command
                                right_key = True
                            if event.key == pygame.K_p:
                                Gvar.command = 'set_epsilon 0.1'
                                game.command = Gvar.command
                                right_key = True

                        game.text_box.text_list.append(str(all_action_value[0][0]) + ':action0')
                        game.text_box.text_list.append(str(all_action_value[0][1]) + ':action1')
                        game.text_box.text_list.append(str(all_action_value[0][2]) + ':action2')
                        game.text_box.text_list.append(str(all_action_value[0][3]) + ':action3')
                        game.text_box.text_list.append(str(action) + ':action')
                        game.surface.fill([255, 255, 255])
                        game.render()
                        right_key = True

                if action == 0:
                    game.player1.set_xy(game.player1.x + 1 * game.player1.speed, game.player1.y)

                if action == 1:
                    game.player1.set_xy(game.player1.x - 1 * game.player1.speed, game.player1.y)

                if action == 2:
                    game.player1.set_xy(game.player1.x, game.player1.y + 1 * game.player1.speed)

                if action == 3:
                    game.player1.set_xy(game.player1.x, game.player1.y - 1 * game.player1.speed)

                reward, done = game.reward_judgment()
                if (game.player1.x > game.surfRect.w):
                    reward = reward - 2
                    done = True
                if (game.player1.x < 0):
                    reward = reward - 2
                    done = True
                if (game.player1.y > game.surfRect.h):
                    reward = reward - 2
                    done = True
                if (game.player1.y < 0):
                    reward = reward - 2
                    done = True

                game.render()

                if episode_step > 500:
                    # reward = reward-1
                    done = True
                observation_ = Game.get_infomation(game)
                observation2_ = np.append(observation_, [episode_step/500])
                RL.store_transition(observation2, action, reward, observation2_)

                memory = np.hstack([observation2, [action, reward], observation2_])
                game.text_box.text_list.append(str(memory) + 'memory')
                game.render()

                # set data in screen
                game.text_box.text_list.clear()
                game.text_box.text_list.append(str(episode) + ':episode:')
                game.text_box.text_list.append(str(episode_step) + ':episode_step:')
                game.text_box.text_list.append(str(step) + ':step:')
                game.text_box.text_list.append(str(reward) + ':reward:')
                game.text_box.text_list.append(str(RL.epsilon ) + ':RL.epsilon')
                # if len(RL.cost_his) > 100:
                #     cost_ave_100 = np.average(RL.cost_his[-100:])
                #     game.text_box.text_list.append(str(cost_ave_100) + ':cost_ave_100')

                if game.command == 'setting':
                    print('number od learning:')
                    number_of_l = int(input())
                    print('eval how many times to replace SOP:')
                    RL.replace_target_iter = int(input())
                    print('learning rate:')
                    RL.lr = float(input())
                    print('gamma:')
                    RL.gamma = float(input())
                    print('epilison:')
                    RL.epsilon = float(input())

                    print('learing...', number_of_l, 'times')
                    print('target_iter:', RL.replace_target_iter)
                    print('learning rate:', RL.lr)
                    print('RL.gamma:', RL.gamma)

                    for i in range(number_of_l):
                        RL.learn()
                    RL.plot_cost()

                    print('repeat how many times??')
                    repeat = int(input())
                    while repeat != 0:
                        for r in range(repeat):
                            print('learing...', number_of_l, 'times')
                            print('target_iter:', RL.replace_target_iter)
                            print('learning rate:', RL.lr)
                            print('RL.gamma:', RL.gamma)
                            for i in range(number_of_l):
                                RL.learn()
                            if r == repeat - 1:
                                RL.plot_cost()
                                print('repeat how many times??')
                                repeat = int(input())
                    print('game.command=manual')
                    Gvar.command = 'manual'
                try:
                    number_of_l
                except NameError:
                    number_of_l = 1
                else:
                    pass
                if game.command != 'setting':
                    if (episode > 4) and(done):
                        print('learing...,', number_of_l, ',times')
                        for i in range(number_of_l):
                            RL.learn()

                # swap observation
                # observation = observation_

                # break while loop when end of this episode
                if done:
                    break
                step += 1

        # end of game
        print('game over')
        game.render()

        RL.plot_cost()
