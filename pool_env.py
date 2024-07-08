###############################################################
#Edition Date: Julay 1st, 2024
#Content: Reinforcement Learning Envirornment for Pool Ball Game
#Author: JK Liu
###############################################################

import pygame
import numpy as np

import collisions
import gamestate
from ball import BallType
import graphics
import config

class poolenv:
    def __init__(self) -> None:
        pygame.init()
        self.game = gamestate.GameState()
        self.was_closed = False
        self.game.current_player = gamestate.Player.Player1

    def reset(self):
        # self.game = gamestate.GameState()
        # print(id(self.game))
        # self.game.reset_state()
        self.game.start_pool()
        self.was_closed = False
        self.game.current_player = gamestate.Player.Player1
        return self._get_state()
    
    def step(self,action):
        self.prev_balls = self.game.balls
        self.game.redraw_all()
        self._apply_action(action)
        # print(f'step white ball 1st hit is set is {self.game.white_ball_1st_hit_is_set}')
        # print(f'step table coloring is  {self.game.table_coloring}')

        while not self.game.all_not_moving():
            collisions.resolve_all_collisions(self.game.balls, self.game.holes, self.game.table_sides)
            self.game.redraw_all()

        penalize, p1_won = self.game.check_pool_rules()
        self.game.cue.make_visible(self.game.current_player)

        if self.game.can_move_white_ball:
            self.game.white_ball.is_active(self.game, self.game.is_behind_line_break())

        state = self._get_state()
        reward = self._calculate_reward(penalize,p1_won)
        done = self._is_done()

        return state,reward,done
    
    def _apply_action(self,action):
        self.game.cue.cue_is_active(self.game, target_ball=action.target_ball, target_hole_num=action.target_hole,adjust_angle=action.angle,power=action.power)

    def _get_state(self):
        matrix_stripes = np.zeros((38,18), dtype = int)
        matrix_solid = np.zeros((38,18), dtype = int)
        matrix_black = np.zeros((38,18), dtype = int)
        matrix_player = np.zeros((38,18), dtype = int)

        for ball in self.game.balls:
            if ball.number > 8:
                x_unit = round(ball.ball.pos[0]/25) - 1
                y_unit = round(ball.ball.pos[1]/25) - 1

                matrix_stripes[x_unit,y_unit] = ball.number
                if self.game.ball_assignment != None and self.game.ball_assignment[self.game.current_player] == BallType.Striped:
                    matrix_player[x_unit,y_unit] = 1
            elif ball.number < 8:
                x_unit = round(ball.ball.pos[0]/25) - 1
                y_unit = round(ball.ball.pos[1]/25) - 1

                matrix_solid[x_unit,y_unit] = ball.number
                if self.game.ball_assignment != None and self.game.ball_assignment[self.game.current_player] == BallType.Solid:
                    matrix_player[x_unit,y_unit] = 1
            elif ball.number == 8:
                x_unit = round(ball.ball.pos[0]/25) - 1
                y_unit = round(ball.ball.pos[1]/25) - 1

                matrix_black[x_unit,y_unit] = ball.number
                if self.game.potting_8ball[self.game.current_player]:
                    matrix_player[x_unit,y_unit] = 1
        
        combined_matrix = np.stack((matrix_stripes, matrix_solid, matrix_black, matrix_player), axis=-1)

        return combined_matrix
    
    # matrix_stripes = combined_matrix[:, :, 0]
    # matrix_solid = combined_matrix[:, :, 1]
    # matrix_black = combined_matrix[:, :, 2]

    def _calculate_reward(self, penalize, p1_won):
        reward = 0
        if penalize == True:
            reward -= 100
        if self.game.is_game_over:
            if self.game.current_player == gamestate.Player.Player1:
                if p1_won:
                    return 100
                else:
                    return -100
            else:
                if not p1_won:
                    return 100
                else:
                    return -100
        else:
            target_type = self.game.get_player_ball_type(self.game.current_player)
            for prev_ball in self.prev_balls:
                if prev_ball.number not in [ball.number for ball in self.game.balls]:
                    # Ball was potted
                    if (target_type == "Striped" and prev_ball.number > 8) or (target_type == "Solid" and prev_ball.number < 8):
                        reward += 10  # Correct ball type potted
                    else:
                        reward -= 5  # Incorrect ball type potted
            
            return reward
                

    def _is_done(self):
        return self.game.is_game_over

    def render(self):
        self.game.redraw_all()

    def close(self):
        pygame.quit()