###############################################################
#Edition Date: Julay 1st, 2024
#Content: Reinforcement Learning Envirornment for Pool Ball Game
#Author: JK Liu
###############################################################

import pygame
import numpy as np

import collisions
import gamestate
import graphics
import config

class poolenv:
    def __init__(self) -> None:
        pygame.init()
        self.game = gamestate.GameState()
        self.was_closed = False
        self.current_player = gamestate.Player.Player1

    def reset(self):
        self.game = gamestate.GameState()
        self.game.start_pool()
        self.was_closed = False
        self.current_player = 1
        return self._get_state()
    


    def step(self,action):
        if not self.was_closed:
            self._apply_action(action)
            self.prev_balls = self.game.balls

            while not self.game.all_not_moving():
                collisions.resolve_all_collisions(self.game.balls, self.game.holes, self.game.table_sides)
                self.game.redraw_all()

            self.game.check_pool_rules()
            self.game.cue.make_invisible(self.game.current_player)

            state = self._get_state()
            reward = self._calculate_reward()
            done = self._is_done()

            return state,reward,done
    
    def _apply_action(self,action):
        self.game.cue.cue_is_active(self.game, target_ball=action.target_ball, target_hole_num=action.target_hole,adjust_angle=action.angle,power=action.power)

    def _get_state(self):
        matrix_stripes = np.zeros((37,17), dtype = int)
        matrix_solid = np.zeros((37,17), dtype = int)
        matrix_black = np.zeros((37,17), dtype = int)

        for ball in self.game.balls:
            if ball.number > 8:
                x_unit = round(ball.ball.pos[0]/25)
                y_unit = round(ball.ball.pos[1]/25)

                matrix_stripes[x_unit,y_unit] = 1
            elif ball.number < 8:
                x_unit = round(ball.ball.pos[0]/25)
                y_unit = round(ball.ball.pos[1]/25)

                matrix_solid[x_unit,y_unit] = 1
            elif ball.number == 8:
                x_unit = round(ball.ball.pos[0]/25)
                y_unit = round(ball.ball.pos[1]/25)

                matrix_black[x_unit,y_unit] = 1
        
        combined_matrix = np.stack((matrix_stripes, matrix_solid, matrix_black), axis=-1)

        return combined_matrix
    
    # matrix_stripes = combined_matrix[:, :, 0]
    # matrix_solid = combined_matrix[:, :, 1]
    # matrix_black = combined_matrix[:, :, 2]

    def _calculate_reward(self):
        if self.game.is_game_over:
            if self.current_player == gamestate.Player.Player1:
                if self.p1_won:
                    return 10
                else:
                    return -100
            else:
                if not self.p1_won:
                    return 10
                else:
                    return -100
        else:
            target_type = self.game.get_player_ball_type(self.current_player)
            reward = 0
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