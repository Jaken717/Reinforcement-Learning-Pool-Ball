import pygame

import numpy as np
import math

import itertools

import config
import ball
import collisions
import cue
import event
import gamestate
import physics
import table_sprites

#get the target ball from user
# def get_target():
#     with open('order.txt','r') as file:
#         array = [int(line.strip()) for line in file]
#     return array
    
#get the best trace
def get_cue_angle(game_state, number, target_hole):
    target_position = (0,0)
    cue_ball_position = (0,0)
    print(f"balls in the list {len(game_state.balls)}\n")
    for ball in game_state.balls:
        # print(f'ball number {ball.number}, position {ball.ball.pos}\n')
        if ball.number == number:
            target_position = ball.ball.pos
        if ball.number == 0:
            cue_ball_position = ball.ball.pos

    # holes_x = [config.table_margin, config.resolution[0] / 2, config.resolution[0] - config.table_margin]
    # holes_y = [config.table_margin, config.resolution[1] - config.table_margin]

    # all_hole_positions = np.array(list(itertools.product(holes_x, holes_y)))
    
    # target_hole = all_hole_positions[0]

    # for hole_pos in all_hole_positions:
    #     # print(f"cue_ball_position {cue_ball_position}, target ball position {target_position}, target pocket position {target_hole}.\n")
    #     if physics.get_angle(cue_ball_position, target_position, target_hole) > physics.get_angle(cue_ball_position, target_position, hole_pos):
    #         target_hole = hole_pos

    tangent_ball_pos = physics.get_tangent_ball_pos(target_hole, target_position)

    tangent_ball_pos = np.array(tangent_ball_pos)
    
    cue_ball_pos = np.array(cue_ball_position)

    cue_angle = physics.get_line_angle(cue_ball_pos, tangent_ball_pos)

    # print(f"cue ball is at {cue_ball_pos}, target ball {number} locates at {target_position}, tangent ball locates at {tangent_ball_pos}, aiming at pocket location {target_hole}, with angle{math.degrees(cue_angle)}\n")

    return cue_angle
    

#check if the calculated cue_angle is correct, and make possible adjusts
def calibrate(game_state):
    pass