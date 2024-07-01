import math

import numpy as np
import pygame

import config
import event
import gamestate
import physics
import calibrate


class Cue(pygame.sprite.Sprite):
    def __init__(self, target):
        pygame.sprite.Sprite.__init__(self)
        self.angle = 0
        self.power = 0
        self.color = config.player1_cue_color
        self.target_ball = target
        self.visible = False
        self.displacement = config.ball_radius
        self.sprite_size = np.repeat(
            [config.cue_length + config.cue_max_displacement], 2)
        self.clear_canvas()

    def clear_canvas(self):
        # create empty surface as a placeholder for the cue
        self.image = pygame.Surface(2 * self.sprite_size)
        self.image.fill((200, 200, 200))
        self.image.set_colorkey((200, 200, 200))
        self.rect = self.image.get_rect()
        self.rect.center = self.target_ball.ball.pos.tolist()

    def update(self, *args):
        if self.visible:
            self.image = pygame.Surface(2 * self.sprite_size)
            # color which will be ignored
            self.image.fill((200, 200, 200))
            self.image.set_colorkey((200, 200, 200))

            sin_cos = np.array([math.sin(self.angle), math.cos(self.angle)])
            initial_coords = np.array([math.sin(self.angle + 0.5 * math.pi), math.cos(self.angle +
                                                                                      0.5 * math.pi)]) * config.cue_thickness
            coord_diff = sin_cos * config.cue_length
            rectangle_points = np.array((initial_coords, -initial_coords,
                                         -initial_coords + coord_diff, initial_coords + coord_diff))
            rectangle_points_from_circle = rectangle_points + self.displacement * sin_cos
            pygame.draw.polygon(self.image, self.color,
                                rectangle_points_from_circle + self.sprite_size)
            # print(np.degrees(self.angle))

            self.points_on_screen = rectangle_points_from_circle + self.target_ball.ball.pos
            self.rect = self.image.get_rect()
            self.rect.center = self.target_ball.ball.pos.tolist()
        else:
            self.clear_canvas()

    def is_point_in_cue(self, point):
        # this algorithm splits up the rectangle into 4 triangles using the point provided
        # if the point provided is inside the triangle the sum of triangle
        # areas should be equal to that of the rectangle
        rect_sides = [config.cue_thickness * 2, config.cue_length] * 2
        triangle_sides = np.apply_along_axis(
            physics.point_distance, 1, self.points_on_screen, point)
        calc_area = np.vectorize(physics.triangle_area)
        triangle_areas = np.sum(
            calc_area(triangle_sides, np.roll(triangle_sides, -1), rect_sides))
        rect_area = rect_sides[0] * rect_sides[1]
        # +1 to prevent rounding errors
        return rect_area + 1 >= triangle_areas

    def update_cue_displacement(self, mouse_pos, initial_mouse_dist):
        displacement = physics.point_distance(
            mouse_pos, self.target_ball.ball.pos) - initial_mouse_dist + config.ball_radius
        if displacement > config.cue_max_displacement:
            self.displacement = config.cue_max_displacement
        elif displacement < config.ball_radius:
            self.displacement = config.ball_radius
        else:
            self.displacement = displacement

    def draw_lines(self, game_state, target_ball, angle, color):
        cur_pos = np.copy(target_ball.ball.pos)
        diff = np.array([math.sin(angle), math.cos(angle)])

        while config.resolution[1] > cur_pos[1] > 0 and config.resolution[0] > cur_pos[0] > 0:
            cur_pos += config.aiming_line_length * diff * 2
            pygame.draw.line(game_state.canvas.surface, color, cur_pos,
                             (cur_pos + config.aiming_line_length * diff))

    def is_clicked(self, events):
        return events["clicked"] and self.is_point_in_cue(events["mouse_pos"])

    def make_visible(self, current_player):
        if current_player == gamestate.Player.Player1:
            self.color = config.player1_cue_color
        else:
            self.color = config.player2_cue_color
        self.visible = True
        self.update()

    def make_invisible(self):
        self.visible = False

    def cue_is_active(self, game_state, target_ball, target_hole_num,adjust_angle,power):
        # initial_mouse_pos = events["mouse_pos"]
        # initial_mouse_dist = physics.point_distance(
        #     initial_mouse_pos, self.target_ball.ball.pos)

        # target_ball_no = 0
        # confirmed = False
        # while not confirmed:
        #     target_ball_no = int(input("Enter the Target Ball No.: "))
        #     if target_ball_no < 0 and target_ball_no >= 15:
        #         print("Please enter a non-negative integer.\n")
        #     else:
        #         confirmed = True

        # select_ball = False
        # while not select_ball:
        #     events = event.events()
        #     if events["clicked"]:
        #         for ball in game_state.balls:
        #             if physics.point_distance(events["mouse_pos"], ball.ball.pos) < config.ball_radius:
        #                 target_ball_no = ball.number
        #                 select_ball = True
        #                 break
        
        i = 0
        target_hole = (0,0)
        for hole_pos in config.all_hole_positions:
            if target_hole_num == i:
                target_hole = hole_pos
                break
            i = i + 1

        # if i == 0:
        #     target_hole = np.array((60, 60))
        # elif i == 1:
        #     target_hole = np.array((60, 440))
        # elif i == 4:
        #     target_hole = np.array((940, 60))
        # elif i == 5:
        #     target_hole = np.array((940, 440))
        adjust_angle = math.radians(adjust_angle)
        self.angle = calibrate.get_cue_angle(game_state=game_state, number=target_ball, target_hole=target_hole) + adjust_angle
        # undraw leftover aiming lines
        # self.draw_lines(game_state, self.target_ball, self.angle +
        #                 math.pi, config.table_color)
        self.power = power
        
        self.ball_hit()

    def ball_hit(self):
        new_velocity = 50 * self.power * np.array([math.sin(self.angle), math.cos(self.angle)])
        change_in_disp = np.hypot(*new_velocity) * 0.1
        while self.displacement - change_in_disp > config.ball_radius:
            self.displacement -= change_in_disp
            self.update()
            pygame.display.flip()
        self.target_ball.ball.apply_force(new_velocity)
        self.displacement = config.ball_radius
        self.visible = False

    def update_cue(self, game_state, initial_mouse_dist, events):
        # updates cue position
        current_mouse_pos = events["mouse_pos"]
        displacement_from_ball_to_mouse = self.target_ball.ball.pos - current_mouse_pos
        self.update_cue_displacement(current_mouse_pos, initial_mouse_dist)
        prev_angle = self.angle
        # hack to avoid div by zero
        if not displacement_from_ball_to_mouse[0] == 0:
            self.angle = 0.5 * math.pi - math.atan(
                displacement_from_ball_to_mouse[1] / displacement_from_ball_to_mouse[0])
            if displacement_from_ball_to_mouse[0] > 0:
                self.angle -= math.pi

        game_state.redraw_all(update=False)
        self.draw_lines(game_state, self.target_ball, prev_angle +
                        math.pi, config.table_color)
        self.draw_lines(game_state, self.target_ball, self.angle +
                        math.pi, (255, 255, 255))
        pygame.display.flip()
