import sys
import random

import pygame


# Set screen size
pygame.init()
SCREEN_SIZE = pygame.display.Info().current_w//2, pygame.display.Info().current_h//2

# Object dimensions
PADDLE1_WIDTH = SCREEN_SIZE[1]//8
PADDLE1_HEIGHT = SCREEN_SIZE[0]//80
PADDLE2_WIDTH = SCREEN_SIZE[1]//8
PADDLE2_HEIGHT = SCREEN_SIZE[0]//80
BALL_DIAMETER = PADDLE1_WIDTH//2

MAX_PADDLE1_Y = SCREEN_SIZE[1] - PADDLE1_WIDTH
MAX_PADDLE2_Y = SCREEN_SIZE[1] - PADDLE2_WIDTH
MAX_BALL_Y = SCREEN_SIZE[1] - BALL_DIAMETER

# Paddle X coordinate
PADDLE2_X = SCREEN_SIZE[0] - PADDLE2_HEIGHT - (BALL_DIAMETER*3)//2
PADDLE1_X = (BALL_DIAMETER*3)//2

# Color constants
BLACK = (0,0,0)
WHITE = (255,255,255)
DARK_GREEN = (0,100,0)
FIREBRICK = (178,34,34)
MEDIUM_PURPLE = (147,112,219)

# Game states
STATE_PLAYING = 0
PLAYER2_WINS = 1
PLAYER1_WINS = 2

N_DIVISION = 8
DIVISION_LENGTH = SCREEN_SIZE[1]//N_DIVISION


class Pong:
    """
    Class for Game Pong.
    """

    def __init__(self):
        # Initilize PyGame
        #pygame.init()

        # Create screen
        self.screen = pygame.display.set_mode(SCREEN_SIZE)#, pygame.FULLSCREEN)
        pygame.display.set_caption("Atari Pong")

        self.clock = pygame.time.Clock()
        if pygame.font:
            self.font =  pygame.font.SysFont('Comic Sans MS', 30)
        else:
            self.font = None

        self.game_init()

    def game_init(self):
        self.startTime = pygame.time.get_ticks()//1000
        self.currentTime = self.startTime
        self.temp = self.startTime

        self.state = STATE_PLAYING

        self.paddle2 = pygame.Rect(PADDLE2_X, (SCREEN_SIZE[1]-PADDLE2_WIDTH)//2, PADDLE2_HEIGHT, PADDLE2_WIDTH)
        self.paddle2_vel = 0

        self.paddle1 = pygame.Rect(PADDLE1_X, (SCREEN_SIZE[1]-PADDLE1_WIDTH)//2, PADDLE1_HEIGHT, PADDLE1_WIDTH)
        self.paddle1_vel = 0

        #self.ball = pygame.Rect((SCREEN_SIZE[0] - BALL_DIAMETER)//2, (SCREEN_SIZE[1] - BALL_DIAMETER)//2, BALL_DIAMETER, BALL_DIAMETER)
        self.fball = pygame.image.load('fireball1.png')
        self.fball = pygame.transform.scale(self.fball, (BALL_DIAMETER, BALL_DIAMETER))
        self.ball = self.fball.get_rect()
        self.ball = self.ball.move(((SCREEN_SIZE[0] - BALL_DIAMETER)//2, (SCREEN_SIZE[1] - BALL_DIAMETER)//2))
        self.ball_vel = random.choice([[-5, -5], [-5, 5], [5, -5], [5, 5]])

    # def check_input(self):
    #     keys = pygame.key.get_pressed()

        # if keys[pygame.K_UP]:
        #     self.paddle2_vel = -SCREEN_SIZE[1]//100
        #     self.paddle2.top += self.paddle2_vel
        #     if self.paddle2.top < 0:
        #         self.paddle2.top = 0
        #         self.paddle2_vel = 0
        # elif keys[pygame.K_DOWN]:
        #     self.paddle2_vel = SCREEN_SIZE[1]//100
        #     self.paddle2.top += self.paddle2_vel
        #     if self.paddle2.top > MAX_PADDLE2_Y:
        #         self.paddle2.top = MAX_PADDLE2_Y
        #         self.paddle2_vel = 0
        # else:
        #     self.paddle2_vel = 0

        # if keys[pygame.K_w]:
        #     self.paddle1_vel = -SCREEN_SIZE[1]//100
        #     self.paddle1.top += self.paddle1_vel
        #     if self.paddle1.top < 0:
        #         self.paddle1.top = 0
        #         self.paddle1_vel = 0
        # elif keys[pygame.K_s]:
        #     self.paddle1_vel = SCREEN_SIZE[1]//100
        #     self.paddle1.top += self.paddle1_vel
        #     if self.paddle1.top > MAX_PADDLE1_Y:
        #         self.paddle1.top = MAX_PADDLE1_Y
        #         self.paddle1_vel = 0
        # else:
        #     self.paddle1_vel = 0


        # if keys[pygame.K_r] and (self.state==PLAYER1_WINS or self.state==PLAYER2_WINS):
        #     self.game_init()

        # if keys[pygame.K_q]:
        #     pygame.quit()
        #     return True

    def move_ball(self):
        self.ball.left += self.ball_vel[0]
        self.ball.top  += self.ball_vel[1]

        if self.ball.top <= 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
            self.ball_vel[0] += -self.ball_vel[0]/abs(self.ball_vel[0]*10)
        elif self.ball.top >= MAX_BALL_Y:
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]
            self.ball_vel[0] += -self.ball_vel[0]/abs(self.ball_vel[0]*10)

    def handle_collisions(self):
        if self.ball.colliderect(self.paddle1):
            self.ball.left = PADDLE1_X + PADDLE1_HEIGHT
            self.ball_vel[1] += self.paddle1_vel//10
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.colliderect(self.paddle2):
            self.ball.left = PADDLE2_X - BALL_DIAMETER
            self.ball_vel[1] += self.paddle1_vel//10
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left < self.paddle1.left:
            self.state = PLAYER2_WINS
        elif self.ball.left+BALL_DIAMETER > self.paddle2.left+PADDLE2_HEIGHT:
            self.state = PLAYER1_WINS

    def show_stats(self, time):
            self.show_message("Time Elapsed: {}s".format(time-self.startTime), y=10)

    def show_message(self, message, x=None, y=None):
        if self.font:
            size = self.font.size(message)
            font_surface = self.font.render(message, False, WHITE)
            if x is None:
                x = (SCREEN_SIZE[0] - size[0])//2
            if y is None:
                y = (SCREEN_SIZE[1] - size[1])//2
            self.screen.blit(font_surface, (x, y))

    def draw(self):
        if self.state == PLAYER1_WINS:
            self.show_message("Player 1(Green) Wins")
            self.show_message("Press 'Enter' to play again", y=SCREEN_SIZE[1]//2+20)
            self.show_message("Press 'q' to QUIT", y=SCREEN_SIZE[1]//2+60)
        elif self.state == PLAYER2_WINS:
            self.show_message("Player 2(Red) Wins")
            self.show_message("Press 'r' to play again", y=SCREEN_SIZE[1]//2+20)
            self.show_message("Press 'q' to QUIT", y=SCREEN_SIZE[1]//2+60)

        # Draw Boundary
        pygame.draw.line(self.screen, WHITE, [PADDLE1_X, 0], [PADDLE1_X, SCREEN_SIZE[1]], 1)
        pygame.draw.line(self.screen, WHITE, [PADDLE1_X+PADDLE1_HEIGHT, 0], [PADDLE1_X+PADDLE1_HEIGHT, SCREEN_SIZE[1]], 1)
        pygame.draw.line(self.screen, WHITE, [PADDLE2_X, 0], [PADDLE2_X, SCREEN_SIZE[1]], 1)
        pygame.draw.line(self.screen, WHITE, [PADDLE2_X+PADDLE2_HEIGHT, 0], [PADDLE2_X+PADDLE2_HEIGHT, SCREEN_SIZE[1]], 1)

        # for i in range(N_DIVISION):
        #     pygame.draw.line(self.screen, WHITE, [PADDLE1_X, DIVISION_LENGTH*i], [PADDLE1_X+PADDLE1_HEIGHT, DIVISION_LENGTH*i], 1)
        #     pygame.draw.line(self.screen, WHITE, [PADDLE2_X, DIVISION_LENGTH*i], [PADDLE2_X+PADDLE2_HEIGHT, DIVISION_LENGTH*i], 1)

        # Draw paddle1
        pygame.draw.rect(self.screen, DARK_GREEN, self.paddle1)
        # Draw paddle2
        pygame.draw.rect(self.screen, FIREBRICK, self.paddle2)
        # Draw ball
        #pygame.draw.rect(self.screen, BLACK, self.ball)
        self.screen.blit(self.fball, (self.ball.left, self.ball.top))

        self.show_stats(self.currentTime)

        pygame.display.flip()

    def run(self):
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()

        self.screen.fill(MEDIUM_PURPLE)
        self.clock.tick(50)

        if self.state == STATE_PLAYING:
            self.move_ball()
            self.handle_collisions()
            self.currentTime = pygame.time.get_ticks()//1000

            if self.currentTime - self.temp > 5:
                self.ball_vel[0] += (0.5 if self.ball_vel[0]>0 else -0.5)
                self.ball_vel[1] += (0.5 if self.ball_vel[1]>0 else -0.5)
                self.temp = self.currentTime

        self.draw()
