import random

import pygame

from typing import List
from typing import Tuple

# Display constants
WIDTH = 800
HEIGHT = 600

# Color constants
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)


class Game:

    def __init__(self, cells_width: int, cells_height: int):
        self.width = cells_width
        self.height = cells_height

    def simulate(self, snakes: List[Snake], seeds: List[int]) -> (List[Snake], List[int]):
        generators = []
        fruits = []
        scores = []
        for i in range(len(snakes)):
            generators.append(random.Random(seeds[i]))
            fruits.append((generators[i].randint(0, self.width), generators[i].randint(0, self.height)))
            while snakes[i].covers(fruits[i]):
                fruits[i] = (generators[i].randint(0, self.width), generators[i].randint(0, self.height))
            scores.append(0)
        while True:
            any_alive = False
            for i in range(len(snakes)):
                if snakes[i].isAlive():
                    any_alive = True
                    snakes[i].step(fruits[i], self.width, self.height)
                    if snakes[i].at(fruits[i]):
                        while snakes[i].covers(fruits[i]):
                            fruits[i] = (generators[i].randint(0, self.width), generators[i].randint(0, self.height))
                        scores[i] += 1
            if not any_alive:
                break
        return snakes, scores

    def show(self, snake: Snake, seed: int, subtitle: str, fps: int = 2) -> None:
        pygame.init()
        pygame.font.init()

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")

        # Step clock
        clock = pygame.time.Clock()

        generator = random.Random(seed)
        fruit = (generator.randint(0, self.width), generator.randint(0, self.height))
        while snake.covers(fruit):
            fruit = (generator.randint(0, self.width), generator.randint(0, self.height))
        score = 0
        while snake.isAlive():
            clock.tick(fps)
            snake.step(fruit, self.width, self.height)
            if snake.at(fruit):
                while snake.covers(fruit):
                    fruit = (generator.randint(0, self.width), generator.randint(0, self.height))
                score += 1
            screen.fill(COLOR_BLACK)
            self.draw_board(screen, COLOR_WHITE)
            self.draw_fruit(screen, COLOR_RED, fruit)
            self.draw_snake(screen, COLOR_BLUE, snake)
            self.draw_score(screen, COLOR_WHITE, score)
            self.draw_subtitle(screen, COLOR_WHITE, subtitle)
        pygame.font.quit()
        pygame.quit()

    def draw_board(self, screen: pygame.display, color: Tuple[int, int, int]) -> None:
        for i in range(self.width+1):
            x = int(WIDTH/2 + (WIDTH/2)*(i/self.width))
            pygame.draw.line(screen, color, (x, HEIGHT), (x, HEIGHT/3), 5)
        for j in range(self.height+1):
            y = int(HEIGHT/3 + (HEIGHT*2/3)*(j/self.height))
            pygame.draw.line(screen, color, (WIDTH, y), (WIDTH/2, y), 5)

    def fill_cell(self, screen: pygame.display, color: Tuple[int, int, int], i: int, j: int):
        xi = int(WIDTH/2 + (WIDTH/2)*(i/self.width))+3
        xf = int(WIDTH/2 + (WIDTH/2)*((i+1)/self.width))-3
        yi = int(HEIGHT/3 + (HEIGHT*2/3)*(j/self.height))+3
        yf = int(HEIGHT/3 + (HEIGHT*2/3)*((j+1)/self.height))-3

        r = pygame.Rect((xi, yi), (xf-xi, yf-yi))

        pygame.draw.rect(screen, color, r, 0)

    def draw_fruit(self, screen: pygame.display, color: Tuple[int, int, int], fruit: Tuple[int, int]):
        self.fill_cell(screen, color, fruit[0], fruit[1])

    def draw_snake(self, screen: pygame.display, color: Tuple[int, int, int], snake: Snake):
        for i, j in snake.get_tail():
            self.fill_cell(screen, color, i, j)

    def draw_score(self, screen: pygame.display, color: Tuple[int, int, int], score: int):
        font = pygame.font.SysFont("Consolas", 15)
        text_surface = font.render("score: "+str(score), False, color)
        screen.blit(text_surface, (0, 0))

    def draw_subtitle(self, screen: pygame.display, color: Tuple[int, int, int], subtitle: str):
        font = pygame.font.SysFont("Consolas", 15)
        text_surface = font.render(subtitle, False, color)
        screen.blit(text_surface, (0, 0))


class Snake:

    def __init__(self, cells_width: int, cells_height: int):
        self.width = cells_width
        self.height = cells_height
        self.positions = []
        self.positions.append((int(self.width), int(self.height)))


