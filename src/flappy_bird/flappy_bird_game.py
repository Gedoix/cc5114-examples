import random

import pygame
from typing import List, Tuple


WIDTH, HEIGHT = 800, 600

LOG = {"Game": True}


class AI:

    def choose(self, horizontal_distance: float, vertical_distance: float,
               horizontal_speed: float, vertical_speed: float) -> List[bool]:
        return [True]


class TestAI(AI):

    def choose(self, horizontal_distance: float, vertical_distance: float,
               horizontal_speed: float, vertical_speed: float) -> List[bool]:
        return [True] if input("Write 'f' to flap:\n") == 'f' else [False]


class Bird:

    ai: AI
    speed: Tuple[float, float]
    position: Tuple[float, float]

    def __init__(self, ai: AI):
        self.position = (0.2, 0.5)
        self.speed = (0.01, 0.0)
        self.living = True
        self.ai = ai

    def step(self, horizontal_distance: float, vertical_distance: float):

        flap = self.ai.choose(horizontal_distance, vertical_distance, self.speed[0], self.speed[1])[0]

        if flap:
            self.speed = (0.01, 0.1)
        else:
            if self.speed[1] > 0:
                self.speed = (0.01, self.speed[1]-0.01)

        self.position = (self.position[0] + self.speed[0], self.position[1] + self.speed[1])

    def get_position(self) -> Tuple[float, float]:
        return self.position

    def kill(self) -> None:
        self.living = False

    def is_alive(self) -> bool:
        return self.living


class Game:

    def __init__(self):
        # Placeholders for future fields, having to do with pygame objects
        self.screen = None
        self.font = None

        # Pygame initialization
        if LOG["Game"]:
            print("[Game] Initializing Pygame")
        pygame.init()
        if LOG["Game"]:
            print("[Game] Initializing Pygame Font")
        pygame.font.init()
        if LOG["Game"]:
            print("[Game] Setting up Font")
        self.font = pygame.font.SysFont("Consolas", 100)

    @staticmethod
    def quit():
        # Pygame is closed
        if LOG["Game"]:
            print("[Game] Quitting Pygame Font")
        pygame.font.quit()
        if LOG["Game"]:
            print("[Game] Quitting Pygame")
        pygame.quit()

    def open_display(self):
        if LOG["Game"]:
            print("[Game] Initializing Pygame Display")
        pygame.display.init()
        # Pygame display and text generator are created
        if LOG["Game"]:
            print("[Game] Setting up Pygame Display")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")

    @staticmethod
    def close_display():
        if LOG["Game"]:
            print("[Game] Quitting Pygame Display")
        pygame.display.quit()

    def simulate(self, birds: List[Bird], seed):
        pass

    def show(self, bird: Bird, seed, fps: int = 30):
        if LOG["Game"]:
            print("[Game] Initializing Graphic Simulation")

        self.open_display()

        # Step clock
        if LOG:
            print("[Game] Setting up Pygame Clock")
        clock = pygame.time.Clock()

        # The fruit is generated
        generator = random.Random(seed)

        # The score is initialized
        distance = 0

        # Main loop
        if LOG:
            print("[Game] Entering Main Loop")
        while bird.is_alive():
            # A frame is rendered
            # TODO: Render
            pygame.display.update()
            clock.tick(fps)
            # The display can be closed
            if pygame.event.peek(pygame.QUIT):
                break
            # Events are cleared to avoid "not-responding" reaction from the system
            pygame.event.clear()
            # A game step is calculated
            # TODO:  bird.step()
            # If a fruit was eaten, it resets and adds to the score
            # TODO: Kill bird
            # if contact(bird.get_position(), ):
            #     bird.kill()
            distance += 1

        # End game screen
        if LOG:
            print("[Game] Drawing End Screen")
        # TODO: End game screen
        pygame.display.update()
        clock.tick(60)

        if LOG["Game"]:
            print("[Game] Quitting Graphic Simulation")

        self.close_display()
