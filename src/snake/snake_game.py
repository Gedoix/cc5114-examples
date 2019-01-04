import random

import pygame

from typing import List, Optional
from typing import Tuple

# Display constants
WIDTH = 800
HEIGHT = 600

# Color constants
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_DARK_GREEN = (0, 170, 0)

# Directions
NORTH = 0
WEST = 1
SOUTH = 2
EAST = 3

# Biggest integer
INFINITY: int = 2**63 - 1

# Log mode
LOG = {"TestAI": False, "Snake": False, "Game": True}


def normalize_distances(x: float) -> float:
    import math
    # import numpy as np
    # return (math.exp(-np.logaddexp(0.0, -4.9*float(x)))-0.5)*2.0
    return math.tanh(x)


class AI:

    def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
               distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
               distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float) -> List[bool]:
        return [False, True, False]


class TestAI(AI):
    """
    Test AI for a snake class, to be used only upon execution of this script (for testing).
    """

    def choose(self, distance_wall_left: float, distance_wall_front: float, distance_wall_right: float,
               distance_tail_left: float, distance_tail_front: float, distance_tail_right: float,
               distance_fruit_left: float, distance_fruit_front: float, distance_fruit_right: float) -> List[bool]:
        """
        Choice function of the AI agent, asks pygame's input key events to know what to do.

        :param distance_wall_left: Ignored
        :param distance_wall_front: Ignored
        :param distance_wall_right: Ignored
        :param distance_tail_left: Ignored
        :param distance_tail_front: Ignored
        :param distance_tail_right: Ignored
        :param distance_fruit_left: Ignored
        :param distance_fruit_front: Ignored
        :param distance_fruit_right: Ignored
        :return: Left direction if 'a' or '<-' are pressed, Right direction if 'd' or '->' are pressed, and
        front direction for any other key, or none
        """

        usage = (distance_wall_left + distance_wall_front + distance_wall_right +
                 distance_tail_left + distance_tail_front + distance_tail_right +
                 distance_fruit_left + distance_fruit_front + distance_fruit_right)*0.0

        # Results vector
        result = [False, False, False] if usage == 0.0 else [False, False, False]

        # A choice is asked for
        if LOG["TestAI"]:
            print("[TestAI] Asking for Input")
        choice = input("enter 'a' to turn left or 'd' to turn right, anything else advances forward\n")

        if choice == 'a':
            result[0] = True
        elif choice == 'd':
            result[2] = True
        else:
            result[1] = True

        if LOG["TestAI"]:
            print("[TestAI] Giving Result")
        return result


class Snake:
    """
    Snake class for running inside the Snake game.

    Contains a list of the positions of the cells it occupies, as well as it's own AI and step method.
    """
    cells_per_side: int
    positions: List[Tuple[int, int]]
    current_direction: int
    ai: AI
    living_state: bool

    def __init__(self, cells_per_side: int, ai: AI):
        """
        Constructor of the Snake.

        :param cells_per_side: Width in cells of the play board
        :param ai: Controller of the Snake, must have a ".choose(...)" method receiving 9 parameters, the distances
        from the Snake's head to the edge of the board, to it's own tail, and to the fruit, in each of the three
        directions left, front and right (respective to the current direction of it's movement)
        """
        if LOG["Snake"]:
            print("[Snake] Initializing Snake")
        self.cells_per_side = cells_per_side
        self.positions = []
        self.positions.append((max(int(self.cells_per_side / 2) - 1, 0), max(int(self.cells_per_side / 2) - 1, 0)))
        self.current_direction = NORTH
        self.ai = ai
        self.living_state = True

    def step(self, fruit: Tuple[int, int]) -> None:
        """
        Runs a single in game step for the Snake.

        Calls to ai for instructions on which direction to follow.

        Advances the snake's body and may add a new section to it's end.

        :param fruit: Position coordinates of the fruit
        """
        # First, the distances to be passed to ai are re-calculated
        # The order is [NORTH, SOUTH, WEST, EAST]
        # Distance to the walls of the board
        distances_to_walls = [self.positions[0][1], self.cells_per_side - self.positions[0][1] - 1,
                              self.positions[0][0], self.cells_per_side - self.positions[0][0] - 1]

        # Shortest distances to the snake's tail (one of them is guaranteed to be 1 if the
        # snake's length is more than 1)
        distances_to_tail: List[int] = [INFINITY, INFINITY, INFINITY, INFINITY]
        for position in self.positions[1:]:
            if position[0] == self.positions[0][0]:
                if position[1] < self.positions[0][1]:
                    distances_to_tail[0] = min(distances_to_tail[2], self.positions[0][1]-position[1]-1)
                else:
                    distances_to_tail[1] = min(distances_to_tail[3], position[1]-self.positions[0][1]-1)

            if position[1] == self.positions[0][1]:
                if position[0] < self.positions[0][0]:
                    distances_to_tail[2] = min(distances_to_tail[0], self.positions[0][0]-position[0]-1)
                else:
                    distances_to_tail[3] = min(distances_to_tail[1], position[0]-self.positions[0][0]-1)

        # Distances to the fruit
        distances_to_fruit = [INFINITY, INFINITY, INFINITY, INFINITY]
        if self.positions[0][0] == fruit[0]:
            if self.positions[0][1] < fruit[1]:
                distances_to_fruit[0] = min(distances_to_fruit[2], fruit[1]-self.positions[0][1]-1)
            else:
                distances_to_fruit[1] = min(distances_to_fruit[3], self.positions[0][1]-fruit[1]-1)

        if self.positions[0][1] == fruit[1]:
            if self.positions[0][0] < fruit[0]:
                distances_to_fruit[2] = min(distances_to_fruit[0], fruit[0]-self.positions[0][0]-1)
            else:
                distances_to_fruit[3] = min(distances_to_fruit[1], self.positions[0][0]-fruit[0]-1)

        # The indices of each direction are found
        if self.current_direction == NORTH:
            left = 2
            front = 0
            right = 3
        elif self.current_direction == SOUTH:
            left = 3
            front = 1
            right = 2
        elif self.current_direction == WEST:
            left = 1
            front = 2
            right = 0
        else:
            left = 0
            front = 3
            right = 1

        # The ai is consulted
        next_direction = self.ai.choose(normalize_distances(distances_to_walls[left]/self.cells_per_side),
                                        normalize_distances(distances_to_walls[front]/self.cells_per_side),
                                        normalize_distances(distances_to_walls[right]/self.cells_per_side),
                                        normalize_distances(distances_to_tail[left]/self.cells_per_side),
                                        normalize_distances(distances_to_tail[front]/self.cells_per_side),
                                        normalize_distances(distances_to_tail[right]/self.cells_per_side),
                                        normalize_distances(distances_to_fruit[left]/self.cells_per_side),
                                        normalize_distances(distances_to_fruit[front]/self.cells_per_side),
                                        normalize_distances(distances_to_fruit[right]/self.cells_per_side))

        # Direction is updated
        if next_direction[0]:
            self.current_direction += 1
            if self.current_direction == 4:
                self.current_direction = NORTH
        elif next_direction[2]:
            self.current_direction -= 1
            if self.current_direction == -1:
                self.current_direction = EAST

        # The head moves
        last_edited_position = self.positions[0]

        if self.current_direction == NORTH:
            self.positions[0] = (self.positions[0][0], self.positions[0][1]-1)
        elif self.current_direction == SOUTH:
            self.positions[0] = (self.positions[0][0], self.positions[0][1]+1)
        elif self.current_direction == WEST:
            self.positions[0] = (self.positions[0][0]-1, self.positions[0][1])
        else:
            self.positions[0] = (self.positions[0][0]+1, self.positions[0][1])

        # Checks for end of game conditions
        if self.positions[0][0] < 0 or self.positions[0][0] >= self.cells_per_side or \
                self.positions[0][1] < 0 or self.positions[0][1] >= self.cells_per_side:
            self.living_state = False

        # The tail moves
        for i, position in enumerate(self.positions[1:]):
            i += 1
            if i != len(self.positions)-1:
                if self.positions[0] == position:
                    self.living_state = False
                for other_position in self.positions[i+1:-1]:
                    if position == other_position:
                        self.living_state = False
            self.positions[i] = last_edited_position
            last_edited_position = position

        # If a fruit was eaten, the tail gains a new section
        if self.positions[0] == fruit:
            self.positions.append(last_edited_position)

    def at(self, cell: Tuple[int, int]) -> bool:
        """
        Checks whether the snake's head is at a particular position.

        :param cell: Position to check
        :return: Whether the head is there
        """
        return cell[0] == self.positions[0][0] and cell[1] == self.positions[0][1]

    def covers(self, cell: Tuple[int, int]) -> bool:
        """
        Checks whether any of the snake's body sections covers a particular position.

        :param cell: Position to check
        :return: Whether the body covers it
        """
        for position in self.positions:
            if cell[0] == position[0] and cell[1] == position[1]:
                return True
        return False

    def is_alive(self) -> bool:
        """
        Checks whether the snake is alive.

        :return: Whether the snake lives
        """
        return self.living_state

    def kill(self) -> None:
        """
        Kills the snake object.
        """
        self.living_state = False

    def get_tail(self) -> List[Tuple[int, int]]:
        """
        Gets the list of positions of each of the snake's body sections.

        :return: List of section positions, ordered starting from the head
        """
        return self.positions

    def get_ai(self):
        """
        Gets the AI agent choosing in side the snake.

        :return: The snake's AI agent
        """
        return self.ai


class Game:
    """
    Game instance class for Snake, uses Snake type objects in it's main loop.

    Can either simulate multiple games of snake without graphics or show a game for a single snake in a separate window.
    """
    cells_per_side: int
    screen: Optional[pygame.display.__class__]
    font: Optional[pygame.font.FontType]

    def __init__(self, cells_per_side: int):
        """
        Constructor of a Game instance, Sets the board's width and height.

        :param cells_per_side: The board's width in cells
        """
        if LOG["Game"]:
            print("[Game] Initializing Game")
        self.cells_per_side = cells_per_side

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

    def simulate(self, snakes: List[Snake], seed: int) -> Tuple[List[int], List[int]]:
        """
        Snake game simulator, can run multiple Snake object's games at the same time.

        Output's values for calculating each snake's AI's respective fitness, like their respective game score and
        the amount of cycles they took to end the game

        :param snakes: List of snakes to simulate
        :param seed: Pseudo-random number generator seed, shared among snakes
        :return: A list of the snakes in their final states, a list of their scores and a
        list of how long they lasted in the game
        """
        if LOG["Game"]:
            print("[Game] Initializing Simulation")
        # Random number generators
        generators = []
        # Each game's fruits
        fruits = []
        # Each game's scores
        scores = []
        # Each game's step counter
        times = []
        # Values are initialized
        for i in range(len(snakes)):
            generators.append(random.Random(seed))

            fruits.append(self.__generate_fruit(generators[i]))
            while snakes[i].covers(fruits[i]):
                fruits[i] = self.__generate_fruit(generators[i])

            scores.append(0)
            times.append(0)

        if LOG["Game"]:
            print("[Game] Entering Simulation Loop")

        # Simulation loop, lasts until there's no more snakes alive (capping at 10000 steps per snake)
        any_alive = True
        while any_alive:

            # Stop condition resets
            any_alive = False

            # For each snake
            for i in range(len(snakes)):

                # Checks if alive
                if snakes[i].is_alive():

                    # Stop condition updates
                    any_alive = True

                    # Step
                    snakes[i].step(fruits[i])

                    # If fruit was eaten, it resets and adds to the score
                    if snakes[i].at(fruits[i]):
                        fruits[i] = self.__generate_fruit(generators[i])
                        while snakes[i].covers(fruits[i]):
                            fruits[i] = self.__generate_fruit(generators[i])
                        scores[i] += 1

                    # The step counter increases
                    times[i] += 1

                    # Checks if it's time to stop
                    if times[i] > max(30*scores[i], 100):
                        snakes[i].kill()

        if LOG["Game"]:
            print("[Game] Quitting Simulation")

        return scores, times

    def show(self, snake: Snake, seed: int, subtitle: str, fps: int = 1) -> None:
        """
        Simulates a single game of snake, displaying the process on a separate window

        :param snake: Snake to simulate
        :param seed: Pseudo-random number generator seed
        :param subtitle: Subtitle of the window
        :param fps: Steps (frames) per second of the window
        """

        if LOG["Game"]:
            print("[Game] Initializing Graphic Simulation")

        self.open_display()

        # Step clock
        if LOG:
            print("[Game] Setting up Pygame Clock")
        clock = pygame.time.Clock()

        # The fruit is generated
        generator = random.Random(seed)
        fruit = self.__generate_fruit(generator)
        while snake.covers(fruit):
            fruit = self.__generate_fruit(generator)

        # The score is initialized
        score = 0

        # Main loop
        if LOG:
            print("[Game] Entering Main Loop")
        while snake.is_alive():
            # A frame is rendered
            self.screen.fill(COLOR_BLACK)
            self.__draw_board(COLOR_WHITE)
            self.__draw_fruit(COLOR_RED, fruit)
            self.__draw_snake(COLOR_GREEN, snake, COLOR_DARK_GREEN)
            self.__draw_score(COLOR_WHITE, score)
            self.__draw_subtitle(COLOR_WHITE, subtitle)
            pygame.display.update()
            clock.tick(fps)
            # The display can be closed
            if pygame.event.peek(pygame.QUIT):
                break
            # Events are cleared to avoid "not-responding" reaction from the system
            pygame.event.clear()
            # A game step is calculated
            snake.step(fruit)
            # If a fruit was eaten, it resets and adds to the score
            if snake.at(fruit):
                fruit = self.__generate_fruit(generator)
                while snake.covers(fruit):
                    fruit = self.__generate_fruit(generator)
                score += 1

        # End game screen
        if LOG:
            print("[Game] Drawing End Screen")
        self.screen.fill(COLOR_RED)
        self.__draw_score(COLOR_BLACK, score)
        pygame.display.update()
        clock.tick(60)

        if LOG["Game"]:
            print("[Game] Quitting Graphic Simulation")

        self.close_display()

    def __generate_fruit(self, generator: random.Random) -> Tuple[int, int]:
        """
        Generates a fruit's position from a pseudo-random generator.

        :param generator: Pseudo-random number generator
        :return: A position within the board, uniformly chosen
        """
        return generator.randint(0, self.cells_per_side - 1), generator.randint(0, self.cells_per_side - 1)

    def __draw_board(self, color: Tuple[int, int, int]) -> None:
        """
        Draws the board's edges into the display.

        :param color: Color of the lines
        """
        # Vertical lines
        for i in range(self.cells_per_side + 1):
            x = int(WIDTH / 2 + (WIDTH/2) * (i / self.cells_per_side))
            pygame.draw.line(self.screen, color, (x, HEIGHT), (x, HEIGHT/3), 5)
        # Horizontal lines
        for j in range(self.cells_per_side + 1):
            y = int(HEIGHT / 3 + (HEIGHT*2/3) * (j / self.cells_per_side))
            pygame.draw.line(self.screen, color, (WIDTH, y), (WIDTH/2, y), 5)

    def __fill_cell(self, color: Tuple[int, int, int], i: int, j: int) -> None:
        """
        Fills a single cell of the board with a particular color.

        :param color: Color to fill the cell with
        :param i: Horizontal coordinate of the cell
        :param j: Vertical coordinate of the cell
        """
        xi = int(WIDTH / 2 + (WIDTH/2) * (i / self.cells_per_side)) + 3
        xf = int(WIDTH / 2 + (WIDTH/2) * ((i+1) / self.cells_per_side)) - 3
        yi = int(HEIGHT / 3 + (HEIGHT*2/3) * (j / self.cells_per_side)) + 3
        yf = int(HEIGHT / 3 + (HEIGHT*2/3) * ((j+1) / self.cells_per_side)) - 3

        r = pygame.Rect((xi, yi), (xf-xi, yf-yi))

        pygame.draw.rect(self.screen, color, r, 0)

    def __draw_fruit(self, color: Tuple[int, int, int], fruit: Tuple[int, int]) -> None:
        """
        Calls on '__fill_cell(...)' to draw a fruit on the board with a certain color.

        :param color: The fruit's color.
        :param fruit: The fruit's position.
        """
        self.__fill_cell(color, fruit[0], fruit[1])

    def __draw_snake(self, color: Tuple[int, int, int], snake: Snake, head_color: Tuple[int, int, int] = None) -> None:
        """
        Draws a snake's body into the board with a certain color.

        A second color for distinguishing it's head may be included.

        :param color: The body color of the snake
        :param snake: The snake to draw
        :param head_color: The snake's head color, if any
        """
        tail = snake.get_tail()
        if head_color is None:
            for i, j in tail:
                self.__fill_cell(color, i, j)
        else:
            self.__fill_cell(head_color, tail[0][0], tail[0][1])
            for i, j in tail[1:]:
                self.__fill_cell(color, i, j)

    def __draw_score(self, color: Tuple[int, int, int], score: int) -> None:
        """
        Adds the score counter to the game's display.

        :param color: Color of the text
        :param score: Score to display
        """
        text_surface = self.font.render("score: "+str(score), False, color)
        self.screen.blit(text_surface, (0, 0))

    def __draw_subtitle(self, color: Tuple[int, int, int], subtitle: str) -> None:
        """
        Adds a subtitle string to the game's display.

        :param color: Color of the text
        :param subtitle: String to write
        """
        text_surface = self.font.render(subtitle, False, color)
        self.screen.blit(text_surface, (0, 100))


def __test():
    """
    Function for testing the program, only executes when running the script.
    """
    __test_width = 11
    __test_height = 11
    __test_fps = 1
    __test_seed = 4
    __test_ai = TestAI()
    __test_snake = Snake(__test_width, __test_ai)
    __test_game = Game(__test_width)
    __test_game.show(__test_snake, __test_seed, "Test run of the program", __test_fps)


# Execution as main
if __name__ == '__main__':
    __test()
