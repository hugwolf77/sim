import pygame
import sys

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

white = (255, 255, 255)
black = (0, 0, 0)

pygame.init()
pygame.display.set_caption("PyGame Example")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# FPS
clock = pygame.time.Clock()

# Resource load
# 이미지 불러오기
background = pygame.image.load("[project_resource_path]/background.png")
character = pygame.image.load("[project_resource_path]/character.png")
character_size = character.get_rect().size # 이미지의 크기
character_width = character_size[0] # image 가로 크기
character_height = character_size[1] # image 세로 크기
character_x_pos = (SCREEN_WIDTH - character_width) / 2 # 화면 가로의 절반 크기에 해당하는 곳에 위치
character_y_pos = SCREEN_HEIGHT - character_height # 화면 세로의 크기 가장 아래에 해당하는 곳에 위치
enemy = pygame.image.load("C:/Users/user/Desktop/PythonWorkspace/pygame_basic/enemy.png")
enemy_size = enemy.get_rect().size # 이미지의 크기를 구해옴
enemy_width = enemy_size[0] # 캐릭터의 가로 크기
enemy_height = enemy_size[1] # 캐릭터의 세로 크기
enemy_x_pos = (SCREEN_WIDTH - enemy_width) / 2 # 화면 가로의 절반 크기에 해당하는 곳에 위치
enemy_y_pos = (SCREEN_HEIGHT - enemy_height) / 2 # 화면 세로의 크기 가장 아래에 해당하는 곳에 위치


# init values 
game_font = pygame.font.Font(None, 40) # 폰트 객체 생성 (폰트, 크기)
total_time = 10
start_ticks = pygame.time.get_ticks() # start time

# move postion
to_x = 0
to_y = 0

# speed
character_speed = 0.6

# game loop
runging = True
while True:
    dt = clock.tick(60) # set FPS

    # event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN: # 키가 눌러졌는지 확인
            if event.key == pygame.K_LEFT: # 캐릭터를 왼쪽으로
                to_x -= character_speed
            elif event.key == pygame.K_RIGHT: # 캐릭터를 오른쪽으로
                to_x += character_speed
            elif event.key == pygame.K_UP: # 캐릭터를 위로
                to_y -= character_speed
            elif event.key == pygame.K_DOWN: # 캐릭터를 아래로
                to_y += character_speed

        if event.type == pygame.KEYUP: # 방향키를 떼면 멈춤
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                to_x = 0
            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                to_y = 0

    character_x_pos += to_x * dt
    character_y_pos += to_y * dt

    if character_x_pos < 0:
        character_x_pos = 0
    elif character_x_pos > SCREEN_WIDTH - character_width:
        character_x_pos = SCREEN_WIDTH - character_width

    if character_y_pos < 0:
        character_y_pos = 0
    elif character_y_pos > SCREEN_HEIGHT - character_height:
        character_y_pos = SCREEN_HEIGHT - character_height

    # collision rect
    character_rect = character.get_rect()
    character_rect.left = character_x_pos
    character_rect.top = character_y_pos

    enemy_rect = enemy.get_rect()
    enemy_rect.left = enemy_x_pos
    enemy_rect.top = enemy_y_pos

    # collision check
    if character_rect.colliderect(enemy_rect):
        print("Impack")
        running = False


    screen.fill(black)
    pygame.draw.circle(screen, white, (pos_x, pos_y), 20)
    pygame.display.update()