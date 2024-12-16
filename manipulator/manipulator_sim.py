import pygame
import math

pygame.init()

window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Моделирование руки с Pygame")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

shoulder_angle = 45
elbow_angle = 45
gripper_angle = 180

upper_arm_length = 150  # длина первого плеча
lower_arm_length = 120  # длина второго плеча
gripper_length = 50  # длина клешни

def draw_arm(shoulder_angle, elbow_angle, gripper_angle):
    # Начальная точка
    shoulder_x = 400
    shoulder_y = 300

    # Позиция локтя
    elbow_x = shoulder_x + upper_arm_length * math.cos(math.radians(shoulder_angle))
    elbow_y = shoulder_y - upper_arm_length * math.sin(math.radians(shoulder_angle))

    # Позиция кисти
    gripper_x = elbow_x + lower_arm_length * math.cos(math.radians(shoulder_angle + elbow_angle))
    gripper_y = elbow_y - lower_arm_length * math.sin(math.radians(shoulder_angle + elbow_angle))

    # Рисуем плечо
    pygame.draw.line(screen, RED, (shoulder_x, shoulder_y), (elbow_x, elbow_y), 5)
    # Рисуем локоть
    pygame.draw.line(screen, GREEN, (elbow_x, elbow_y), (gripper_x, gripper_y), 5)

    # Рисуем клешню
    gripper_end_x = gripper_x + gripper_length * math.cos(math.radians(gripper_angle))
    gripper_end_y = gripper_y - gripper_length * math.sin(math.radians(gripper_angle))
    pygame.draw.line(screen, BLUE, (gripper_x, gripper_y), (gripper_end_x, gripper_end_y), 5)

running = True
while running:
    screen.fill(WHITE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        shoulder_angle -= 1
    if keys[pygame.K_RIGHT]:
        shoulder_angle += 1
    if keys[pygame.K_UP]:
        elbow_angle -= 1
    if keys[pygame.K_DOWN]:
        elbow_angle += 1
    if keys[pygame.K_w]:
        gripper_angle -= 1
    if keys[pygame.K_s]:
        gripper_angle += 1

    shoulder_angle = max(0, min(180, shoulder_angle))
    elbow_angle = max(0, min(180, elbow_angle))
    gripper_angle = max(0, min(180, gripper_angle))

    draw_arm(shoulder_angle, elbow_angle, gripper_angle)

    pygame.display.flip()

    pygame.time.Clock().tick(60)

pygame.quit()
