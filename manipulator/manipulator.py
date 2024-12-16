import pygame
import time  # подключаем библиотеку, чтобы задействовать функции задержки в программе
import serial

# Фиктивный класс для эмуляции работы Arduino
class MockArduino:
    def write(self, data):
        print(f"Arduino mock received: {data.decode().strip()}")

# Заменяем реальный объект Serial фиктивным
ArduinoSerial = MockArduino()

# Определите несколько цветов.
BLACK = pygame.Color('black')
WHITE = pygame.Color('white')


# Это простой класс, который поможет нам печатать на экране.
class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()

# Установить ширину и высоту экрана (ширина, высота).
screen = pygame.display.set_mode((500, 700))
pygame.display.set_caption("My Game")

# Цикл, пока пользователь не нажмет кнопку закрытия.
done = False

# Используется для управления скоростью обновления экрана.
clock = pygame.time.Clock()

# Инициализировать джойстики.
pygame.joystick.init()

# Будьте готовы к печати.
textPrint = TextPrint()

# -------- Основной цикл программы -----------
while not done:
    # ЭТАП ОБРАБОТКИ СОБЫТИЯ
    # Возможные действия джойстика: JOYAXISMOTION, JOYBALLMOTION, JOYBUTTONDOWN,
    # JOYBUTTONUP, JOYHATMOTION
    for event in pygame.event.get():  # Пользователь что-то сделал.
        if event.type == pygame.QUIT:  # Если пользователь нажал кнопку "Закрыть".
            done = True  # Отметить, что мы закончили, чтобы выйти из этого цикла.
        elif event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        elif event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # ШАГ РИСОВАНИЯ
    screen.fill(WHITE)
    textPrint.reset()

    # Получите количество джойстиков.
    joystick_count = pygame.joystick.get_count()
    textPrint.tprint(screen, "Number of joysticks: {}".format(joystick_count))
    textPrint.indent()

    for i in range(joystick_count):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()

        try:
            jid = joystick.get_instance_id()
        except AttributeError:
            # get_instance_id () - это метод SDL2
            jid = joystick.get_id()
        textPrint.tprint(screen, f"Joystick {jid}")
        textPrint.indent()

        # Получить имя из ОС для контроллера / джойстика.
        name = joystick.get_name()
        textPrint.tprint(screen, f"Joystick name: {name}")

        try:
            guid = joystick.get_guid()
        except AttributeError:
            # get_guid () - это метод SDL2
            pass
        else:
            textPrint.tprint(screen, "GUID: {}".format(guid))

        # Обычно оси работают парами, вверх / вниз для одной и влево / вправо для
        # другой.
        axes = joystick.get_numaxes()
        textPrint.tprint(screen, "Number of axes: {}".format(axes))
        textPrint.indent()

        # Работа с осями и кнопками
        # 5 - поворот всей руки
        # 4 - наклон первого плеча
        # 3 - наклон второго плеча
        # 2 - поворот клешни
        # 1 - наклон клешни
        # 0 - сжатие клешни

        servo_config = {0: (5, 0.025), 1: (3, 0.025), 2: (1, -0.025), 3: (4, -0.025)}
        servo_configb = {4: (6, 0.025), 5: (0, -0.025), 6: (7, -0.025), 7: (2, 0.025)}

        current_angles = {servo: 90 for servo, _ in servo_config.values()}  # Все приводы стартуют с 90 градусов

        total_angle = 180  # Угол, чтобы клешня оставалась параллельной полу
        select_button = ...

        for axis, (servo, coeff) in servo_config.items():
            axis_input = joystick.get_axis(axis) #получили текущее значение оси
            angle_input = axis_input * coeff
            current_angles[servo] += angle_input  # Обновили угол

            #ArduinoSerial.write(f'{servo} {angle_input}\n'.encode())

            # Если это плечо (3 или 4), компенсируем угол клешни (servo 1)
            if servo in [3, 4] and joystick.get_button(select_button):  # Плечи
                shoulder_angle = current_angles[4]  # Угол первого плеча
                elbow_angle = current_angles[3]    # Угол второго плеча

                # Рассчитываем угол клешни
                current_angles[1] = total_angle - shoulder_angle - elbow_angle
                current_angles[1] = max(0, min(180, current_angles[1]))

            # Отправляем данные на Arduino
            ArduinoSerial.write(f'{servo} {current_angles[servo]}\n'.encode())

        for buttons1, (servo, coeff) in servo_configb.items():
            if joystick.get_button(buttons1) == 1:
                ArduinoSerial.write(f'{servo} {joystick.get_button(buttons1) * coeff}\n'.encode())
                time.sleep(0.1)
                ArduinoSerial.write(f'{servo} 0\n'.encode())

        # Отображение значений осей
        axes = joystick.get_numaxes()
        textPrint.tprint(screen, f"Number of axes: {axes}")
        textPrint.indent()
        for i in range(axes):
            axis = joystick.get_axis(i)
            textPrint.tprint(screen, f"Axis {i} value: {axis:.3f}")
        textPrint.unindent()

        # Отображение значений кнопок
        buttons = joystick.get_numbuttons()
        textPrint.tprint(screen, f"Number of buttons: {buttons}")
        textPrint.indent()
        for i in range(buttons):
            button = joystick.get_button(i)
            textPrint.tprint(screen, f"Button {i} value: {button}")
        textPrint.unindent()

        # Положение шляпы. Все или ничего для направления, а не поплавок, как
        # get_axis (). Позиция - это набор значений типа int (x, y).
        for i in range(hats):
            hat = joystick.get_hat(i)
            textPrint.tprint(screen, "Hat {} value: {}".format(i, str(hat)))
        textPrint.unindent()

        textPrint.unindent()

    pygame.display.flip()
    clock.tick(30) # увеличить число(фпс), чтобы плавнее двигался, при этом нужно уменьшить угловую скорость.

pygame.quit()
