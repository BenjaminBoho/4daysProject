import win32gui
import pyautogui
import cv2
import numpy as np
import time
import csv
import keyboard
import os

key_list = [0, 0, 0, 0]  # 上、下、左、右
done = False

def get_window_pos(name):
    win_handle = win32gui.FindWindow(None, name)
    if win_handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(win_handle), win_handle

def print_pressed_keys(e):
    global key_list
    key_list = [0, 0, 0, 0]
    get_key_list = [code for code in keyboard._pressed_events]
    if 72 in get_key_list:
        key_list[0] = 1
    if 80 in get_key_list:
        key_list[1] = 1
    if 75 in get_key_list:
        key_list[2] = 1
    if 77 in get_key_list:
        key_list[3] = 1

def check_exit(e):
    global done
    if e.name == 'esc':
        done = True

result = get_window_pos('東方紅魔郷　～ the Embodiment of Scarlet Devil')
if result is None:
    print("Window not found. Please check the window title and make sure it's open.")
    exit(1)
else:
    (x1, y1, x2, y2), handle = result

text = win32gui.SetForegroundWindow(handle)  # Bring window to front
keyboard.hook(print_pressed_keys)
keyboard.on_press(check_exit)

time.sleep(15)

csv_folder = '../Dataset'
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

image_directory = '../Dataset/Capture'
if not os.path.exists(image_directory):
    # If it doesn't exist, create it
    os.makedirs(image_directory)

i = 0
while not done:
    img = pyautogui.screenshot(region=[x1 + 40, y1 + 50, x2 - x1 - 320, y2 - y1 - 70])  # x,y,w,h
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (122, 141))
    img.save('../Dataset/Capture/' + str(i) + '.jpg')
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    print(key_list)
    with open("../Dataset/KeyCapture.csv", "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(key_list)
    i += 1

    # Check every few iterations to keep the loop responsive
    time.sleep(0.1)

# Cleanup: Remove the keyboard hook
keyboard.unhook_all()
