import win32gui
import pyautogui
import cv2
import numpy as np
import time
import ctypes
import torch
import torchvision.transforms as transforms
import model.ResNet

# Windows API SendInput function
SendInput = ctypes.windll.user32.SendInput

# C struct redefinitions for keyboard and mouse input
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Function to simulate key press
def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# Function to simulate key release
def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# Function to get the position of a window by its name
def get_window_pos(name):
    win_handle = win32gui.FindWindow(0, name)
    if win_handle == 0:
        return None
    else:
        return win32gui.GetWindowRect(win_handle), win_handle

# Function to control movement using keyboard scan codes
def move(act):
    key = [0x48, 0x50, 0x4D, 0x4B]  # up, down, left, right
    press = [key[i] for i, e in enumerate(act) if e == 1]
    for i in press:
        PressKey(i)
    time.sleep(0.02)
    for i in press:
        ReleaseKey(i)
    if not press:
        time.sleep(0.02)

# Setting up device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading and preparing the neural network model
resnet = model.ResNet.ResNet18()
resnet.load_state_dict(torch.load('ResNet_1.pt'))
resnet.to(device)
resnet.eval()

# Getting window position
(x1, y1, x2, y2), handle = get_window_pos('東方紅魔郷　～ the Embodiment of Scarlet Devil')
win32gui.SetForegroundWindow(handle)  # Bring the window to the front

# Simulate key presses to navigate through the game menu
def navigate_menu():
    keys = [0x1C] * 5  # Enter key
    for key in keys:
        PressKey(key)
        time.sleep(0.1)
        ReleaseKey(key)
        time.sleep(1)

navigate_menu()

# Main loop for gameplay interaction
while True:
    # Screenshotting the game window and preprocessing for the model
    img = pyautogui.screenshot(region=[x1 + 40, y1 + 50, x2 - x1 - 320, y2 - y1 - 70])
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (122, 141))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])(img)
    img = torch.unsqueeze(img, 0).to(device)

    # Model prediction and action
    with torch.no_grad():
        action = torch.squeeze(resnet(img)) * 100
    action = action.cpu().numpy().tolist()
    action = [1 if i > 0.5 else 0 for i in action]

    # Simulate shooting and movement in the game
    PressKey(0x2C)  # Press 'Z' key for shooting
    time.sleep(0.01)
    ReleaseKey(0x2C)
    move(action)
