import cv2
import mediapipe as mp
import numpy as np
import random
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def classify_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    fingers = []

    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    thumb = 1 if landmarks[4].x < landmarks[3].x else 0
    fingers.insert(0, thumb)

    if fingers == [0, 0, 0, 0, 0]:
        return "Rock"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Paper"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Scissors"
    else:
        return "Unknown"

def check_winner(player, ai):
    if player == ai:
        return "Tie"
    if (player == "Rock" and ai == "Scissors") or \
       (player == "Paper" and ai == "Rock") or \
       (player == "Scissors" and ai == "Paper"):
        return "You Win!"
    return "AI Wins!"

cap = cv2.VideoCapture(0)
player_score, ai_score = 0, 0
choices = ["Rock", "Paper", "Scissors"]
last_move_time = time.time()
cooldown = 3
player_move = "Waiting..."

# Initialize ball properties
ball_radius = 20
ball_x = 320  # Initial horizontal position (center)
ball_y = 400  # Fixed vertical position
ball_color = (0, 255, 255)  # Yellow color
ball_status = "In play"

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_height, img_width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_detected = False
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            lm_list = handLms.landmark
            player_move = classify_gesture(lm_list)

            # Control ball position based on hand landmark 9 (middle finger MCP)
            hand_detected = True
            hand_x = int(lm_list[9].x * img_width)
            hand_y = int(lm_list[9].y * img_height)

            # Update ball x position based on hand x position
            ball_x = hand_x

            # Check if hand is down (y coordinate greater than threshold)
            if hand_y > img_height * 0.7:
                ball_status = "Out"
            else:
                ball_status = "In play"

    # Cooldown logic
    if time.time() - last_move_time > cooldown and player_move in choices:
        ai_move = random.choice(choices)
        result = check_winner(player_move, ai_move)

        if result == "You Win!":
            player_score += 1
        elif result == "AI Wins!":
            ai_score += 1

        last_move_time = time.time()
        print(f"Player: {player_move}, AI: {ai_move}, Result: {result}")

    # Draw the ball
    if ball_status == "In play":
        cv2.circle(img, (ball_x, ball_y), ball_radius, ball_color, -1)
    else:
        # Optionally, draw the ball in red or hide it when out
        cv2.putText(img, "Ball Out!", (ball_x - 50, ball_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display UI
    cv2.putText(img, f"Your Move: {player_move}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"AI Move: {ai_move if 'ai_move' in locals() else ''}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, f"Score - You: {player_score}  AI: {ai_score}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"Ball Status: {ball_status}", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors - Hand Gesture Game", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
