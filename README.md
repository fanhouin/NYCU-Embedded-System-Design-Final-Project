# NYCU-Embedded-System-Design-Final

## Goal
- Developed face recognition in E9V3 board.

## Architecture
![image](https://user-images.githubusercontent.com/46760916/190098133-8edb0cd8-21a3-40e5-a412-4c6b493fae7a.png)
- E9V3 board
    - Used OpenCV to compress the image and send it to the server for recognition via socket
    - The result is received and displayed on the screen.
- Server
    - The facial features are extracted and labeled from the image, and fed into the pretrained model to identify people. 
    - Finally, the results are sent back to the board

## Result
- We can correctly detect faces and facial features
- the performance around 2-3 frames per second. 
- The main time-consuming place is to receive pictures on the embedded device. 
- If we just show the predicted image on the server, the can around 6-7 frames per second.
![image](https://user-images.githubusercontent.com/46760916/190094343-019e020b-90fb-4035-b381-69c60d78723c.png)