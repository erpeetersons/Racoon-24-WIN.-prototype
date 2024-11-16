import cv2
import time

image_when_person = cv2.imread("film_on.jpg") # Image to display when a person is detected.
image_when_no_person = cv2.imread("film_off.jpg") # Image to display when a person is not detected.

new_image_size = (1280, 720) # To what width, hight images should be resized.

min_face_size = (30, 30) # What should be the minimal width, hight for detecting a face.

time_buffer = 1 # How many seconds should the buffer be before the images change.

camera_index = 0 # Index of camera. If camera does not open with 0, try increasing this integer.

if __name__ == "__main__":
    image_when_person = cv2.resize(image_when_person, new_image_size)
    image_when_no_person = cv2.resize(image_when_no_person, new_image_size)

    current_image = image_when_no_person

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    camera = cv2.VideoCapture(camera_index)

    last_detection_time = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)

        if len(faces) > 0:
            if time.time() - last_detection_time > time_buffer:
                current_image = image_when_person
                last_detection_time = time.time()
        else:
            if time.time() - last_detection_time > time_buffer:
                current_image = image_when_no_person
                last_detection_time = time.time()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Camera - Face Detection", frame)
        cv2.imshow("Image", current_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()