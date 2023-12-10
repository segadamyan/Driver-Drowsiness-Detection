import cv2
import numpy as np
from tensorflow.keras.models import load_model


class DriverDrowsinnessModel:
    """
    Driver Drowness Model developed by Sergey Adamyan
    """
    IMG_SHAPE = (244, 244, 3)
    LABELS = ('Closed', 'Open')
    EYE_CASCADE = cv2.CascadeClassifier("/Users/sadamyan/Documents/DNN/drowsiness_detection/haarcascade.xml")

    def __init__(self, path: str):
        self.model = load_model(path)
        self.croped_windows = []
    
    def __destroy_croped(self):
        for window in self.croped_windows:
            cv2.destroyWindow(window)
    
    def __predict(self, frame, show_croped=False):
        eyes = DriverDrowsinnessModel.EYE_CASCADE.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        predictions = []
        for (x, y, w, h) in eyes:
            eye_roi = frame[y:y + h, x:x + w]

            if show_croped:
                window_name = f"window{x}{y}"
                cv2.imshow(window_name, eye_roi)
                self.croped_windows.append(window_name)

            img_array = cv2.resize(eye_roi, (DriverDrowsinnessModel.IMG_SHAPE[0], DriverDrowsinnessModel.IMG_SHAPE[1]))
            img_array = np.expand_dims(img_array, axis=0)


            prediction = self.model.predict(img_array)
            print(prediction)
            class_index = np.argmax(prediction)
            predictions.append(class_index)

        return all(0 == x for x in predictions)

    
    def run(self, show_croped = False):
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()

            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key == ord("c"):
                if self.__predict(frame, show_croped):
                    print("Closed")
                else:
                    print("Opened")
            elif key == ord("q"):
                break
            elif key == ord("x"):
                self.__destroy_croped()

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = DriverDrowsinnessModel("/Users/sadamyan/Documents/DNN/drowsiness_detection/drowiness.h5")
    model.run(False)