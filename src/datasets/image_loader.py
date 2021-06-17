import cv2 as cv


def opencv_loader(path):
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print(f'Excertion {e} while reading image {path}.')
