from xmodelvlm.train.train import train

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == "__main__":
    train()
