import numpy as np
from PIL import Image

# Binarização da imagem lida
im_gray = np.array(Image.open('./imagens/foto.jpg').convert('L'))
segimg = np.zeros_like(im_gray)

for i in range(im_gray.shape[0]):
        for j in range(im_gray.shape[1]):
            if im_gray[i][j] >= 0 and im_gray[i][j] < 51:
                segimg[i][j] = 25
            elif im_gray[i][j] >= 51 and im_gray[i][j] < 101:
                segimg[i][j] = 75
            elif im_gray[i][j] >= 101 and im_gray[i][j] < 151:
                segimg[i][j] = 125
            elif im_gray[i][j] >= 151 and im_gray[i][j] < 201:
                segimg[i][j] = 175
            else:
                segimg[i][j] = 255

Image.fromarray(np.uint8(segimg)).save('./fotosegmentada.jpg')
