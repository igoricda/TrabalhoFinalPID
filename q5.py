import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carregar a imagem em escala de cinza
im_gray = np.array(Image.open('./imagens/7.jpg').convert('L'))

# Criar imagens vazias para armazenar os resultados
im2x2 = np.zeros_like(im_gray)
im3x3 = np.zeros_like(im_gray)
im5x5 = np.zeros_like(im_gray)
im7x7 = np.zeros_like(im_gray)

# Criar kernels de diferentes tamanhos (Box Filter)
kernel2x2 = np.ones((2, 2)) * (1 / 4)
kernel3x3 = np.ones((3, 3)) * (1 / 9)
kernel5x5 = np.ones((5, 5)) * (1 / 25)
kernel7x7 = np.ones((7, 7)) * (1 / 49)

# Adicionar padding às imagens para preservar o tamanho original após a convolução
pad2, pad3, pad5, pad7 = 1, 1, 2, 3

im_padded_2x2 = np.pad(im_gray, pad2, mode='constant', constant_values=0)
im_padded_3x3 = np.pad(im_gray, pad3, mode='constant', constant_values=0)
im_padded_5x5 = np.pad(im_gray, pad5, mode='constant', constant_values=0)
im_padded_7x7 = np.pad(im_gray, pad7, mode='constant', constant_values=0)

# Aplicar Box Filter para cada kernel
for i in range(im_gray.shape[0]):  
    for j in range(im_gray.shape[1]):
        im2x2[i, j] = np.sum(im_padded_2x2[i:i + 2, j:j + 2] * kernel2x2)
        im3x3[i, j] = np.sum(im_padded_3x3[i:i + 3, j:j + 3] * kernel3x3)
        im5x5[i, j] = np.sum(im_padded_5x5[i:i + 5, j:j + 5] * kernel5x5)
        im7x7[i, j] = np.sum(im_padded_7x7[i:i + 7, j:j + 7] * kernel7x7)

# Criar uma tela com 2 linhas e 3 colunas para exibir as imagens
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Mostrar imagens
axes[0, 0].imshow(im_gray, cmap='gray')
axes[0, 0].set_title("Original")

axes[0, 1].imshow(im2x2, cmap='gray')
axes[0, 1].set_title("Filtro 2x2")

axes[0, 2].imshow(im3x3, cmap='gray')
axes[0, 2].set_title("Filtro 3x3")

axes[1, 0].imshow(im5x5, cmap='gray')
axes[1, 0].set_title("Filtro 5x5")

axes[1, 1].imshow(im7x7, cmap='gray')
axes[1, 1].set_title("Filtro 7x7")

# Remover eixos para uma visualização mais limpa
for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

