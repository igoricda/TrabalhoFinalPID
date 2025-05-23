import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label

def otsu(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    MaxVar = 0
    T = 0  # Inicializa o limiar
    Sum = np.sum(np.arange(256) * histogram)
    
    wb = 0
    Sumb = 0

    for i in range(0, 255):
        wb += histogram[i]
        if wb == 0:
            continue  
        
        wf = np.sum(histogram) - wb
        if wf == 0:
            break 
        
        Sumb += i * histogram[i]
        mb = Sumb / wb
        mf = (Sum - Sumb) / wf
        AVar = wb * wf * (mb - mf) ** 2
        
        if AVar > MaxVar:
            MaxVar = AVar
            T = i
    
    binary_image = np.zeros_like(image)
    binary_image[image >= T] = 255
    return binary_image
    
def find_objects(img):
    visited = np.zeros_like(img, dtype=bool)
    objects = 0
    change_j = [-1, 0, 1,  -1, 1,  -1, 0, 1]  # Movimento em colunas (x)
    change_i = [-1, -1, -1,  0, 0,   1, 1, 1]  # Movimento em linhas (y)
    
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255 and not visited[i, j]:
                stack = [(i, j)]
                count = 0
                #DFS
                while stack:
                    x, y = stack.pop()
                    if not visited[x, y]:
                        visited[x, y] = True
                        for dx, dy in zip(change_i, change_j):
                            nx, ny = x + dx, y + dy
                            #Para nao extrapolar os limites da matriz
                            if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                                if img[nx, ny] == 255 and not visited[nx, ny]:
                                    stack.append((nx, ny))
                                    count += 1
                if count > 200:
                    objects += 1
    return objects



# Carregar a imagem
image = cv2.imread("./imagens/4.jpg", cv2.IMREAD_GRAYSCALE)

# Aplicar Otsu
bin_image = otsu(image)

# Inverter a imagem se a maioria dos pixels forem brancos (255)
if np.sum(bin_image == 255) > np.sum(bin_image == 0):
    bin_image = 255 - bin_image



# Contar objetos
num_objects = find_objects(bin_image)
print(f"Número de objetos encontrados: {num_objects}")

# Exibir imagens
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(bin_image, cmap="gray")
plt.title("Imagem Binarizada (Otsu)")
plt.axis("off")


plt.show()

