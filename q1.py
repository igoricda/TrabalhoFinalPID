import numpy as np 
import cv2
import matplotlib
import matplotlib.pyplot as plt

###########################################################################################
#                               RESPOSTA DA QUESTAO 2                                     #
###########################################################################################
# Marr-Hildreth: Suaviza a imagem com um filtro passa-baixa e calcula-se o lapaciano da ima-
# gem, obtendo-se a imagem LoG, entao se obtem os cruzamentos de zero na imagem LoG, testa-
# do na vizinhanca-8 de um pixel p, os sinais de pelo menos dois de seus vizinhos opostos 
# sejam diferentes, ao encontrar pares com sinais diferentes, compara-se a diferenca entre
# eles com um limiar T, calculado com T = 0.02 * np.max(np.abs(log)), se a diferenca for 
# maior que T, o ponto e marcado como borda
#
# Canny: Suaviza a imagem com um gaussiano passa-baixa, entao se obtem os gradientes x e y 
# por Sobel e se calcula a magnitude da imagem com magnitude = np.sqrt(grad_x**2 + grad_y**2)
# e as direcoes das bordas, com direction = np.arctan2(grad_y, grad_x) * 180 / np.pi. Apos 
# isso, e aplicada a supressao nao maxima na imagem, onde, em cada pixel, se pega o angulo
# a partir da matriz direction e se normaliza o angulo com angle % 180, entao, a partir do 
# angulo, se compara a intensidade do pixel com seus vizinhos na direcao especifica, e se 
# este pixel for maior que estes vizinhos ele e mantido na matriz supressed. Em sequencia
# e feita a dupla limiarizacao atraves dos limiares recebidos, com strong_edges recebendo
# os pontos de supressed de intensidade maior ou igual ao limiar alto e as bordas fracas
# recebem os pontos que estao entre os 2 limiares, entao e criada a matriz edges, que nos
# pontos de borda forte recebe 255 e nos de borda fraca recebe 75, entao e feita a analise
# de conectividade, testando nas weak edges se tem uma strong edge na vizinhanca-8 e, se sim,
# esse ponto recebe 255, se nao, esse ponto é descartado, recebendo 0.
#
# O canny produz bordas mais suaves, é menos sensivel a ruidos e tem bordas melhor localizadas
# que o marr-hildreth, que produz varias bordas falsas por ser muito sensivel ao ruido, tambem 
# tem dificuldade com bordas curvas, com alguns pontos deslocados nessas curvas.
#
###########################################################################################
#                          INICIO DAS FUNCOES AUXILIARES                                  #
###########################################################################################

def erode(image, kernel):
    maxval = 255  # Branco
    kernel_rows, kernel_cols = kernel.shape
    k_center_x = kernel_cols // 2
    k_center_y = kernel_rows // 2

    # Criação da imagem com padding
    rows, columns = image.shape
    padded_image = np.zeros((rows + 2 * k_center_y, columns + 2 * k_center_x), np.uint8)
    padded_image[k_center_y:k_center_y + rows, k_center_x:k_center_x + columns] = image

    # Criação da imagem de saída
    eroded_image = np.zeros_like(padded_image)

    # Itera sobre cada pixel da imagem com padding
    for i in range(k_center_y, rows + k_center_y):
        for j in range(k_center_x, columns + k_center_x):
            # Define a região da imagem sob o kernel
            region = padded_image[i - k_center_y:i + k_center_y + 1, j - k_center_x:j + k_center_x + 1]
            region = np.array(region)
            match = True
            for x in range(kernel_rows):
                for y in range(kernel_cols):
                    if kernel[x][y] == 255:
                        if kernel[x][y] != region[x][y]: #Testa cada pixel da regiao e, se um for diferente, muda a variavel match para falso
                            match = False
                            break
                if match == False:
                    break
                        
            if match:  # Se for verdadeiro
                eroded_image[i, j] = maxval  # Iguala o pixel central a branco

    # Remove o padding para retornar à dimensão original
    return eroded_image[k_center_y:-k_center_y, k_center_x:-k_center_x]
    
# Função de dilatação
def dilate(image, kernel):
    maxval = 255  # Branco
    kernel_rows, kernel_cols = kernel.shape
    k_center_x = kernel_cols // 2
    k_center_y = kernel_rows // 2

    # Criação da imagem com padding
    rows, columns = image.shape
    padded_image = np.zeros((rows + 2 * k_center_y, columns + 2 * k_center_x), np.uint8)
    padded_image[k_center_y:k_center_y + rows, k_center_x:k_center_x + columns] = image

    # Criação da imagem de saída
    dilated_image = np.zeros_like(padded_image)

    # Itera sobre cada pixel da imagem com padding
    for i in range(k_center_y, rows + k_center_y):
        for j in range(k_center_x, columns + k_center_x):
            # Define a região da imagem sob o kernel
            region = padded_image[i - k_center_y:i + k_center_y + 1, j - k_center_x:j + k_center_x + 1]
            region = np.array(region)
            match = False
            for x in range(kernel_rows):
                for y in range(kernel_cols):
                    if kernel[x][y] == 255:
                        if kernel[x][y] == region[x][y]: #Testa cada pixel da regiao e, se um for igual, muda a variavel match para verdadeiro
                            match = True
                            break
                if match:
                    break
            
            if match:  #Se match for verdadeiro
                dilated_image[i, j] = maxval  # Iguala o pixel central a branco

    return dilated_image[k_center_y:-k_center_y, k_center_x:-k_center_x]

def abertura(image, kernel):
    im_erode = erode(image, kernel)
    return dilate(im_erode, kernel)
    
#DFS
def connected_components(binary_image):
  
    labels = np.zeros_like(binary_image, dtype=np.int32)
    current_label = 1

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255 and labels[i, j] == 0:
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if labels[x, y] == 0:
                        labels[x, y] = current_label
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < binary_image.shape[0] and 0 <= ny < binary_image.shape[1]:
                                if binary_image[nx, ny] == 255 and labels[nx, ny] == 0:
                                    stack.append((nx, ny))
                current_label += 1

    return labels

def distance_transform(binary_image):
    rows, cols = binary_image.shape
    distance = np.zeros_like(binary_image, dtype=np.float32)

    # Inicializa mapa de distancia
    distance[binary_image == 1] = float('inf')

    # Primeira passada - acima a esquerda para embaixo a direita
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 0:
                distance[i, j] = 0
            else:
                if i > 0:
                    distance[i, j] = min(distance[i, j], distance[i-1, j] + 1)
                if j > 0:
                    distance[i, j] = min(distance[i, j], distance[i, j-1] + 1)
                if i > 0 and j > 0:
                    distance[i, j] = min(distance[i, j], distance[i-1, j-1] + np.sqrt(2))

    # Segunda passada: embaixo a direita para acima a esquerda refinando as distancias
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            if i < rows-1:
                distance[i, j] = min(distance[i, j], distance[i+1, j] + 1)
            if j < cols-1:
                distance[i, j] = min(distance[i, j], distance[i, j+1] + 1)
            if i < rows-1 and j < cols-1:
                distance[i, j] = min(distance[i, j], distance[i+1, j+1] + np.sqrt(2))

    return distance
    
###########################################################################################
#                              FIM DAS FUNCOES AUXILIARES                                 #
###########################################################################################


def marr_hildreth(image, sigma):
    size = int(2*(np.ceil(3*sigma))+1) #Tamanho do kernel baseado no sigma

    #Gera coordenadas para um kernel quadrado 
    x, y = np.meshgrid(np.arange(-size/ 2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2) #Normalizacao da equacao

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal #LoG

    kern_size = kernel.shape[0]
    log = np.zeros_like(image, dtype=float)

     #Aplicacao do filtro
    for i in range(image.shape[0]-(kern_size-1)):
        for j in range(image.shape[1]-(kern_size-1)):
            window = image[i:i+kern_size, j:j+kern_size] * kernel
            log[i, j] = np.sum(window)

    log = log.astype(np.int64, copy=False)

    # Define T como uma fração do valor máximo da matriz LoG
    T = 0.02 * np.max(np.abs(log))
    zero_crossing = np.zeros_like(log)

    # Cruzamento de zeros
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if (((log[i][j-1] != 0 and log[i][j+1] != 0) and (log[i][j-1] * log[i][j+1]) <= 0) and np.abs(log[i][j-1] - log[i][j+1]) > T) or \
               (((log[i-1][j] != 0 and log[i+1][j] != 0) and (log[i-1][j] * log[i+1][j] <= 0)) and np.abs(log[i-1][j] - log[i+1][j]) > T) or \
               (((log[i-1][j-1] != 0 and log[i+1][j+1] != 0) and (log[i-1][j-1] * log[i+1][j+1] <= 0)) and np.abs(log[i-1][j-1] - log[i+1][j+1]) > T) or \
               (((log[i-1][j+1] != 0 and log[i+1][j-1] != 0) and (log[i-1][j+1] * log[i+1][j-1] <= 0)) and np.abs(log[i-1][j+1] - log[i+1][j-1]) > T):
                zero_crossing[i][j] = 255
    return zero_crossing

def canny(img, sigma, low_thresh, high_thresh):
   
# Filtro gaussiano
    size = int(2 * (np.ceil(3 * sigma)) + 1)
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                       np.arange(-size // 2 + 1, size // 2 + 1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) * normal
    kernel = kernel / np.sum(kernel) 
    
    #padding para nao alucinar bordas
    pad_size = kernel.shape[0] // 2 
    img_padded = np.pad(img, pad_size, mode='reflect')


    # Aplicar filtro
    kern_size = kernel.shape[0]
    gauss = np.zeros_like(img, dtype=float)
    for i in range(img_padded.shape[0] - (kern_size - 1)):
        for j in range(img_padded.shape[1] - (kern_size - 1)):
            window = img_padded[i:i + kern_size, j:j + kern_size]
            gauss[i, j] = np.sum(window * kernel)

    #Retirar padding
    gauss = gauss[pad_size:-pad_size, pad_size:-pad_size]
    
    # Calcular gradiente com Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = np.zeros_like(gauss, dtype=float)
    grad_y = np.zeros_like(gauss, dtype=float)

    for i in range(1, gauss.shape[0] - 1):
        for j in range(1, gauss.shape[1] - 1):
            grad_x[i, j] = np.sum(gauss[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            grad_y[i, j] = np.sum(gauss[i - 1:i + 2, j - 1:j + 2] * sobel_y)

    #Calcular magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    #Calcular direcoes
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # Aplicar supressao nao maxima para afinar as bordas
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            #Pegar o angulo a partir da direcao
            angle = direction[i, j]
            #Normalizar para [0, 180)
            angle = angle % 180
            #Manter o maximo local
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180): #Bordas horizontais
                if magnitude[i, j] >= magnitude[i, j-1] and magnitude[i, j] >= magnitude[i, j+1]:
                    suppressed[i, j] = magnitude[i, j]
            elif (22.5 <= angle < 67.5): #Bordas diagonais (positiva)
                if magnitude[i, j] >= magnitude[i-1, j-1] and magnitude[i, j] >= magnitude[i+1, j+1]:
                    suppressed[i, j] = magnitude[i, j]
            elif (67.5 <= angle < 112.5): #Bordas verticais
                if magnitude[i, j] >= magnitude[i-1, j] and magnitude[i, j] >= magnitude[i+1, j]:
                    suppressed[i, j] = magnitude[i, j]
            elif (112.5 <= angle < 157.5): #Bordas diagonais (negativa)
                if magnitude[i, j] >= magnitude[i-1, j+1] and magnitude[i, j] >= magnitude[i+1, j-1]:
                    suppressed[i, j] = magnitude[i, j]

    # Dupla limiarizacao
    strong_edges = (suppressed >= high_thresh)
    weak_edges = (suppressed >= low_thresh) & (suppressed < high_thresh)

    # Analise de conectividade
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    edges[strong_edges] = 255
    edges[weak_edges] = 75

    # Conectar as bordas
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(edges[i - 1:i + 2, j - 1:j + 2] == 255):
                    edges[i, j] = 255
                else:
                    edges[i, j] = 0

    return edges
    
def otsu(image):
    #1)Construir o histograma da imagem;
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    #2) Fazer MaxVar = T = Sum = 0;
    MaxVar = 0
    t = 0
    #3) Calcular Sum = ∑L−1 i ∗ fi ;
    Sum = np.sum(np.arange(256) * histogram)
    #inicializar demais variaveis
    wb = 0
    Sumb = 0

    #4) para i (0 a L-1)
    for i in range (0, 255):
        wb += histogram[i]  #1) wb = wb+fi;
        if wb == 0:
            continue  # 2) se (wb = 0) continue;
        
        wf = np.sum(histogram) - wb  # 3) wf = Número de pixels da imagem – wb;
        if wf == 0:
            break  # 4) se (wf = 0) break;
        
        Sumb += i * histogram[i]  # 5) Sumb = Sumb + i * fi; mb = Sumb/wb; 
        
        # 5)  mb = Sumb/wb;
        mb = Sumb / wb 
        # 5) mf = (sum –Sumb)/wf;
        mf = (Sum - Sumb) / wf 
        
        # 5) AVar = wb * wf * (mb - mf)^2;
        AVar = wb * wf * (mb - mf) ** 2
        
        # 6) se (AVar > MaxVar) MaxVar = Avar e Limiar = i.
        if AVar > MaxVar:
            MaxVar = AVar
            T = i
    
    binary_image = np.zeros_like(image)
    binary_image[image >= T] = 255
    return binary_image
    
def watershed(image):
    #Aplicar Otsu para binarizar a imagem e separar os objetos do fundo
    bin_image = otsu(image)
    kernel = np.ones((3, 3), np.uint8)
    kernel *= 255
    
    #abertura para retirar ruidos
    openimage = abertura(bin_image, kernel)
    
    #Dilata a imagem 3 vezes para obter o sure backgroud (Com certeza é fundo)
    sure_bg = dilate(openimage, kernel)
    sure_bg = dilate(sure_bg, kernel)
    sure_bg = dilate(sure_bg, kernel)
    
    #Obtém a transformada de distancia para obter o limiar do sure foreground
    dist_transform = distance_transform(openimage)
    threshold_value = 0.7 * np.max(dist_transform)
    
    #Obtem o sure foreground e se estiver acima do limiar, coloca como 255, pois com certeza eh um objeto da imagem
    sure_fg = np.zeros_like(image, dtype=np.uint8)
    sure_fg[dist_transform > threshold_value] = 255
    
    #Subtrair o sure_bg de sure_fg, para obter a regiao que nao ha certeza se eh objeto ou fundo
    unknown = sure_bg - sure_fg
    
    #Separa os objetos marcando-os, com o fundo marcado como 0
    markers = connected_components(sure_fg)
    #Adiciona 1 para todos os marcadores para que sure bg seja 1 e nao 0
    markers = markers + 1
    #Marca toda a regiao desconhecida como zero
    markers[unknown == 255] = 0
    
    image[markers == -1] = 255
    
    return image

#MAIN
image = cv2.imread("./imagens/3.png", cv2.IMREAD_GRAYSCALE)
sigma = 2
low_thresh = 50
high_thresh = 100
# Exibir os resultados em uma única tela
plt.figure(figsize=(15, 5))


plt.subplot(1, 5, 1)
plt.imshow(image, cmap='gray')
plt.title("Imagem Original")

plt.subplot(1, 5, 2)
plt.imshow(marr_hildreth(image, sigma), cmap='gray')
plt.title('Marr-Hildreth')


plt.subplot(1, 5, 3)
plt.imshow(canny(image, sigma, low_thresh, high_thresh), cmap='gray')
plt.title('Canny Edge Detection')

plt.subplot(1, 5, 4)
plt.imshow(otsu(image), cmap='gray')
plt.title('Otsu Thresholding')


plt.subplot(1, 5, 5)
plt.imshow(watershed(image))
plt.title('Watershed Segmentation')


plt.tight_layout()
plt.show()


