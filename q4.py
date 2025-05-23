import numpy as np
from PIL import Image

# Binarização da imagem lida
im_gray = np.array(Image.open('./imagens/j.png').convert('L'))
thresh = 128
maxval = 255
img = (im_gray > thresh) * maxval  # Imagem binária

# Direções da cadeia de Freeman
directions = [0, 1, 2,
              7,   3,
              6, 5, 4]

dir2idx = dict(zip(directions, range(len(directions))))

# Movimento em colunas (x) e linhas (y) correspondentes às direções de Freeman
change_j = [1, 1, 0, -1, -1, -1, 0, 1]  # Movimento em colunas (x)
change_i = [0, -1, -1, -1, 0, 1, 1, 1]  # Movimento em linhas (y)

def find_objects(img):
    visited = np.zeros_like(img, dtype=bool)
    objects = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255 and not visited[i, j]:
                objects.append((i, j))
                stack = [(i, j)]
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
    return objects

# Encontrar todos os objetos
start_points = find_objects(img)

all_chains = []
for start_point in start_points:
    border = []
    chain = []
    curr_point = start_point
    prev_direction = 3  # Começa olhando para cima 
    visited = set()
    
    while True:
        border.append(curr_point)
        visited.add(curr_point)
        found_next = False
        
        for i in range(8):  # Percorre as 8 direções possíveis
            direction = (prev_direction + i) % 8
            new_i = curr_point[0] + change_i[direction]
            new_j = curr_point[1] + change_j[direction]
            
            if 0 <= new_i < img.shape[0] and 0 <= new_j < img.shape[1]:
                if img[new_i, new_j] == 255 and (new_i, new_j) not in visited:
                    chain.append(direction)
                    curr_point = (new_i, new_j)
                    prev_direction = (direction + 5) % 8  # Ajusta a direção para a próxima iteração
                    found_next = True
                    break
        
        if not found_next or curr_point == start_point:
            break  # Encerra se não encontrar próximo ponto ou retornar ao início
    
    all_chains.append(chain)

# Exibe o código da cadeia de Freeman para cada objeto
for idx, chain in enumerate(all_chains):
    print(f"Objeto {idx + 1} - Código da Cadeia de Freeman:", chain)
