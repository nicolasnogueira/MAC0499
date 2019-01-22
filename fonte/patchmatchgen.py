import numpy as np
import io
import queue
from urllib.request import urlopen
from imageio import imread
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import IPython.display as ipd
import random
import heapq

def ssd(patchA, patchB):
    #dist2 = np.abs(patchA - patchB)
    #dist2 = 100*np.linalg.norm(patchA - patchB)**2/np.linalg.norm(patchB)**2
    dist2 = (patchA - patchB)**2
    return dist2.sum()          

def rgb2gray(rgb):
    fil = [0.299, 0.587, 0.114]
    return np.dot(rgb, fil)

# converte o indice na lista de patches para as coordenadas na imagem de dado patch
# x = (int) idx/tamanho_linha, y = idx%tamanho_linha
# [coluna, linha]
def index1dto2d(idx, n, psize):
    tamanho_linha = n - (psize - 1)
    y = int (idx/tamanho_linha)
    x = idx%tamanho_linha
    return [x, y]

# converte indice de coordenadas para o indice no conjunto de patches
def index2dto1d(x, y, n, psize):
    tamanho_linha = n - (psize - 1)
    return int (y*tamanho_linha + x)

def reorganizePatches(NNF_heaps, patches, num_patches):
    newpatches = []
    for i in range(num_patches):
        best = NNF_heaps[i][0]
        for item in NNF_heaps[i]:
            if (item[0] > best[0]):
                best = item
        newpatches.append(patches[best[1]])
    return newpatches

# retorna vizinhos disponíveis de V para comparação com vizinhos de U ao redor de uma
# distância offset (retorna até 8 vizinhos de V possíveis + patch V)
# indexação cartesiana
def getNeighbors(U, V, offset, m, n, psize):
    listNeighbors = []
    listoffsets = []
    for a in np.arange(offset):
        a = a + 1
        listoffsets.append([0,-a])
        listoffsets.append([-a, 0])
        listoffsets.append([-a, -a])
        listoffsets.append([0, a])
        listoffsets.append([a, 0])
        listoffsets.append([a, a])
        listoffsets.append([-a, a])
        listoffsets.append([a, -a])

    listNeighbors.append([U,V])

    for off in listoffsets:
        if ((U[0] + off[0]) >= 0 and (U[0] + off[0]) < (n-psize-1) 
            and (V[0] + off[0]) >= 0 and (V[0] + off[0]) < (n-psize-1)
            and (U[1] + off[1]) >= 0 and (U[1] + off[1]) < (m-psize-1) 
            and (V[1] + off[1]) >= 0 and (V[1] + off[1]) < (m-psize-1)):
            listNeighbors.append([U + off, V + off])
    return listNeighbors

def getNeighborsSingle(U, offset, m, n, psize):
    listNeighbors = []
    listoffsets = []
    for a in np.arange(offset):
        a = a + 1
        listoffsets.append([0,-a])
        listoffsets.append([-a, 0])
        listoffsets.append([-a, -a])
        listoffsets.append([0, a])
        listoffsets.append([a, 0])
        listoffsets.append([a, a])
        listoffsets.append([-a, a])
        listoffsets.append([a, -a])

    listNeighbors.append([U])

    for off in listoffsets:
        if ((U[0] + off[0]) >= 0 and (U[0] + off[0]) < (n-psize-1) 
            and (U[1] + off[1]) >= 0 and (U[1] + off[1]) < (m-psize-1)):
            listNeighbors.append([U + off])
    return listNeighbors

def buildMask(maskIdxPair, m, n, psize):
    maskM = np.zeros((m,n))
    for U, V in maskIdxPair:
        for ui in np.arange(psize):
            for uj in np.arange(psize):
                maskM[U[1] + ui][U[0] + uj] = 1
                maskM[V[1] + ui][V[0] + uj] = 1 
    return maskM



IMG = rgb2gray(imread('inputs/001_F.png'))

plt.imshow(IMG, cmap='gray')
plt.show()

print(IMG.shape)

# Image shape

m = IMG.shape[0]
n = IMG.shape[1]

print("m = ", m, " n = ", n)
# construir uma matriz m x n de max-heaps contendo K correspondências aleatórias

# params
# K - quantidade de elementos no max-heap

K = 5
psize = 7
patch = [psize,psize]
num_patches = (m - (psize - 1))*(n - (psize - 1))
print("num_patches: ", num_patches)

# no lugar de trabalhar com uma matriz, trabalhamos com uma lista de heaps
NNF_heaps = [[] for i in range(num_patches)]
NNF_lists = [[] for i in range(num_patches)]

# espaços de patches contendo (m - 6)*(n - 6)
patches = image.extract_patches_2d(IMG, (psize,psize))

for i in range(num_patches):
    while (len(NNF_heaps[i]) != K):
        corresp = random.randint(0, num_patches-1)

        # n pode haver dois elementos iguais no max-heap
        exists = False
        for item in NNF_heaps[i]:
            if (item[1] == corresp):
                exists = True
                break
        if (not exists):
            # negativo da distância ssd para usar min-heap como max-heap
            tempdist = -1 * ssd(patches[i], patches[corresp])
            heapq.heappush(NNF_heaps[i], [tempdist, corresp])

            NNF_lists[corresp].append(i)

print("NNF inicializado!")

print(patches.shape)
print(len(NNF_heaps))
#print(NNF_heaps[0])

num_iters = 5

for iter in range(num_iters):
    
    # PROPAGACAO

    tempCorresp = [[-1, -1] for i in range(num_patches)]
    #print(tempCorresp)
    
    for i in range(num_patches):
        melhorCorresp = NNF_heaps[i][0]
        change = False

        # comparar o patch examinado e os MAX-HEAP dos alvos de propagação
        if (iter%2 == 1):
            if ((i - 1) >= 0):
                # esquerda
                target_max = NNF_heaps[i - 1][0]
                tempdist = -1*ssd(patches[i], patches[target_max[1]])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, target_max[1]]
                    change = True

            if ((i - (m - psize - 1)) >= 0):
                # cima
                target_max = NNF_heaps[i - (m - psize - 1)][0]
                tempdist = -1*ssd(patches[i], patches[target_max[1]])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, target_max[1]]
                    change = True

        else:
            if ((i + 1) < num_patches):
                # direita
                target_max = NNF_heaps[i + 1][0]
                tempdist = -1*ssd(patches[i], patches[target_max[1]])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, target_max[1]]
                    change = True

            if ((i + (m - psize - 1)) < num_patches):
                # baixo
                target_max = NNF_heaps[i + (m - psize - 1)][0]
                tempdist = -1*ssd(patches[i], patches[target_max[1]])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, target_max[1]]
                    change = True

        if (change):
            tempCorresp[i] = melhorCorresp


        top_actual = NNF_heaps[i][0][1]
        top_next_try = NNF_heaps[top_actual][0][1]
        tempdist = -1*ssd(patches[i], patches[top_next_try])
        if (tempdist > NNF_heaps[i][0][0]):
            # candidato melhora o heap do patch i
            tempCorresp[i] = [tempdist, top_next_try]

    # faz as alteracoes da propagacao        
    for i in range(num_patches):
        if (tempCorresp[i][1] != -1):
            exists = False
            for item in NNF_heaps[i]:
                if (item[1] == tempCorresp[i][1]):
                    exists = True
                    break
            if (not exists):
                removed = heapq.heappop(NNF_heaps[i])
                NNF_lists[removed[1]].remove(i)
                heapq.heappush(NNF_heaps[i], tempCorresp[i])
                NNF_lists[tempCorresp[i][1]].append(i)
    print("Propagação feita! [", iter, "]")
    
    # BUSCA ALEATORIA

    # Seja v_0 o atual nearest neighbor, w é a maior dimensão da imagem, 
    # R_i é um uniforme aleatório [-1,1]x[-1,1]
    # alpha é a razão entre o tamanho de janelas consecutivas

    # i = 0,1,2... até que w*alpha**i < 1 
    # u_i = v_0 + w*R_i*alpha**i

    w = max(m-psize-1,n-psize-1)
    alpha = 0.5

    tempCorresp = [[-1, -1] for i in range(num_patches)]
    
    for j in range(num_patches):
        melhorCorresp = NNF_heaps[j][0]
        change = False
        for item in NNF_heaps[j]:
            i = 0

            searchradius = w*(alpha**i)

            while (searchradius > 1):
                [v_x,v_y] = index1dto2d(item[1], n, psize)

                progress = False

                while (not progress):
                    u_x = v_x + searchradius*random.randint(-1,1)
                    if (u_x >= 0 and u_x < (n-psize-1)):
                        progress = True

                progress = False

                while (not progress):
                    u_y = v_y + searchradius*random.randint(-1,1)
                    if (u_y >= 0 and u_y < (m-psize-1)):
                        progress = True

                v_index1d = index2dto1d(u_x, u_y, n, psize)
                # print("j ", j, " v_index1d ", v_index1d)
                tempdist = -1*ssd(patches[j], patches[v_index1d])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, v_index1d]
                    change = True

                i = i + 1
                searchradius = w*(alpha**i)

        if (change):
            tempCorresp[j] = melhorCorresp

    # alteracoes da busca aleatoria
    for i in range(num_patches):
        if (tempCorresp[i][1] != -1):
            exists = False
            for item in NNF_heaps[i]:
                if (item[1] == tempCorresp[i][1]):
                    exists = True
                    break
            if (not exists):
                removed = heapq.heappop(NNF_heaps[i])
                NNF_lists[removed[1]].remove(i)
                heapq.heappush(NNF_heaps[i], tempCorresp[i])
                NNF_lists[tempCorresp[i][1]].append(i)
    print("Busca aleatória feita! [", iter, "]")

    # ENRIQUECIMENTO

    # inverso
    tempCorresp = [[-1, -1] for i in range(num_patches)]

    for i in range(num_patches):
        melhorCorresp = NNF_heaps[i][0]
        change = False
        for index in NNF_lists[i]:
            tempdist = -1*ssd(patches[i], patches[index])
            if (tempdist > melhorCorresp[0]):
                melhorCorresp = [tempdist, index]
                change = True
        if (change):
            tempCorresp[i] = melhorCorresp

    # alteracoes do enriq inverso
    for i in range(num_patches):
        if (tempCorresp[i][1] != -1):
            exists = False
            for item in NNF_heaps[i]:
                if (item[1] == tempCorresp[i][1]):
                    exists = True
                    break
            if (not exists):
                removed = heapq.heappop(NNF_heaps[i])
                NNF_lists[removed[1]].remove(i)
                heapq.heappush(NNF_heaps[i], tempCorresp[i])
                NNF_lists[tempCorresp[i][1]].append(i)

    print("Enriquecimento inverso feito! [", iter, "]")
    
    # direto
    tempCorresp = [[-1, -1] for i in range(num_patches)]

    for i in range(num_patches):
        melhorCorresp = NNF_heaps[i][0]
        change = False
        for item in NNF_heaps[i]:
            idxitemitem = item[1]
            for itemitem in NNF_heaps[idxitemitem]:
                tempdist = -1*ssd(patches[i], patches[itemitem[1]])
                if (tempdist > melhorCorresp[0]):
                    melhorCorresp = [tempdist, itemitem[1]]
                    change = True
        if (change):
            tempCorresp[i] = melhorCorresp

    # alteracoes do enriq direto
    for i in range(num_patches):
        if (tempCorresp[i][1] != -1):
            exists = False
            for item in NNF_heaps[i]:
                if (item[1] == tempCorresp[i][1]):
                    exists = True
                    break
            if (not exists):
                removed = heapq.heappop(NNF_heaps[i])
                NNF_lists[removed[1]].remove(i)
                heapq.heappush(NNF_heaps[i], tempCorresp[i])
                NNF_lists[tempCorresp[i][1]].append(i)

    print("Enriquecimento direto feito! [", iter, "]")

# Reconstrução aproximada da imagem tomando média dos overlaps

newpatches = reorganizePatches(NNF_heaps, patches, num_patches)
NEWIMG = image.reconstruct_from_patches_2d(np.array(newpatches), IMG.shape).astype(np.uint8)

plt.imshow(NEWIMG, cmap='gray')
plt.show()

print("Patchmatch Generalizado finalizado!")
print("Distorção: ", (np.linalg.norm(IMG - NEWIMG)**2/np.linalg.norm(IMG)**2)*100, "%")

# DETECÇÃO DE CÓPIA COLAGEM

T = 20 # distância euclidiana limite entre patches
D = 100 # distância de similaridade limite
offset = 2 # distância de procura ao redor do patch alvo

detec_map = np.zeros(IMG.shape)
detec_list = []
detec_list_idx = []
# u patch atual, v patch a ser testado
# counter = 0

# percorre NNF final da esquerda para a direita de cima para baixo
for i in range(num_patches):
    u = index1dto2d(i, n, psize)

    # visitamos o maxheap de cada patch
    for item in NNF_heaps[i]:
        v = index1dto2d(item[1], n, psize)

        dist = np.linalg.norm(np.asarray(u)-np.asarray(v))
        if (dist > T):
            valid_neighbors = getNeighbors(u, v, offset, m, n, psize)
            for neighbor in valid_neighbors:
                nindex1dU = index2dto1d(neighbor[0][0], neighbor[0][1], n, psize) # vizinho do patch
                nindex1dV = index2dto1d(neighbor[1][0], neighbor[1][1], n, psize) # vizinho da correspondencia

                # if (detec_map[neighbor[0][1]][neighbor[0][0]] == 1 and detec_map[neighbor[1][1]][neighbor[1][0]] == 1):
                #    continue

                distSSD = ssd(patches[nindex1dU], patches[nindex1dV])
                if (distSSD < D):
                    u_x1, u_y1 = index1dto2d(nindex1dU, n, psize)
                    v_x1, v_y1 = index1dto2d(nindex1dV, n, psize)

                    detec_list_idx.append([[u_x1, u_y1], [v_x1, v_y1]]) # adicionando no mapa de deteccao

NNEWIMG = buildMask(detec_list_idx, m, n, psize)
#print(detec_map)
plt.imshow(NNEWIMG, cmap='gray')
plt.show()


