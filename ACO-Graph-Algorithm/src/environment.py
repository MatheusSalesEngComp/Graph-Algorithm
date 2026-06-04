import math
import numpy as np

class TSPEnvironment:
    def __init__(self, dados_tsplib):
        self.nome = dados_tsplib.get('NAME', 'TSP_Problem')
        self.n_cidades = dados_tsplib['DIMENSION']
        self.coordenadas = np.array(dados_tsplib['COORDS'])
        self.distancias = self._calcular_matriz_distancias()
        self.k_vizinhos = max(5, int(self.n_cidades * 0.20))
        self.lista_candidatos = self._gerar_radar()

    def _calcular_matriz_distancias(self):
        n = self.n_cidades
        dist_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                xd = self.coordenadas[i][0] - self.coordenadas[j][0]
                yd = self.coordenadas[i][1] - self.coordenadas[j][1]
                rij = math.sqrt(xd*xd + yd*yd)
                dij = int(math.floor(rij + 0.5))
                dist_matrix[i][j] = dist_matrix[j][i] = dij
        return dist_matrix

    def _gerar_radar(self):
        return np.argsort(self.distancias, axis=1)[:, 1:self.k_vizinhos + 1]

    def vizinho_mais_proximo(self):
        n = self.n_cidades
        visitados = np.zeros(n, dtype=bool)
        rota = [0]
        visitados[0] = True
        dist_total = 0
        atual = 0

        for _ in range(n - 1):
            dist_atuais = self.distancias[atual].astype(float)
            dist_atuais[visitados] = np.inf
            proximo = np.argmin(dist_atuais)
            rota.append(proximo)
            visitados[proximo] = True
            dist_total += self.distancias[atual][proximo]
            atual = proximo

        dist_total += self.distancias[rota[-1]][rota[0]]
        return dist_total
