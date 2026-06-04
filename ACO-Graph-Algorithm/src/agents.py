import numpy as np

class Ant:
    def __init__(self, n_cidades, alfa, beta):
        self.n_cidades = n_cidades
        self.alfa = alfa
        self.beta = beta
        self.rota = []
        self.distancia_total = 0

    def escolher_proxima(self, atual, nao_visitadas, matriz_feromonio, env):
        usar_radar = self.n_cidades > 30
        
        if usar_radar:
            candidatos = [c for c in env.lista_candidatos[atual] if c in nao_visitadas]
            if not candidatos: 
                candidatos = list(nao_visitadas)
        else:
            candidatos = list(nao_visitadas)

        pesos = []
        for proxima in candidatos:
            f = matriz_feromonio[atual][proxima]
            d = 1.0 / (env.distancias[atual][proxima] + 1e-9)
            pesos.append((f ** self.alfa) * (d ** self.beta))

        soma = sum(pesos)
        probs = [p / soma for p in pesos]
        return np.random.choice(candidatos, p=probs)

    def percorrer_rota(self, env, matriz_feromonio):
        self.rota = [0]
        self.distancia_total = 0
        nao_visitadas = set(range(1, self.n_cidades))
        atual = 0

        while nao_visitadas:
            proxima = self.escolher_proxima(atual, nao_visitadas, matriz_feromonio, env)
            self.distancia_total += env.distancias[atual][proxima]
            self.rota.append(proxima)
            nao_visitadas.remove(proxima)
            atual = proxima
        self.distancia_total += env.distancias[self.rota[-1]][self.rota[0]]


class PheromoneManager:
    def __init__(self, n_cidades, rho, Q):
        self.n_cidades = n_cidades
        self.rho = rho
        self.Q = Q
        self.matriz_feromonio = np.ones((n_cidades, n_cidades))

    def evaporar(self):
        self.matriz_feromonio *= (1.0 - self.rho)

    def depositar(self, lista_formigas):
        for formiga in lista_formigas:
            deposito = self.Q / formiga.distancia_total
            rota_arr = np.array(formiga.rota)

            origens = rota_arr[:-1]
            destinos = rota_arr[1:]

            self.matriz_feromonio[origens, destinos] += deposito
            self.matriz_feromonio[destinos, origens] += deposito

            u, v = rota_arr[-1], rota_arr[0]
            self.matriz_feromonio[u, v] += deposito
            self.matriz_feromonio[v, u] += deposito

