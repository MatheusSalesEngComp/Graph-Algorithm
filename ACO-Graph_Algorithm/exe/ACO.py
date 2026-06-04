import sys
import math
import time
import numpy as np
import csv



#SEMENTE = 42
#np.random.seed(SEMENTE)

# ==========================================
# 1. FUNÇÃO DE LEITURA DOS DADOS RANDOM FOREST
# ==========================================

def carregar_parametros_da_ia(n_cidades):
    caminho_tabela_ia = "parametros_gerais_ia.csv"
    
    alfa, beta, iteracoes = 1.0, 3.0, 50 
    
    with open(caminho_tabela_ia, mode='r', encoding='utf-8') as f:
        linhas = f.read().splitlines()
    

    for linha in linhas[1:]:
        if not linha.strip():
            continue
            
        coluna = linha.split(',') 
        
        if int(coluna[0]) == n_cidades:
            alfa = round(float(coluna[1]), 2)
            beta = round(float(coluna[2]), 2)
            iteracoes = int(coluna[3])
            break
            
    return alfa, beta, iteracoes        
# ==========================================
# 2. FUNÇÃO DE LEITURA DA ENTRADA PADRÃO (STDIN)
# ==========================================


def ler_instancia_real_da_entrada_padrao():
    linhas = sys.stdin.read().splitlines()
    
    dados = {
        'NAME': 'instancia_desconhecida',
        'DIMENSION': 0,
        'COORDS': []
    }
    
    escutando_coordenadas = False
    
    for linha in list(linhas):
        linha = linha.strip()
        if not linha or linha == "EOF":
            break
            
        if linha.startswith("NAME:"):
            dados['NAME'] = linha.split(":")[1].strip()
        elif linha.startswith("DIMENSION:"):
            dados['DIMENSION'] = int(linha.split(":")[1].strip())
        elif linha.startswith("NODE_COORD_SECTION"):
            escutando_coordenadas = True
            continue
            
        if escutando_coordenadas:
            partes = linha.split()
            if len(partes) == 3:
                x = float(partes[1])
                y = float(partes[2])
                dados['COORDS'].append([x, y])
                
    return dados


# ==========================================
# 3. CLASSES DO ALGORITMO (MUNDO E AGENTES)
# ==========================================
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


class ACOTrainer:
    def __init__(self, env, p_manager, n_formigas, n_iteracoes):
        self.env = env
        self.p_manager = p_manager
        self.n_formigas = n_formigas
        self.n_iteracoes = n_iteracoes
        self.melhor_distancia_global = float('inf')
        self.melhor_rota_global = None

    def treinar(self, alfa, beta):
        for i in range(self.n_iteracoes):
            formigas = [Ant(self.env.n_cidades, alfa, beta) for _ in range(self.n_formigas)]
            for formiga in formigas:
                formiga.percorrer_rota(self.env, self.p_manager.matriz_feromonio)

                if formiga.distancia_total < self.melhor_distancia_global:
                    self.melhor_distancia_global = formiga.distancia_total
                    self.melhor_rota_global = formiga.rota.copy()

            self.p_manager.evaporar()
            self.p_manager.depositar(formigas)


# ==========================================
# 4. FLUXO PRINCIPAL DE EXECUÇÃO (MAIN)
# ==========================================
if __name__ == "__main__":
    dados_instancia = ler_instancia_real_da_entrada_padrao()
    n_cidades_final = dados_instancia['DIMENSION']
    nome_instancia = dados_instancia['NAME']
    
    if n_cidades_final == 0:
        sys.exit("Erro: Instância vazia ou formato inválido.")


    alfa_exec, beta_exec, n_iteracoes_final = carregar_parametros_da_ia(n_cidades_final) 

    n_formigas_final = 30 if n_cidades_final > 200 else int(n_cidades_final * 0.5)

    env_final = TSPEnvironment(dados_instancia)
    p_man_final = PheromoneManager(n_cidades_final, rho=0.1, Q=env_final.vizinho_mais_proximo())
    treinador_final = ACOTrainer(env_final, p_man_final, n_formigas_final, n_iteracoes_final)

    treinador_final.treinar(alfa=alfa_exec, beta=beta_exec)

    
    print(f"Alfa:{alfa_exec}")
    print(f"Beta:{beta_exec}")
    print(f"Iterações:{n_iteracoes_final}")
    print()
    print(f"NAME: {nome_instancia}")
    print(f"COMMENT: Matheus, Dagoberto, Fernanda - Colonia de Formigas")
    print("TYPE: TOUR")
    print(f"DIMENSION: {n_cidades_final}")
    print(f"TOTAL WEIGHT: {treinador_final.melhor_distancia_global}")
    print("TOUR_SECTION")
    
    for cidade in treinador_final.melhor_rota_global:
        print(cidade + 1)
        
    print("EOF")