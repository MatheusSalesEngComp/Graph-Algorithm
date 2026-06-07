import sys
import numpy as np
import time

from src.utils import ler_instancia_real_da_entrada_padrao, carregar_parametros_da_ia
from src.environment import TSPEnvironment
from src.agents import PheromoneManager
from src.trainer import ACOTrainer


SEMENTE = 42
np.random.seed(SEMENTE)

if __name__ == "__main__":
    dados_instancia = ler_instancia_real_da_entrada_padrao()
    n_cidades_final = dados_instancia['DIMENSION']
    nome_instancia = dados_instancia['NAME']
    
    if n_cidades_final == 0:
        sys.exit("Erro: Instância vazia ou formato inválido.")


    tempo_inicio = time.time()

    alfa_exec, beta_exec, n_iteracoes_final = carregar_parametros_da_ia(n_cidades_final) 

    n_formigas_final = 30 if n_cidades_final > 200 else int(n_cidades_final * 0.5)

    env_final = TSPEnvironment(dados_instancia)
    p_man_final = PheromoneManager(n_cidades_final, rho=0.1, Q=env_final.vizinho_mais_proximo())
    treinador_final = ACOTrainer(env_final, p_man_final, n_formigas_final, n_iteracoes_final)

    treinador_final.treinar(alfa=alfa_exec, beta=beta_exec)

    tempo_total = time.time() - tempo_inicio


    print(f"Alfa:{alfa_exec}")
    print(f"Beta:{beta_exec}")
    print(f"Iterações:{n_iteracoes_final}")
    print()
    print(f"NAME: {nome_instancia}")
    print(f"COMMENT: Matheus, Dagoberto, Fernanda - Colônia de Formigas")
    #print(f"COMMENT: {tempo_total / 60:.3f}m" if tempo_total > 60 else f"COMMENT: {tempo_total:.3f}s")
    print(f"COMMENT: {tempo_total:.3f}s")
    print("TYPE: TOUR")
    print(f"DIMENSION: {n_cidades_final}")
    print(f"TOTAL WEIGHT: {treinador_final.melhor_distancia_global}")
    print("TOUR_SECTION")    
    print(*(cidade + 1 for cidade in treinador_final.melhor_rota_global), sep='\n')
    print("EOF")