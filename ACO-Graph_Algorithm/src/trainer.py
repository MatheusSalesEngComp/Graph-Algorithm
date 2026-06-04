from src.agents import Ant

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

