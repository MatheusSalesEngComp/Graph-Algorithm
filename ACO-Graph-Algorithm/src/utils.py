import sys
import os

# ==========================================
# 1. FUNÇÃO DE LEITURA DOS DADOS RANDOM FOREST
# ==========================================

def carregar_parametros_da_ia(n_cidades):
    caminho_tabela_ia = os.path.join("data", "processed", "parametros_gerais_ia.csv")
    
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
