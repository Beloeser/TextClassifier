import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'nao\s+', 'nao_', text)
    text = re.sub(r'não\s+', 'nao_', text)
    text = re.sub(r'[^a-zA-ZÀ-ÿ\s_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class GeneticOptimizer:
    def __init__(self, X_train, y_train, population_size=20, generations=25):
        self.X_train = X_train
        self.y_train = y_train
        self.population_size = population_size
        self.generations = generations
        self.best_fitness_history = []

        #Ranges para a exploracao evolutiva

        max_df_values = [round(x, 2) for x in np.arange(0.2, 0.99, 0.03)]

        self.param_ranges = {
    'C': [0.5,0.7,1.0,1.5,2.0,3.0,4.0,5.0,7.0,10.0,12.0,15.0,20.0,25.0,30.0,40.0,50.0],
    'max_df': max_df_values,
    'min_df': [1, 2, 3, 4, 5, 6, 7],
    'ngram_max': [1, 2, 3, 4]
}




    def create_individual(self):
        return {
            'C': random.choice(self.param_ranges['C']),
            'max_df': random.choice(self.param_ranges['max_df']),
            'min_df': random.choice(self.param_ranges['min_df']),
            'ngram_max': random.choice(self.param_ranges['ngram_max'])
        }

    def fitness(self, individual):
        try:
            if individual['min_df'] >= len(self.X_train) * individual['max_df']:
                return 0.0

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_df=individual['max_df'],
                    min_df=individual['min_df'],
                    ngram_range=(1, individual['ngram_max']),
                    lowercase=True,
                    stop_words=None
                )),
                ('svm', SVC(kernel='linear', C=individual['C'], random_state=42))
            ])

            cv_folds = min(3, len(set(self.y_train)))
            scores = cross_val_score(pipeline, self.X_train, self.y_train,
                                   cv=cv_folds, scoring='f1_macro')
            return np.mean(scores)
        except Exception as e:
            return 0.0

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        selected = []
        for _ in range(self.population_size):
            candidates = random.sample(list(range(len(population))),
                                     min(tournament_size, len(population)))
            winner_idx = max(candidates, key=lambda i: fitnesses[i])
            selected.append(population[winner_idx].copy())
        return selected

    def crossover(self, parent1, parent2):
        child1, child2 = {}, {}

        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2

    def mutate(self, individual, mutation_rate=0.9):
        mutated = individual.copy()

        for key in mutated.keys():
            if random.random() < mutation_rate:
                if key == 'C':
                    current_idx = self.param_ranges[key].index(mutated[key])
                    new_idx = max(0, min(len(self.param_ranges[key])-1,
                                        current_idx + random.randint(-2, 2)))
                    mutated[key] = self.param_ranges[key][new_idx]
                else:
                    mutated[key] = random.choice(self.param_ranges[key])

        return mutated

    def optimize(self):
        population = [self.create_individual() for _ in range(self.population_size)]

        print("Iniciando Algoritmo Genético...")
        print(f"População: {self.population_size}, Gerações: {self.generations}")

        for generation in range(self.generations):
            fitnesses = [self.fitness(individual) for individual in population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            self.best_fitness_history.append(best_fitness)

            print(f"Geração {generation:2d}: Melhor={best_fitness:.4f}, Média={avg_fitness:.4f}")

            if best_fitness > 0.95:
                print("Convergência atingida!")
                break

            selected = self.tournament_selection(population, fitnesses)
            new_population = []

            elite_indices = np.argsort(fitnesses)[-2:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            while len(new_population) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)

                if random.random() < 0.8:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        final_fitnesses = [self.fitness(ind) for ind in population]
        best_individual = population[np.argmax(final_fitnesses)]

        print(f"\nOtimização concluída!")
        print(f"Melhor F1-score: {max(final_fitnesses):.4f}")
        print(f"Evolução: {self.best_fitness_history[0]:.4f} → {self.best_fitness_history[-1]:.4f}")

        return best_individual
    

    
# Dataset
positivos = [
    "bem completo", "muito intuitivo", "gostei da atualizacao",
    "navegacao facil", "app muito bom", "design simples e bonito", "uso todo dia",
    "bem util", "otimo desempenho", "surpreendeu", "carrega rapido",
    "bem satisfatorio", "muito pratico", "me ajudou bastante", "app excelente mesmo",
    "bem construído", "funciona redondo", "resultados precisos", "simples e bom",
    "melhor que eu esperava", "rapido e leve", "gostei demais", "bem interessante",
    "otimo app", "achei excelente", "gostei muito do app", "funciona direitinho",
    "excelente atualizacao", "design muito bonito", "muito bom o atendimento", "nao trava, gostei",
    "bem top", "achei bem interessante", "funciona perfeitamente", "curti o design",
    "otimo atendimento", "bem util mesmo", "experiencia muito boa", "top o app",
    "funciona certinho", "bem intuitivo", "bem rapidinho", "gostei bastante do layout",
    "muito bom de verdade", "bem completo mesmo", "nao trava nunca", "app excelente demais",
    "gostei da experiencia","Funciona super bem, show!", "bem satisfatorio mesmo", "otima performance", "fantastico app",
    "perfeito funcionamento", "interface linda", "muito responsivo", "adorei as funcionalidades",
    "app incrivel", "design moderno", "facil de usar", "muito estavel",
    "carregamento instantaneo", "funcoes uteis", "bem organizado", "interface amigavel",
    "amei o app","Aplicativo show, recomendo para todos" "amei a interface", "amo a ideia", "amo tudo isso", "superou minhas expectativas", "muito eficiente", "ótimo suporte",
    "app confiável", "interface intuitiva", "uso fácil e agradável",
    "recomendo para todos", "ótima experiência geral", "funcionamento impecável",
    "fácil de navegar", "ótima atualização", "muito prático", "design elegante",
    "performance excelente", "responde rápido", "ótima funcionalidade",
    "extremamente útil", "gosto muito do app", "atendimento excepcional","Interface show, fácil de usar",
     "simples e funcional", "ótimo para o dia a dia",
    "tudo perfeito", "adorei o layout", "ótima estabilidade", "ótima organização",
    "super intuitivo", "muito bem desenvolvido", "experiência incrível", "ótimo app", "muito útil", "super prático", "facil de usar", "interface agradável",
    "gostei bastante", "ótima experiência", "funciona perfeitamente", "app muito bom",
    "simples e eficiente", "recomendo fortemente", "resolutivo", "curti muito", "design bonito","Design show, moderno e bonito",
    "navegação fácil", "bem organizado", "excelente atendimento", "rápido e leve",
    "ótimo desempenho", "adorei o app", "fácil de entender", "interface limpa", "ótima funcionalidade",
    "ótima atualização", "uso diário", "bom demais", "top app", "muito intuitivo",
    "ótima performance", "app muito estável","Performance show, rápido e eficiente", "funciona sem problemas", "gostei da interface",
    "bom design", "app incrível", "experiência agradável", "ótima ideia", "top demais",
    "gostei do layout", "muito responsivo", "ótima organização", "simples de usar",
    "excelente interface", "bem feito", "recomendo o app", "funcional e bonito",
    "ótima experiência de usuário", "muito bem construído", "excelente funcionalidade",
    "curti o design", "interface moderna", "ótimo funcionamento", "uso sem complicações",
    "muito prático e rápido", "ótimo app para o dia a dia", "bom atendimento", "excelente performance",
    "muito intuitivo e simples", "ótima estabilidade", "interface amigável", "gostei do desempenho",
    "app de qualidade", "bem útil e fácil", "ótima navegação", "excelente experiência geral",
    "funciona redondo", "ótima solução", "simples e eficiente", "curti bastante o app",
    "bem satisfatório", "muito confiável", "ótimo para todos", "uso sem erros",
    "excelente visual", "ótima velocidade", "design elegante", "ótima funcionalidade geral",
    "adorei as funcionalidades", "muito prático e intuitivo", "app top", "uso muito fácil",
    "ótimo para começar", "interface clara", "muito bom de usar", "experiência completa",
    "app super útil", "funciona perfeitamente", "ótima experiência", "adorei cada detalhe",
    "muito intuitivo", "excelente design", "funciona muito bem", "ótima performance",
    "curti bastante", "recomendo para todos", "interface linda", "ótimo atendimento",
    "resolveu meu problema", "simples e eficaz", "muito estável", "uso todos os dias",
    "app muito prático", "fantástico", "surpreendeu positivamente", "bem rápido",
    "bem pensado", "excelente atualização", "muito bom mesmo", "design moderno e bonito",
    "app confiável", "funciona redondo", "ótimo desempenho", "muito fácil de usar",
    "interface amigável", "ótima navegação", "atendimento excelente", "curti a experiência",
    "bem completo", "muito intuitivo e rápido", "app leve e funcional", "excelente utilidade",
    "funciona sem problemas", "ótima experiência de uso", "app eficiente", "design impecável",
    "adorei a atualização", "bem organizado", "ótimo layout", "funcional e prático",
    "muito satisfatório", "app incrível", "facilita muito a vida", "bem estruturado",
    "excelente para o dia a dia", "curti demais", "ótima ferramenta" , "app super rápido",
    "ótima interface", "funciona perfeitamente bem", "muito intuitivo e fácil",
    "curti o design", "excelente experiência de uso", "app confiável", "muito prático no dia a dia",
    "ótima atualização", "funcional e eficiente", "app leve", "interface bonita",
    "ótima ferramenta", "muito satisfeito", "uso fácil e rápido", "responde bem",
    "app muito útil", "interface clara", "ótima performance", "app bem construído",
    "curti cada detalhe", "funciona redondinho", "adorei o layout", "ótimo funcionamento",
    "app estável", "fácil de navegar", "ótima usabilidade", "funcionalidade excelente",
    "ótimo design", "uso agradável", "muito bom de usar", "interface moderna",
    "funciona sem problemas", "app leve e rápido", "curti bastante a atualização",
    "muito intuitivo e rápido", "resolutivo", "bem prático", "ótimo para o dia a dia",
    "excelente utilidade", "interface organizada", "uso tranquilo", "ótima experiência geral",
    "app super estável", "uso eficiente", "ótimo layout e design", "app bem estruturado",
    "funciona perfeitamente", "ótima ferramenta para mim", "interface bem pensada", "excelente app", "super recomendo",
    "excelente funcionalidade", "adoro usar", "muito intuitivo",
    "melhor app que já usei", "ótima experiência", "satisfeito com o app", "funciona perfeitamente",
    "interface incrível", "ótimo suporte", "muito rápido", "bem prático", "adorei o design",
    "ótimo desempenho", "muito útil", "simplicidade incrível", "excelente ideia",
    "gostei demais","Show de design, moderno e agradável",
    "Show de qualidade, nada a reclamar",
    "Show de organização, tudo intuitivo","Show de eficiência, executa rápido", "surpreendente",
    "ótimo atendimento", "perfeito funcionamento",
    "app confiável", "muito eficiente", "bem organizado", "excelente app",
    "gosto muito dessa ferramenta", "ótimo para meu dia a dia", "fantástico", "excelente suporte",
    "muito intuitivo e prático", "app impecável", "funciona redondo", "design bonito e limpo",
    "ótima experiência do usuário", "aprovado", "muito satisfeito", "app confiável e rápido",
    "excelente atualização", "melhor que esperava", "app muito bom mesmo", "bem construído",
    "ótimo para minhas necessidades", "curti muito", "app leve e rápido", "simples e funcional",
    "ótima performance", "excelente funcionalidade", "muito intuitivo", "app incrível",
    "funciona muito bem", "gosto bastante", "excelente app", "muito prático", "bem útil",
    "ótima interface", "app confiável", "excelente experiência", "fácil de usar",
    "ótima usabilidade", "app completo", "muito bom mesmo", "adorei usar", "app satisfatório",
    "ótima navegação", "super fácil de usar", "muito bem feito", "app funcional",
    "excelente design", "gostei bastante", "ótima performance", "muito estável",
    "app leve e eficiente", "ótimo aplicativo", "excelente funcionalidade", "bem planejado",
    "adorei as funcionalidades", "ótimo desempenho", "excelente suporte", "muito bom o atendimento",
    "app simples e eficiente", "ótima experiência do usuário", "fácil e rápido", "muito útil",
    "ótimo design", "excelente app para mim", "funciona perfeitamente", "app de qualidade",
    "ótima performance", "excelente interface", "muito satisfatório", "gostei muito",
    "ótimo para o dia a dia", "app confiável e rápido", "bem intuitivo", "excelente para meu uso",
    "ótima experiência geral", "muito bom mesmo", "app fácil e funcional", "ótimo atendimento",
    "app super rápido", "ótima experiência", "funciona perfeitamente", "design agradável",
    "recomendo bastante", "bem intuitivo", "simples e eficiente", "atendimento excelente",
    "muito útil", "super fácil de usar", "adorei a navegação", "ótima performance",
    "interface limpa", "bem construído", "resolve tudo rapidamente", "app confiável",
    "fantástico serviço","Show de estabilidade, nenhum bug até agora",
    "Show de performance, rápido e confiável",
    "Show de confiabilidade, nunca travou",
    "Show de recursos, muitas opções úteis", "ótima funcionalidade", "fácil de entender", "ótimo para o dia a dia",
    "muito prático", "ótima atualização", "super recomendado", "adorei o app",
    "design moderno", "muito responsivo", "interface amigável", "excelente suporte",
    "uso diário sem problemas", "bem organizado", "aplicativo incrível", "adorei as funções",
    "funciona redondo", "interface intuitiva", "ótima performance geral", "muito eficiente",
    "atende perfeitamente", "top demais", "excelente app", "muito bom mesmo"
]

neutros = [
    "poderia ser melhor otimizado",
    "app mediano", "podia ter mais funcoes", "bem mais ou menos", "gostei mas podia melhorar",
    "nao achei grande coisa", "poderia funcionar melhor", "nao gostei muito nao", "achei ok",
    "da pro gasto", "cumpre o basico", "nada de especial", "comum",
    "mediano mesmo", "nem bom nem ruim", "razoavel", "aceitavel",
    "dentro do esperado", "normal", "padrao", "basico",
    "simples demais", "falta algo", "poderia ter mais", "meio limitado",
    "funciona mas nada demais", "serve", "da pra usar", "nao impressiona",
    "comum demais", "sem grandes novidades", "basico demais", "meio simples","nem ótimo nem ruim", "regular", "aceitável pelo menos", "ok para uso",
    "nada de especial", "funciona mas poderia melhorar", "mais ou menos satisfatório",
    "uso básico", "mediano para mim", "nada impressionante", "cumpre a função",
    "ok, sem surpresas", "regularzinho", "nem bom nem ruim para mim", "normalzinho",
    "uso simples", "satisfatório sem mais", "ok, mas nada demais", "cumpre o básico",
    "funciona, mas sem destaque", "razoável para o dia a dia", "mais ou menos funcional",
    "não decepciona, mas não impressiona", "uso tranquilo", "regular, sem falhas",
    "ok, nada além disso", "cumpre o que promete, nada mais", "mediano, dá pro gasto",
    "satisfatório mas sem charme", "uso aceitável", "normal, sem problemas",
    "funciona dentro do esperado", "ok, mas poderia ser melhor", "apenas funcional",
    "nenhum destaque", "ok, mas sem entusiasmo", "uso neutro", "cumpre sua função",
    "razoável, nada especial", "ok para o básico", "uso simples e direto", "nada a reclamar mas nada a elogiar", "ok, nada especial", "mais ou menos satisfatório", "funciona, mas simples", "podia ser melhor",
    "normalzinho", "nada de diferente", "funciona, mas sem graça", "pode melhorar",
    "cumpre o básico", "não é ruim, mas não impressiona", "mediano", "razoável",
    "simples de usar", "interface comum", "esperava mais", "bom, mas simples", "uso diário tranquilo",
    "funciona como esperado", "nenhum problema, nenhuma surpresa", "adequado", "ok, serve",
    "meio limitado", "nem ótimo nem ruim", "aceitável", "básico demais", "sem grandes novidades",
    "não atrapalha", "não me incomodou", "funciona, mas sem destaque", "comum e simples",
    "regular", "satisfatório", "não impressiona muito", "poderia ter mais recursos",
    "nada de especial", "apenas funcional", "serve para o que precisa", "meio básico",
    "não tem problemas", "funciona, mas sem charme", "ok para uso", "não muda nada",
    "normal e previsível", "poderia ser mais intuitivo", "neutro", "satisfatório sem mais",
    "uso simples", "interface padrão", "funciona corretamente", "não é ruim, nem excelente",
    "apenas razoável", "pouco atrativo", "nada demais", "cumpre o que promete",
    "uso ok", "sem surpresas", "intermediário", "nem impressiona, nem decepciona",
    "funciona, mas simples demais", "razoável para o básico", "meio monótono", "adequado, mas simples",
    "não é ruim, mas poderia melhorar", "pouco interessante", "uso básico tranquilo", "sem grandes falhas",
    "funciona normal", "interface simples e comum", "ok, nada de errado", "apenas básico",
    "nem bom, nem ruim", "poderia ser mais agradável", "uso neutro", "cumpre sem exageros",
    "simples e direto", "uso básico", "interface comum e funcional", "razoável e simples",
    "funciona corretamente, mas sem graça", "nenhum destaque", "ok para uso diário", "sem grandes emoções",
    "uso neutro, sem problemas", "interface clara, mas básica", "nada de surpreendente", "apenas funcional", "ok, cumpre o básico", "mais ou menos", "aceitável", "não é ruim nem ótimo",
    "funciona mas simples", "poderia melhorar", "mediano", "não impressiona",
    "uso comum", "interface padrão", "razoável", "não me incomodou nem agradou",
    "cumpre o que promete", "bem regular", "uso diário ok", "nada de especial",
    "normalzinho", "funciona sem grandes novidades", "serve para o básico", "nem ruim nem bom",
    "simples demais", "pouco impressionante", "ok para uso casual", "cumpre a função",
    "regular", "bem mais ou menos", "não compromete", "funciona mas sem charme", "meio limitado",
    "uso aceitável", "funciona razoavelmente", "poderia ser mais intuitivo", "nada fora do comum",
    "simples e básico", "ok para começar", "uso tranquilo", "não é ruim, não é ótimo",
    "funciona, mas nada demais", "regular mas útil", "cumpre o esperado", "não me decepcionou",
    "interface básica", "uso prático mas simples", "não impressionou", "razoável para todos",
    "cumpre seu papel", "ok para uso cotidiano", "regular de forma geral", "não chama atenção",
    "funciona, mas sem diferencial", "ok, sem problemas" , "uso ok", "cumpre o básico", "mais ou menos", "não é ruim nem ótimo",
    "regular", "funciona sem grandes novidades", "mediano", "ok para uso casual",
    "uso diário normal", "interface simples", "nada de especial", "cumpre o esperado",
    "não impressiona", "ok, sem problemas", "uso tranquilo mas simples", "regular mas funcional",
    "funciona, nada mais", "simples e básico", "uso sem surpresas", "regularzinho",
    "interface padrão", "cumpre o prometido", "ok para começar", "uso razoável",
    "neutro", "não me incomodou", "uso sem frustração", "nada fora do comum", "ok para todos",
    "regular e simples", "cumpre sua função", "interface comum", "funciona razoavelmente",
    "uso médio", "regular de forma geral", "não chama atenção", "uso simples e tranquilo",
    "cumpre seu papel", "ok, sem grandes problemas", "funciona, mas sem charme", "uso padrão",
    "nada especial", "ok para uso cotidiano", "uso sem destaque", "cumpre o necessário",
    "interface neutra", "regular para todos", "ok, funcional", "uso sem impacto",
    "cumpre o básico esperado", "nada impressionante", "mais ou menos", "razoável", "ok, sem novidade", "funciona, mas nada demais",
    "nada impressionante", "normal", "dentro do esperado", "podia melhorar",
    "não é ruim", "médio", "aceitável", "ok", "ok, mas podia melhorar", "mais ou menos útil",
    "serve para o que precisa", "cumpre o básico", "funciona, mas lento", "não é ruim nem ótimo",
    "meio limitado", "mais ou menos satisfatório", "padrão", "nenhum problema",
    "funciona razoavelmente", "ok, mas poderia ser melhor", "meio básico", "simples",
    "normalzinho", "nada de especial", "razoavelmente útil", "poderia ter mais funções",
    "cumpre sua função", "ok para uso diário", "não impressiona", "funciona mas nada demais",
    "meio fraco", "ok, mas falta algo", "mais ou menos eficiente", "cumpre o que promete",
    "ok, razoável", "funciona mediano", "não é ruim, nem ótimo", "meio regular",
    "bastante simples", "ok, nada incrível", "poderia ser mais útil", "serve bem",
    "aceitável para o uso", "ok, mas poderia melhorar", "mediano", "cumpre o esperado",
    "funciona básico", "meio limitado", "razoável para o que precisa", "nenhuma surpresa",
    "pode ser melhor", "ok, nada de especial", "mais ou menos funcional", "cumpre",
    "simples mas útil", "funciona bem, mas nada demais", "ok, dentro do padrão",
    "regular", "meio básico", "serve para tarefas simples", "funciona normal",
    "ok, sem problemas", "razoável, mas poderia melhorar", "aceitável, nada de especial",
    "funciona, mas limitações", "ok, mas básico", "meio fraco", "cumpre sua função",
    "ok, sem erros", "razoável, atende ao esperado", "mediano, nada impressionante",
    "ok, mas poderia ser melhor", "normal, nada demais",
]

negativos = [
    "nao atende as expectativas", "experiencia frustrante", "nao recomendo mesmo", "travou bastante",
    "nao funciona como deveria", "achei confuso", "nao abre de primeira", "achei fraco",
    "odiei a interface", "bateria vai embora rapido", "nao gostei do layout", "tem muitos bugs",
    "nao gostei das cores", "falta recursos", "decepcionou", "nao e intuitivo",
    "nao e tudo isso", "bem ruimzinho", "precisa melhorar muito", "nao entrega o que promete",
    "muito instavel", "nao presta", "nao gostei da experiencia", "app muito ruim",
    "lento demais", "horrivel de usar", "app instavel", "nao recomendo o app",
    "bem lento", "nao vale o download", "achei meio fraco", "experiencia muito ruim",
    "nao gostei do desempenho", "demora muito", "mal otimizado demais", "nao entrega nada",
    "problemas demais", "nao gostei do atendimento", "bem decepcionante", "odiei o desempenho",
    "app horrivel", "muito ruim mesmo", "pessimo app", "nao funciona nunca",
    "sempre trava", "muito bugado", "interface terrivel", "design horrivel",
    "muito confuso", "nao entendo nada", "muito complicado", "interface feia",
    "design feio", "muito desorganizado", "bagunca total", "nao faz sentido",
    "muito mal feito", "parece amador", "muito primitivo", "design antigo", "não gostei muito", "poderia ser melhor", "esperava mais do app", "uso difícil às vezes",
    "funciona, mas com erros", "interface confusa em alguns pontos", "não tão intuitivo",
    "demora para carregar algumas telas", "uso pouco prático", "razoável mas simples",
    "não atende totalmente", "problemas pequenos de performance", "funcional, mas limitado",
    "precisa de ajustes", "não é muito confiável", "alguns bugs persistem", "uso pouco agradável",
    "não cumpre todas as expectativas", "demora em certas funções", "pouco eficiente",
    "não entrega tanto quanto promete", "simples demais", "uso mediano", "pode melhorar bastante",
    "não impressiona", "cumpre, mas sem destaque", "uso normal", "razoável para tarefas básicas",
    "uso limitado", "pouco prático", "não tão responsivo", "alguns contratempos",
    "funciona, mas não é ótimo", "precisa de otimizações", "demora em funções simples",
    "não é excelente", "uso apenas suficiente", "pouco intuitivo", "funciona mas limitado",
    "não cumpre tudo", "uso básico e limitado",
    "péssima experiência", "muito lento", "travou várias vezes", "não funciona direito",
    "interface horrível", "problemas constantes", "muito bugado", "demora demais",
    "frustrante de usar", "ruim de verdade", "não recomendo", "não atende nada",
    "muito confuso", "uso complicado", "mal otimizado", "travando sempre", "pouco confiável",
    "causa problemas", "instável", "não cumpre o que promete", "desempenho ruim",
    "horrível", "muito decepcionante", "não vale o download", "ruim mesmo", "não funciona nunca",
    "desorganizado", "bagunçado", "interface feia", "muito mal feito", "quase inútil",
    "não entrega nada", "problemas demais", "pessimo desempenho", "muito instável",
    "lento demais", "péssima usabilidade", "odiei usar", "não presta", "muito ruim mesmo",
    "não recomendo de jeito nenhum", "travou bastante", "totalmente frustrante",
    "demora para tudo", "nunca funciona direito", "ruim de usar", "interface confusa demais",
    "não vale nada", "uso terrível", "não funciona como deveria", "ruim demais para o dia a dia",
    "pior app que já usei", "problemas graves", "não dá para usar", "não entrega resultados",
    "extremamente instável", "demora excessiva", "mal projetado", "uso péssimo", "confuso e lento",
    "insuportável às vezes", "experiência ruim demais", "não recomendo para ninguém", "pior experiência possível", "não gostei muito", "funciona mal às vezes", "bem lento", "interface confusa", "pouco intuitivo",
    "travou no meio do uso", "não atendeu minhas expectativas", "poderia ser melhor", "não recomendo",
    "muito instável", "erros frequentes", "demora para abrir", "não funciona direito", "ruim mesmo",
    "mais fraco do que esperava", "precisa melhorar muito", "não entrega o que promete", "bugs constantes",
    "uso complicado", "decepcionante", "não vale o esforço", "muito básico", "não gostei da atualização",
    "mais lento que outros apps", "interface ruim", "difícil de usar", "não é confiável", "mal feito",
    "precisa de ajustes", "muito instável ainda", "trava direto", "não funciona como deveria", "ruimzinho",
    "não cumpre o esperado", "mais decepcionante do que útil", "demora demais para carregar", "muito confuso",
    "não atende o que promete", "funciona, mas dá problema", "não recomendo para uso diário", "interface fraca",
    "muito básico para o preço", "uso difícil", "erros inesperados", "não gostei da experiência", "problemas frequentes",
    "mais fraco que outros", "muito bugado", "travou várias vezes", "difícil de entender", "não intuitivo",
    "mal projetado", "pouco funcional", "não me agradou", "instável demais", "ruim e confuso", "não vale a pena",
    "uso problemático", "decepcionou bastante", "não gostei da interface", "mais lento do que deveria", "muito fraco",
    "não cumpre o que promete", "interface complicada", "trava muito", "não atende minhas necessidades",
    "funciona, mas irrita", "muito ruim para uso constante", "não gostei do layout", "precisa de melhorias urgentes",
    "mais problemas do que soluções", "não é confiável", "muito confuso e lento", "experiência ruim", "ruim demais",
    "não entrega resultados", "mal otimizado", "interface feia", "muito instável e confuso", "uso frustrante",
    "não recomendo o app", "demora, trava e confunde", "não funciona direito na prática", "pessimo desempenho", "muito ruim", "não funciona direito", "travou várias vezes", "interface confusa",
    "uso frustrante", "não cumpre o prometido", "demora demais", "experiência decepcionante",
    "mais problemas do que soluções", "pouco intuitivo", "mal otimizado", "muito lento",
    "uso irritante", "não gostei da interface", "erro constante", "instável",
    "não recomendo", "decepcionante", "ruim demais", "muito básico", "funciona mal",
    "não entrega resultados", "difícil de usar", "uso complicado", "interface ruim",
    "não atende minhas necessidades", "muito confuso", "travou direto", "uso problemático",
    "mais fraco do que esperava", "não vale a pena", "péssimo desempenho", "mal feito",
    "não confiável", "não gostei do layout", "precisa melhorar muito", "ruim e lento",
    "experiência ruim", "não cumpre expectativas", "uso frustrante e lento", "funciona mal às vezes",
    "interface pouco amigável", "muito instável e confuso", "erro inesperado", "uso difícil",
    "decepcionou bastante", "não atende o prometido", "mal projetado", "difícil de entender",
    "interface complicada", "não cumpre o esperado", "app travando sempre", "muito bugado", "não funciona direito", "péssima experiência",
    "ruim e lento", "uso frustrante", "interface confusa e ruim", "não atende expectativas",
    "mais problemas que soluções", "erro constante", "instável", "não recomendo para ninguém",
    "pessima performance", "difícil de usar", "mal otimizado", "demora muito para carregar",
    "não gostei do layout", "uso problemático", "interface pouco amigável", "muito instável",
    "erro inesperado", "difícil de navegar", "mal feito", "ruim de usar", "uso complicado",
    "não cumpre o prometido", "não funciona como deveria", "travou direto", "não é confiável",
    "pouco intuitivo", "uso irritante", "experiência decepcionante", "interface ruim e confusa",
    "erro frequente", "não entrega resultados", "difícil de entender", "não vale a pena",
    "ruim demais", "uso frustrante e lento", "app instável e pesado", "interface complicada",
    "não cumpre expectativas", "muito confuso", "uso problemático e lento", "travando constantemente",
    "não entrega o prometido", "experiência negativa", "interface feia", "uso difícil e confuso",
    "péssima usabilidade", "não recomendo a ninguém", "não gostei", "muito lento", "travou várias vezes", "péssimo atendimento",
    "não funciona direito", "interface ruim", "decepcionante", "bugado",
    "falha constante", "não atende às expectativas", "app horrível", "frustrante",
    "pouco intuitivo", "muito ruim", "trava sempre", "mal feito", "complicado de usar",
    "falha de desempenho", "app instável", "difícil de navegar", "demora demais",
    "mal otimizado", "não funciona como esperado", "ruim demais", "problemas constantes",
    "confuso", "travou demais", "não recomendaria", "mal projetado", "app péssimo",
    "interface feia", "muito lento para abrir", "não funciona", "demora pra carregar",
    "ruim para usar", "não cumpre o que promete", "desagradável", "muito fraco",
    "deixa a desejar", "não vale a pena", "muito lento", "travou várias vezes", "não funciona direito", "péssimo atendimento",
    "interface confusa", "app horrível", "demora para carregar", "difícil de usar",
    "muito bugado", "experiência frustrante", "não gostei do design", "mal otimizado",
    "não entrega o que promete", "pior que eu esperava", "horrível de usar", "ruim demais",
    "não atende as expectativas", "problemas frequentes", "travou no meio do uso", "demora demais",
    "interface feia", "não recomendo", "muito confuso", "bateria acaba rápido",
    "precisa melhorar muito", "mal feito", "não gostei do layout", "lento demais",
    "funciona quando quer", "não é intuitivo", "ruim para navegar", "falha constante",
    "mais fraco que o esperado", "péssima usabilidade", "muito instável", "decepcionante",
    "problemas demais", "não abre direito", "mal construído", "app ruim"
]


comentarios = positivos + neutros + negativos
labels_num = [1] * len(positivos) + [0] * len(neutros) + [-1] * len(negativos)
labels = ['positivo' if l==1 else 'neutro' if l==0 else 'negativo' for l in labels_num]

def prepare_data():
    print(f"Dataset: {len(comentarios)} comentários")
    print(f"Positivos: {len(positivos)}, Neutros: {len(neutros)}, Negativos: {len(negativos)}")

    comentarios_processados = [preprocess_text(c) for c in comentarios]

    X_train, X_test, y_train, y_test = train_test_split(
        comentarios_processados, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Treinamento: {len(X_train)}, Teste: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    optimizer = GeneticOptimizer(X_train, y_train, population_size=10, generations=15)
    best_params = optimizer.optimize()

    print("\nTreinando modelo final...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_df=best_params['max_df'],
            min_df=best_params['min_df'],
            ngram_range=(1, best_params['ngram_max']),
            lowercase=True
        )),
        ('svm', SVC(kernel='linear', C=best_params['C'], random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, best_params

def evaluate_model(pipeline, X_test, y_test, best_params):
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*70)
    print("RESULTADOS FINAIS")
    print("="*70)
    print(f"Acurácia: {accuracy:.4f}")
    print(f"F1-Score: {f1_macro:.4f}")
    print(f"Parâmetros otimizados: {best_params}")
    print(f"\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

def predict_sentiment(pipeline, text):
    processed = preprocess_text(text)
    return pipeline.predict([processed])[0]


def interactive_test(pipeline):
    text = input("Digite um comentário: ").strip()
    try:
        print("Classificação:", predict_sentiment(pipeline, text).upper())
    except Exception as e:
        print("Erro:", e)


def main():
    global pipeline, best_params # Declara como global

    print("CLASSIFICADOR COM ALGORITMO GENÉTICO ")

    X_train, X_test, y_train, y_test = prepare_data()
    pipeline, best_params = train_model(X_train, y_train)
    evaluate_model(pipeline, X_test, y_test, best_params)


if __name__ == "__main__":
    main()

interactive_test(pipeline)

    # Comentários de teste com rótulos esperados
comentarios_teste = [
    ("muito bom, gostei bastante", "positivo"),
    ("funciona bem", "positivo"),
    ("mais ou menos", "neutro"),
    ("nao gostei", "negativo"),
    ("app excelente", "positivo"),
    ("demora pra carregar", "negativo"),
    ("ok, nada demais", "neutro"),
    ("odiei a experiencia", "negativo"),
    ("bom, mas podia melhorar", "neutro"),
    ("pessimo atendimento", "negativo"),
    ("gostei do design", "positivo"),
    ("trava as vezes", "negativo"),
    ("cumpre o que promete", "positivo"),
    ("nao e ruim", "neutro"),
    ("muito lento", "negativo"),
    ("adorei o app", "positivo"),
    ("experiencia razoavel", "neutro"),
    ("horrivel, nao recomendo", "negativo"),
    ("bom demais", "positivo"),
    ("deixa a desejar", "negativo"),
    ("top demais", "positivo"),
    ("funciona quando quer", "negativo"),
    ("excelente servico", "positivo"),
    ("nada a reclamar", "positivo"),
    ("interface confusa", "negativo"),
    ("nao funciona direito", "negativo"),
    ("legalzinho", "neutro"),
    ("bem ruim", "negativo"),
    ("app ok", "neutro"),
    ("muito bom mesmo", "positivo"),
    ("esperava mais", "neutro"),
    ("otima ideia", "positivo"),
    ("mal otimizado", "negativo"),
    ("curti bastante", "positivo"),
    ("problemas frequentes", "negativo"),
    ("simples e eficiente", "positivo"),
    ("nao curti", "negativo"),
    ("muito fraco", "negativo"),
    ("recomendo", "positivo"),
    ("mais lento que o esperado", "negativo"),
    ("boa experiencia", "positivo"),
    ("pessima usabilidade", "negativo"),
    ("resolve o problema", "positivo"),
    ("decepcionante", "negativo"),
    ("show de bola", "positivo"),
    ("quebra um galho", "neutro"),
    ("ruim demais", "negativo"),
    ("excelente, parabens", "positivo"),
    ("nao vale a pena", "negativo"),
    ("funciona normal", "neutro")
]

# Contadores
acertos = {"positivo": 0, "neutro": 0, "negativo": 0}
total = {"positivo": 0, "neutro": 0, "negativo": 0}

print("="*70)
print("RESULTADOS DETALHADOS DE CADA COMENTÁRIO")
print("="*70)

# Testae
for texto, rotulo in comentarios_teste:
    pred = predict_sentiment(pipeline, texto)
    total[rotulo] += 1
    correto = pred == rotulo
    if correto:
        acertos[rotulo] += 1
    print(f"Comentário: '{texto}'")
    print(f"Esperado: {rotulo} | Predito: {pred} | {'ACERTOU' if correto else 'ERROU'}\n")

# Resultados resumidos
print("="*70)
print("RESUMO POR CATEGORIA")
print("="*70)
for cat in ["positivo", "neutro", "negativo"]:
    print(f"{cat.capitalize()}: {acertos[cat]}/{total[cat]} corretos | "
          f"Acurácia: {acertos[cat]/total[cat]:.2%}")

acuracia_geral = sum(acertos.values()) / sum(total.values())
print("\nAcurácia geral:", f"{acuracia_geral:.2%}")
