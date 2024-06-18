
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib.animation import FuncAnimation
import random, numpy
import arch
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
F = torch.nn.functional

seed = 1234
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

space_func = {
	0: ('int', '1'),
	1: ('int', 'torch.e'),
	2: ('int', 'torch.pi'),
	3: ('int', '0.5'),
	4: ('func', 'x'),
	5: ('func', 'F.sigmoid(x)'),
	6: ('func', 'F.tanh(x)'),
	7: ('func', 'F.silu(x)'),
	8: ('func', 'F.gelu(x)'),
	9: ('func', 'F.tanh(x * 1.46)'),
}

space_ops = {
	0: ('op', '+'),
	1: ('op', '*'),
	2: ('op', '/'),
}
consider_x = [4, 5, 6, 7, 8, 9]

def create_function(funcs, ops):
	txt = f'lambda x: {funcs[0][1]} {ops[0][1]} {funcs[1][1]} {ops[1][1]} {funcs[2][1]}'
	return eval(txt), txt


def feedback(selected_features, get_txt=False):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	t = torch.tensor(selected_features[:27]).view(3, 9).sum(dim=1)
	o = torch.tensor(selected_features[27:]).view(2, 2).sum(dim=1)
	if all([i not in consider_x for i in t]):
		return 0

	funcs = [space_func[x.item()] for x in t]
	ops = [space_ops[x.item()] for x in o]
	act, txt = create_function(funcs, ops)
	if get_txt:
		return arch.train(act), txt
	# print(txt)
	return 1000 - arch.train(act)


# o = feedback([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# 			  1, 0, 0, 0, 0, 1, 0, 0, 1, 1,
# 			  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
# 			  0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
# 			  0, 0, 1, 0, 0, 0, 0, 0, 0])
# print(o)
# exit()



def pop_create(**args):
	zl = torch.zeros(49)
	zl[torch.cat([torch.randint(10, (1,)) + (10 * x) for x in range(4)])] = 1
	return zl.tolist()
# zl[]


def test_score(selected_features):
	return feedback(selected_features),

NUM_OF_FEATURES = 31


POPULATION_SIZE = 30
P_CROSSOVER = 0.7
P_MUTATION = 0.3
MAX_GENERATIONS = 30
HALL_OF_FAME_SIZE = 8

toolbox = base.Toolbox()

# Trying to maximize the accuracy; that's why 1
creator.create('fitness_max', base.Fitness, weights=(1.0,))

creator.create('individual', list, fitness=creator.fitness_max)
toolbox.register('zero_or_one', lambda x, y: 1 if random.random() <= 0.5 else 0, 0, 1)
toolbox.register('individual_creator', tools.initRepeat, creator.individual, toolbox.zero_or_one, NUM_OF_FEATURES)
toolbox.register('population_creator', tools.initRepeat, list, toolbox.individual_creator)

toolbox.register('evaluate', test_score)

# Genetic operators for binary data
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0/NUM_OF_FEATURES)




def search():
	population = toolbox.population_creator(n=POPULATION_SIZE)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register('min', numpy.min)
	stats.register('avg', numpy.mean)

	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
	population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
													  ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)
	best = hof.items[0]
	for x in hof.items:
		print('\tSelected feature=', feedback(x, True))
	print(f'\tAccuracy ={best.fitness.values[0]}')

	min_fit, mean_fit = logbook.select('min', 'avg')

	# Plotting
	sns.set_style('whitegrid')
	plt.plot(min_fit, color='red')
	plt.plot(mean_fit, color='green')
	plt.xlabel('Generation')
	plt.ylabel('Min / Average Fitness')
	plt.title('Min and Average fitness over Generations')
	plt.show()

search()

