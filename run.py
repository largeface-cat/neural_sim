from neural import *
from multiprocessing import Pool, Process, Queue
from copy import deepcopy


np.random.seed(42)
ENV_SIZE = 100
plt.ion()
x = np.arange(ENV_SIZE)
y = np.arange(ENV_SIZE)
xx, yy = np.meshgrid(x, y)
plot_x = 4
plot_y = 2
fig, ax = plt.subplots(plot_x, plot_y, figsize=(plot_x*3, plot_y*5))
ax = ax.flatten()
ENV = Environment(space=(ENV_SIZE, ENV_SIZE),
                  ax=ax[:4]
                  )

def random_neuron_list(neuron_params=[5, 5, 5], need_shuffle=True):
    neuron_list = []
    for i in range(neuron_params[0]):
        neuron = Neuron([randint(ENV_SIZE, size=(2,)) for _ in range(1000)], 
                        [randint(ENV_SIZE, size=(2,))], 
                        randint(5), 
                        {2: sig_prob(0.5), 3: sig_prob(0.5)}, 
                        {2: (rand()), 3: rand()}, 
                        {2: rand(), 3: rand()}, 
                        randint(1, 5),
                        'Neuron'+str(i), ENV)
        neuron_list.append(neuron)
    for i in range(neuron_params[1]):
        receptor = Receptor((0, [randint(ENV_SIZE, size=(2,)) for _ in range(20)]), 
                            [randint(ENV_SIZE, size=(2,))], 
                            randint(5),
                            {0: sig_prob(1)}, # 20% hot, 80% cold
                            {2: rand()*3, 3: rand()*3}, 
                            randint(1, 5),
                            'Receptor'+str(i), ENV)
        neuron_list.append(receptor)
    for i in range(neuron_params[2]):
        actuator = Actuator([randint(ENV_SIZE, size=(2,)) for _ in range(20)], 
                            (1, (randint(ENV_SIZE, size=(2,)), )), 
                            randint(5),
                            {2: sig_prob(0.5), 3: sig_prob(0.5)}, 
                            {2: rand()*3, 3: rand()*3}, 
                            randint(2, 5),
                            'Actuator'+str(i), ENV)
        neuron_list.append(actuator)
    if need_shuffle:
        shuffle(neuron_list)
    return neuron_list

# neuron_list1 = random_neuron_list(neuron_params=[200, 1000, 200])
# network1 = NeuralNetwork(env=ENV)
# network1.reverse_translate(neuron_list1)
# network1.translate()

# neuron_list2 = random_neuron_list(neuron_params=[200, 1000, 200])
# network2 = NeuralNetwork(env=ENV)
# network2.reverse_translate(neuron_list2)
# network2.translate()

# new_gene = network1.gene[:len(network1.gene)//2] + network2.gene[len(network2.gene)//2:]
# network = NeuralNetwork(env=ENV, ax=ax[4:])
# network.gene = new_gene
# network.translate()
def work(q:Queue, gene:str, env:Environment):
    network = NeuralNetwork(env=env, ax=ax[4:])
    network.gene = gene
    network.translate()
    loss = 0
    for i in range(100):
        space_dict_copy = network.update()
        for name, space in space_dict_copy.items():
            env.space_dict[name] += space
        env.legalize(exclude=[0, 1], eliminate_excess=True)
        env.visualize()
        network.visualize()
        loss += (env.space_dict[0] * env.space_dict[1]).sum()
        plt.pause(.1)
        
        circle = (xx - env.time % 100)**2 + (yy - env.time%100)**2 <= (ENV_SIZE//5)**2
        env.space_dict[0] = np.zeros((ENV_SIZE, ENV_SIZE))
        env.space_dict[0][circle] = 100
        env.space_dict[1] = np.zeros((ENV_SIZE, ENV_SIZE))
        env.time += 1
        # print(env.time, env.space_dict[2].max(), env.space_dict[2].max(), loss)
        
    q.put([loss, len(gene), len(network.neurons)])

if __name__ == '__main__':
    neuron_list1 = random_neuron_list(neuron_params=[100, 500, 200])
    network1 = NeuralNetwork(env=ENV)
    network1.reverse_translate(neuron_list1)
    network1.translate()
    original_gene = network1.gene
    
    q = Queue()
    p_list = []
    n_process = 10
    # p = Process(target=work, args=(q, deepcopy(original_gene), deepcopy(ENV)))
    work(q, original_gene, ENV)
    # p.start()
    # p_list.append(p)
    # for i in range(n_process):
    #     gene = original_gene
    #     start, end = len(gene)//n_process*i, len(gene)//n_process*(i+1)
    #     p = Process(target=work, args=(q, deepcopy(gene[start:end]), deepcopy(ENV)))
    #     p.start()
    #     p_list.append(p)
    # for p in p_list:
    #     p.join()
    # for i in range(n_process+1):
    #     print(q.get())
    print('done')
