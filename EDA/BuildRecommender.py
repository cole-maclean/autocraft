import os
import numpy as np
import json
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import random
from deap import base, creator, tools
import math

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, max_val_loss_delta):
        self.best_val_loss = math.inf
        self.max_val_loss_delta = max_val_loss_delta
    
    def on_epoch_end(self, training_state):
        """ 
        This is the final method called in trainer.py in the epoch loop. 
        We can stop training and leave without losing any information with a simple exception.  
        """
        self.best_val_loss = min(training_state.val_loss,self.best_val_loss)
        if (training_state.val_loss - self.best_val_loss) >= self.best_val_loss*self.max_val_loss_delta:
            print("Terminating training at the end of epoch", training_state.epoch)
            print("Epoch loss = %s vs best loss = %s" %(training_state.val_loss, self.best_val_loss))
            raise StopIteration      
    
    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training, 
        (or when training ends regardless). This would be a good time to store any additional 
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model loss:", training_state.val_loss)
       
        
class BuildRecommender():

    """

    """

    def __init__(self, replay_data_dir=None,build_order_file=None):
        self.replay_data_dir = replay_data_dir
        self.build_order_file = build_order_file
        self.unit_ids = self.load_unit_ids()
        self.vocab = []
        self.max_seq_len = 0
        self.X = []
        self.y = []

    def load_unit_ids(self):
        with open("unit_ids.json") as infile:
            unit_ids = json.load(infile)
            unit_ids = {int(unit_id):name for unit_id,name in unit_ids.items()}
        return unit_ids

    def yield_replay_data(self):
        for root, dirs, files in os.walk(self.replay_data_dir):
            for name in files:
                with open(self.replay_data_dir + "/" + name) as infile:
                    replay_data = [json.loads(line) for line in infile]
                yield replay_data

    def get_replay_build_order(self,replay_data):
        build_orders = []
        friendly_build = []
        enemy_build = []
        for state in replay_data:
            state_build = []
            for unit_data in state[4]:
                if unit_data[1] > 0:
                    unit = self.unit_ids[unit_data[0]] 
                    if unit not in friendly_build:
                        friendly_build.append(unit)
                        state_build = state_build + [unit + str(0)]
            for unit_data in state[5]:
                if unit_data[1] > 0:
                    unit = self.unit_ids[unit_data[0]] 
                    if unit not in enemy_build:
                        enemy_build.append(unit)
                        state_build = state_build + [unit + str(1)]
            if state_build:
                build_orders = build_orders + state_build
        return build_orders
        
    def save_all_build_orders(self):
        #clear save file is exists
        with open(self.build_order_file, 'w') as outfile: pass
        for replay_data in self.yield_replay_data():
            build_order = self.get_replay_build_order(replay_data)
            with open(self.build_order_file, 'a') as outfile:
                json.dump(build_order,outfile)
                outfile.write('\n')

    def get_build_order_vocab(self):
        self.vocab = []
        with open(self.build_order_file,'r') as infile:
            for line in infile:
                for unit in json.loads(line):
                    if unit not in self.vocab:
                        self.vocab.append(unit)
        return self.vocab

    def make_training_data(self):
        if self.vocab == []:
            self.get_build_order_vocab()
        self.X = []
        self.y = []
        with open(self.build_order_file,'r') as infile:
            for line in infile:
                build_order = json.loads(line)
                self.max_seq_len = max(self.max_seq_len,len(build_order)-1)
                for i in range(len(build_order)-1):
                    self.X.append([self.vocab.index(unit) for unit in build_order[0:i+1]])
                    self.y.append(self.vocab.index(build_order[i+1]))
        return self.X, self.y

    def decode_individual(self,individual):
        arch = round(individual[0])
        n_units = int(individual[0]*256)
        dropout = individual[1]
        learning_rate = individual[2]/10
        hyperparams = [arch,n_units,dropout,learning_rate]
        return hyperparams

    def evaluate(self,individual):
        hyperparams = self.decode_individual(individual)
        model = self.train(hyperparams)
        model_score = model.evaluate(self.testX, self.testY)
        print("Model score = %s" %(model_score))
        return model_score

    def preprocessing(self,X,y):
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        # Sequence padding
        trainX = pad_sequences(trainX, maxlen=self.max_seq_len, value=0.,padding='post')
        testX = pad_sequences(testX, maxlen=self.max_seq_len, value=0.,padding='post')

        # Converting labels to binary vectors
        trainY = to_categorical(trainY, nb_classes=len(self.vocab))
        testY = to_categorical(testY, nb_classes=len(self.vocab))
        return trainX, testX, trainY, testY

    def train(self, hyperparams):
        #reset graph from previously trained iterations
        tf.reset_default_graph()
        if self.X == []:
            self.make_training_data()
        
        self.trainX, self.testX, self.trainY, self.testY = self.preprocessing(self.X,self.y)    

        # Hyperparameters
        num_epochs = 25
        arch,n_units,dropout,learning_rate = hyperparams

        # Network building
        net = tflearn.input_data([None, self.max_seq_len])
        net = tflearn.embedding(net, input_dim=len(self.vocab), output_dim=128,trainable=True)
        if arch == 0:
            net = tflearn.lstm(net, n_units=n_units, dropout=dropout,weights_init=tflearn.initializations.xavier(),return_seq=False)
        else:
            net = tflearn.gru(net, n_units=n_units, dropout=dropout,weights_init=tflearn.initializations.xavier(),return_seq=False)
        net = tflearn.fully_connected(net, len(self.vocab), activation='softmax',weights_init=tflearn.initializations.xavier())
        net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                                 loss='categorical_crossentropy')

        # Training
        model = tflearn.DNN(net,tensorboard_dir='/tmp/tflearn_logs/shallow_gru/', tensorboard_verbose=2)
                            #checkpoint_path='/tmp/tflearn_logs/shallow_lstm/',
                            #best_checkpoint_path="C:/Users/macle/Desktop/UPC Masters/Semester 2/CI/SubRecommender/models/")
 
        early_stopping_cb = EarlyStoppingCallback(max_val_loss_delta=0.01)
        #Need to catch early stopping to return model
        try:
            model.fit(self.trainX, self.trainY, validation_set=(self.testX, self.testY), show_metric=False,snapshot_epoch=True,
                      batch_size=64,n_epoch=num_epochs,run_id="%s-%s-%s-%s" %(arch,learning_rate,n_units,dropout),
                      callbacks=early_stopping_cb)
            return model
        except StopIteration:
            return model
   
    def evolve(self,n_pop,co_prob,mut_prob,n_generations):

        #Setup fitness (maximize val_loss)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        #Individual represented as list of 4 floats 
        #(num_epochs,n_units,dropout,learning_rate)
        #floats are later scaled to appropriate sizes
        #for each hyperparamter

        IND_SIZE=4
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=IND_SIZE)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        pop = toolbox.population(n=n_pop)
        best_ind = pop[0]
        best_fit = -math.inf

        # Evaluate the entire population
        fitnesses = map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(n_generations):
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < co_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                #keep the single best perfoming ind through all gens
                if fit[0] > best_fit:
                    best_ind = ind
                    best_fit = fit[0]
                

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return best_ind, pop


if __name__ == "__main__":
  builder = BuildRecommender("replay_state_data",'build_orders.json')
  #builder.train([256,0.1,0.01])
  builder.evolve(10,0.2,0.2,50)