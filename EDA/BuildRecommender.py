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
import dask.array as da
from tensorflow.python.framework import graph_util
import multiprocessing
from tensorflow.python.client import device_lib


#Work around to let tensorflow use CPUs instead of GPU
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#DEAP creator definition needs to be in global scope to parallelize
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#Individual represented as list of 4 floats 
#(arch,n_units,dropout,learning_rate)
#arch is the cell architecture 0 = lstm, 1 = gru
#floats are later decoded to appropriate sizes
#for each hyperparamter
IND_SIZE=4
creator.create("Individual", list, fitness=creator.FitnessMax)

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    """
    Early stopping class to exit training when validation loss begins increasing
    """
    def __init__(self, max_val_loss_delta):
        """
        best_val_loss - stores the best epochs validation loss for current val loss compariason
                        to check if increasing. Initizalized at -inf
        max_val_loss_delta - the maximum percent the current val loss can be above the best_val_loss 
                             without exiting training
        """ 
        self.best_val_loss = math.inf
        self.max_val_loss_delta = max_val_loss_delta
    
    def on_epoch_end(self, training_state):
        """ 
        This is the final method called in trainer.py in the epoch loop. 
        We can stop training and leave without losing any information with a simple exception.
        On epoch end, check if validation loss has increased by more than max_val_loss_delta, if True,
        exit training  
        """
        #check if current loss better than previous best and store
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
    Main class for data loading, parsing, model training and predictions.
    """

    def __init__(self, replay_data_dir=None,build_order_file=None,load_graph_file=None,down_sample=0.2,epochs=30):
        """
        replay_data_dir - directory of replay data files
        build_order_file - file location for save/stored parsed build order data
        load_graph_file - location of existing trained model to load_all_build_orders
        down_sample - the percent of total build_order training data samples to include in model training
        epochs - the number of epochs to run model for
        """
        self.replay_data_dir = replay_data_dir
        self.build_order_file = build_order_file
        self.unit_ids = self.load_unit_ids()
        self.vocab = []
        self.max_seq_len = 0
        self.down_sample = down_sample
        self.X = []
        self.y = []
        self.best_model_score = -math.inf
        self.best_ind = None
        self.model = None
        self.epochs = epochs
        self.race_units = {'Terran':[],'Zerg':[],'Protoss':[]}
        if load_graph_file:
            self.graph = self.load_graph(load_graph_file)
        else:
            self.graph = None     

    def load_unit_ids(self):
        with open("unit_ids.json") as infile:
            unit_ids = json.load(infile)
            unit_ids = {int(unit_id):name for unit_id,name in unit_ids.items()}
        return unit_ids

    def yield_replay_data(self):
        '''generator to yield into memory each replay data file individually.
        replay_data is a list of states, with each state containing the following
        data at the respective index:
        0: replay_id
        1: map_name
        2: player_id
        3: minimap
        4: friendly_army - [unit_id,count]
        5: enemy_army - [unit_id,count]
        6: player (resource data)
        7: availible_actions
        8: taken actions
        9: winner
        10: race
        11: enemy race
        '''
        for root, dirs, files in os.walk(self.replay_data_dir):
            for name in files:
                with open(self.replay_data_dir + "/" + name) as infile:
                    try:
                        replay_data = [json.loads(line) for line in infile]
                        yield replay_data
                    except ValueError:
                        print("Replay %s load failed" %(name))
                        
    def get_replay_build_order(self,replay_data):
        '''Script to parse out the build orders from a single replay_data file. 
        Build orders are the unique units seen from player1's persepective for both players
        in the order they are seen in the replay.
        '''
        build_order = []
        friendly_build = []
        enemy_build = []
        #iterate over state data and find unique units from friendly and enemy player through the states
        for state in replay_data:
            state_build = []
            #load all friendly units seen in this state
            for unit_data in state[4]:
                #check if unit count > 0
                if unit_data[1] > 0:
                    unit = self.unit_ids[unit_data[0]]
                    #lookup unit_id name and append to build if we haven't
                    #seen this unit in previous states    
                    if unit not in friendly_build:
                        friendly_build.append(unit)
                        #append player marker (friendly = 0) to identify player's unit
                        #in build order
                        state_build = state_build + [unit + str(0)]
            for unit_data in state[5]:
                if unit_data[1] > 0:
                    unit = self.unit_ids[unit_data[0]]
                    if unit not in enemy_build:
                        enemy_build.append(unit)
                        state_build = state_build + [unit + str(1)]
            #update build order if new unit seen this state for either player
            if state_build:
                build_order = build_order + state_build
        #gather static data from  first state
        player_id = replay_data[0][2]
        winner = replay_data[0][9]
        if player_id == winner:
            won = True
        else:
            won = False
        race = replay_data[0][10]
        enemy_race = replay_data[0][11]
        game_map = replay_data[0][1]
        replay_id = replay_data[0][0]
        build_data = [build_order,won,race,enemy_race,game_map,replay_id]
        return build_data        
       
    def save_all_build_orders(self):
        #clear save file is exists
        with open(self.build_order_file, 'w') as outfile: pass
        for replay_data in self.yield_replay_data():
            build_order = self.get_replay_build_order(replay_data)
            with open(self.build_order_file, 'a') as outfile:
                json.dump(build_order,outfile)
                outfile.write('\n')
    
    def load_all_build_orders(self):
        '''load parsed build order data and build vocabulary and race_unit
         lookup dictionaries from dataset.
         build_data - [build_order,won,race,enemy_race,game_map,replay_id]
         '''
        build_orders = []
        with open(self.build_order_file,'r') as infile:
            for line in infile:
                build_data = json.loads(line)
                self.max_seq_len = max(self.max_seq_len,len(build_data[0])-1)
                build_orders.append(build_data)
                races = [build_data[2],build_data[3]]
                for build in build_data[0]:
                    #pop of player marker for race_units dict
                    unit = build[0:-1]
                    player = int(build[-1])
                    if build not in self.vocab:
                        self.vocab.append(build)
                    if unit not in self.race_units[races[player]]:
                        self.race_units[races[player]].append(unit)          
        return build_orders

    def make_training_data(self):
        self.X = []
        self.y = []
        if not self.vocab:
            self.load_all_build_orders()
        with open(self.build_order_file,'r') as infile:
            for line in infile:
                #filter out down_sample % of training data
                if random.random() <= self.down_sample:
                    build_order = json.loads(line)[0]
                    #iterate over entire build_order sequence to produce samples/labels at
                    #each state of the build order ie from a sequence: ["Hatchery0", "Drone0", "Larva0"] get:
                    #sample = ["Hatchery0"] label = "Drone0"
                    #sample = ["Hatchery0","Drone0"] label = "Larva0" 
                    for i in range(len(build_order)-1):
                        self.X.append([self.vocab.index(unit) for unit in build_order[0:i+1]])
                        self.y.append(self.vocab.index(build_order[i+1]))
        return self.X, self.y

    def decode_individual(self,individual):
        #Scale the evolved individual's params (each between 0-1) to approriate sizes for each parameter
        arch = round(individual[0])
        n_units = int(individual[1]*256)
        dropout = individual[2]*.9
        learning_rate = individual[3]/50
        hyperparams = [arch,n_units,dropout,learning_rate]
        return hyperparams

    def evaluate(self,individual):
        #evaluate accuracy of an individual by training model
        #using evolved params and store best individuals model
        hyperparams = self.decode_individual(individual)
        model = self.train(hyperparams)
        model_score = model.evaluate(self.testX, self.testY)
        if model_score[0] > self.best_model_score:
            self.model = model
            self.best_model_score = model_score[0]
            self.best_ind = individual
        print("Model score = %s" %(model_score))
        return model_score
    
    def pred_preprocessing(self,pred_input):
        #Integer encode and pad prediction sequences
        X = [[self.vocab.index(unit) for unit in pred_input]]
        X = pad_sequences(X, maxlen=self.max_seq_len, value=0.,padding='post')
        return X

    def preprocessing(self,X,y):
        #Train/Test data splitting
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        # Sequence padding
        trainX = pad_sequences(trainX, maxlen=self.max_seq_len, value=0.,padding='post')
        testX = pad_sequences(testX, maxlen=self.max_seq_len, value=0.,padding='post')
        
        #Out of memory array building
        chunks = 10
        trainX = da.from_array(np.asarray(trainX), chunks=chunks)
        trainY = da.from_array(np.asarray(trainY), chunks=chunks)
        testX = da.from_array(np.asarray(testX), chunks=chunks)
        testY = da.from_array(np.asarray(testY), chunks=chunks)

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
        arch,n_units,dropout,learning_rate = hyperparams

        # Network building
        net = tflearn.input_data([None, self.max_seq_len])
        net = tflearn.embedding(net, input_dim=len(self.vocab), output_dim=128,trainable=True)

        if arch == 0:
            net = tflearn.lstm(net, n_units=n_units,
                               dropout=dropout,
                               weights_init=tflearn.initializations.xavier(),return_seq=False)
        else:
            net = tflearn.gru(net, n_units=n_units,
                              dropout=dropout,
                              weights_init=tflearn.initializations.xavier(),return_seq=False)
        net = tflearn.fully_connected(net, len(self.vocab), activation='softmax',
                                      weights_init=tflearn.initializations.xavier())
        net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                                 loss='categorical_crossentropy')
        model = tflearn.DNN(net, tensorboard_verbose=2,
                            tensorboard_dir='C:/Users/macle/Desktop/Open Source Projects/autocraft/EDA/tensorboard',
                            best_checkpoint_path='C:/Users/macle/Desktop/Open Source Projects/autocraft/EDA/models')       

        # Training
        early_stopping_cb = EarlyStoppingCallback(max_val_loss_delta=0.01)
        #Need to catch early stopping to return model
        try:
            model.fit(self.trainX, self.trainY, validation_set=(self.testX, self.testY), show_metric=False,snapshot_epoch=True,
                      batch_size=128,n_epoch=self.epochs,run_id="%s-%s-%s-%s" %(arch,n_units,dropout,learning_rate),
                      callbacks=early_stopping_cb)
            return model
        except StopIteration:
            return model
      
    def freeze_graph(self):
        # We precise the file fullname of our freezed graph
        output_graph = "C:/Users/macle/Desktop/Open Source Projects/autocraft/webapp/model/model.pb"

        # Before exporting our graph, we need to precise what is our output node
        # This is how TF decides what part of the Graph he has to keep and what part it can dump
        # NOTE: this variable is plural, because you can have multiple output nodes
        output_node_names = "InputData/X,FullyConnected/Softmax"

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We import the meta graph and retrieve a Saver
        #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We retrieve the protobuf graph definition
        graph = self.model.net.graph
        input_graph_def = graph.as_graph_def()

        # We start a session and restore the graph weights
        # We use a built-in TF helper to export variables to constants
        sess = self.model.session
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
    
    def load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the 
        # unserialized graph_def
        with tf.gfile.GFile("C:/Users/macle/Desktop/Open Source Projects/autocraft/webapp/model/model.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the 
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                name="prefix", 
                op_dict=None, 
                producer_op_list=None
            )
        self.graph = graph
        return graph
    
    def predict(self,pred_input):
        #preprocess and 'flow' prediction into trained model
        if not self.graph:
            self.freeze_graph()
            self.load_graph()
        pred_input = self.pred_preprocessing(pred_input)
        x = self.graph.get_tensor_by_name('prefix/InputData/X:0')
        y = self.graph.get_tensor_by_name("prefix/FullyConnected/Softmax:0")
        with tf.Session(graph=self.graph) as sess:
            build_probs = sess.run(y, feed_dict={
                x: pred_input
            })
        #return numpy array of all label probabilities
        return build_probs
    
    def recurse_predictions(self,pred_input,preds,races,copy_vocab):
        '''recurse through prediction probabilities to select the highest probably admissable
        label. Admissable labels are those that haven't been seen in the inputted prediction sequence 
        and exist in the appropriate race's race_unit dictionary'''
        #get most probable label
        rec_build = np.argmax(preds[0])
        rec = copy_vocab[rec_build]
        #pop player identifier off unit to checkin race_units dict
        unit = rec[0:-1]
        player = int(rec[-1])
        #check admissability of predicted label
        if rec not in pred_input and unit in self.race_units[races[player]]:
            return rec
        else:
            #if not admissable, delete prediction from recs and vocab and recurse 
            preds = np.delete(preds,rec_build)
            copy_vocab.pop(rec_build)
            return self.recurse_predictions(pred_input,preds,races,copy_vocab)       
    
    def predict_build(self,pred_input,build_length,races):
        '''
        pred_input - the input sequence the predict the next build order from
        build_length - the number of next units to predict and append to the current build
        races - Strings of friendly and enemy race names - [friendly_race,enemy_race]
        '''
        #predict the build_length most probable,admissible label from input sequence
        #using list creates copy
        copy_vocab = list(self.vocab)
        for i in range(build_length):
            build_probs = self.predict(pred_input)
            rec = self.recurse_predictions(pred_input,build_probs,races,copy_vocab)
            pred_input.append(rec)
        return pred_input

def evolve(builder,n_pop,co_prob,mut_prob,n_generations):
        '''Evolve the models hyperarameters (arch,n_units,dropout,learning_rate)'''

        toolbox = base.Toolbox()
        #Setup fitness (maximize val_loss)
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=IND_SIZE)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", builder.evaluate)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        #assign the number of processors to run parallel fitness evaluations on
        pool = multiprocessing.Pool(6)
        toolbox.register("map", pool.map)

        pop = toolbox.population(n=n_pop)
        best_ind = pop[0]
        best_fit = -math.inf

        # Evaluate the entire population
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            print("Evaluating %s" %ind)
            ind.fitness.values = fit

        for g in range(n_generations):
            print("Running generation %s" %(g))
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
            fitnesses = list(toolbox.map(toolbox.evaluate, pop))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                #keep the single best perfoming ind through all gens
                if fit[0] > best_fit:
                    best_ind = ind
                    best_fit = fit[0]

            # The population is entirely replaced by the offspring
            pop[:] = offspring
        return best_ind

if __name__== "__main__":
    builder = BuildRecommender("replay_state_data",'build_orders.json',down_sample=0.05)
    builder.load_all_build_orders()
    best_ind = evolve(builder,6,0.4,0.2,3)
    print(best_ind)