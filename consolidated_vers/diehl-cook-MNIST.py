import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import pickle
import brian2 as b
from struct import unpack
from brian2 import *
from brian2tools import *
import string


### Runtime functions

# result_monitor contains [since_last_update, spike_count_for_sample_per_neuron]
# input_labels contains [label_for_last_26_samples]
def get_new_assignments(result_monitor, input_labels):
    assignments = np.zeros(n_e)
    np_input_labels = np.asarray(input_labels)
    maximum_rate = [0] * n_e
    for j in range(10):
        num_assignments = len(np.where(np_input_labels == j)[0]) #selects the samples with label j
        if num_assignments > 0:
            #print('np_input_labels  shape ' + str(np_input_labels.shape))
            #print('result_monitor shape ' + str(result_monitor.shape))
            #print(result_monitor.shape)
            #print("is this the 20, 400")
            relevant_sample_results = result_monitor[np_input_labels == j]
            rate = np.sum(relevant_sample_results, axis = 0) / num_assignments #takes the avg response of
                                                                                    # each neuron during the samples 
                                                                                    # with label j
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments


### All this probably won't work because the input_dim is not a square value, oh well figure that out later
def get_2d_input_weights():
    weight_matrix = np.zeros((input_dim, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(input_dim))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
    connMatrix = np.zeros((input_dim, n_e))
    connMatrix[input_synapses.i, input_synapses.j] = input_synapses.w
    weight_matrix = np.copy(connMatrix)

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def normalize_input_weights():
    len_source = len(input_synapses.source)
    len_target = len(input_synapses.target)
    synapse_weights = np.zeros((len_source, len_target))
    synapse_weights[input_synapses.i, input_synapses.j] = input_synapses.w
    temp_conn = np.copy(synapse_weights)
    colSums = np.sum(temp_conn, axis = 0) # creates an array of n_e dimension, with the sum of the weights of all conections per input channel 
    colFactors = ee_input_weight_sum_norm/colSums # creates an array with values that if multiplied into a column will make that input neurons weights = 78
    for j in range(n_e):                  # 
        temp_conn[:,j] *= colFactors[j]
    input_synapses.w = temp_conn[input_synapses.i, input_synapses.j]
                    

# This function grabs the label for which neurons with that label have the highest activity on average
# spike_rates contains [spike_count_for_sample_per_neuron]

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    #print(spike_rates) 
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
    
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
        
    return np.argsort(summed_rates)[::-1]



### Plotting fuctions

def plot_performance(fig_num):
    num_evaluations = int(sample_count/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = figure(fig_num, figsize = (5, 5))
    #fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #zxzhijia's 
    ylim(ymax = 100)
    title('Classification performance')
    plt.show()
    return im2, performance, fig_num, fig


def update_performance_plot(im, performance, current_example_num, fig):
    num_evaluations = int(sample_count/update_interval)
    time_steps = range(0, num_evaluations)
    performance = get_current_performance(performance, current_example_num)
    fig = figure(fig_num, figsize = (5, 5))
    #fig_num += 1
    #print(fig_num)
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #zxzhijia's 
    ylim(ymax = 100)
    title('Classification performance')
    plt.show()
    return im, performance


def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_labels[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance


def plot_2d_input_weights(fig_num):
    weights = get_2d_input_weights()
    fig = plt.figure(fig_num, figsize = (18, 18))
    im2 = plt.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    plt.colorbar(im2)
    plt.title('weights of input connection')
    fig.canvas.draw()
    return (fig_num + 1), im2, fig


def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


# For using numpy directly for simulation instead of C++ codegen
prefs.codegen.target = 'cython'

notebook_path = os.path.abspath("diehl-cook-MNIST.py")
print(notebook_path)


input_dim = 5
epoch_num = 100 #100 chosen to be in a similar order of magnitude as the number of training samples used in [1] (3 epochs of 60000 sized MNIST data)
epoch_sample_count = (26*19)
sample_count = (26 * 19) * epoch_num #(26 letters per trial, 19 trials)
n_e = 400 # excitatory neuron count in [1] github: 400
neurons_per_categories = 8 # used in creating starting assignments (NOT CURRENTLY)
n_i = n_e # for now n_i must be equal to n_i, because of the way the weight matrix is generated with a for loop over n_e
input_sample_time = 350*ms
inter_input_time = 150*ms
full_sample_time = (input_sample_time + inter_input_time)
full_runtime = full_sample_time * sample_count


# These values just taken from [1]
v_rest_e = -65. * mV 
v_rest_i = -60. * mV 
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms


update_interval = 52 # How often the assignments are updated
weight_update_interval = 26 # how often the plotted weights are updated, just uninteresting plot variable


weight = {}
delay = {}
ee_input_weight_sum_norm = 78.  # The average amount of weight from the inputs to each excitatory population neuron
weight['ee_input'] = 0.3 # original had 0.3, currently not implemented so = 1 which is fine because our dataset much much smaller
                            # Weights are normalised to be 78 per n_e which would mean that in o.g. paper on avg about 0.1 per input neuron 
                            # but here means 15.6 per input neuron and given that [1] began with 0.3, that'd imply here to keep avg weight the same, 
                            # this num should be about 45! 
weight['ei_input'] = 0.2 # original had 0.2
weight['ee'] = 0.1       # 
weight['ei'] = 10.4
weight['ie'] = 17.0
weight['ii'] = 0.4

### How connected the synapses ought to be, what 
pConn = {}
pConn['ee_input'] = 1.0 # all to all
pConn['ei_input'] = 0.1 # this archtecture does not contain this type of synapse
pConn['ee'] = 1.0    # simple all to all
pConn['ei'] = 0.0025 # This one is more or less meaningless, it is simply 1/n_e it's currently set at 1/400
pConn['ie'] = 0.9    # In the current network this type of connection is not used
pConn['ii'] = 0.1    # Same here
    

delay['ee_input'] = (0*ms,10*ms) 
delay['ei_input'] = (0*ms,5*ms) #This one isn't used currently

# I pumped this input intensity from 64 to 256 because the input dimensionality is so low that it's probably not generating enough spikes
# This is an imperfect fix cuz the dimensionality is much better than 
input_intensity = 64. # maximum Hz of input (gets increased if not generating enough output)
start_input_intensity = input_intensity # resets to this after each input, if it was increased


tc_pre_ee = 20*ms
tc_post_1_ee = 20*ms
tc_post_2_ee = 40*ms
nu_ee_pre =  0.0001      # Change in weight w/ learning
nu_ee_post = 0.01        # Change in weight w/ learning
wmax_ee = 1.0
exp_ee_pre = 0.2
exp_ee_post = exp_ee_pre
STDP_offset = 0.4 

## Reset stuff
theta_plus_e = 0.05 * mV
e_reset_str = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
i_reset_str = 'v=v_reset_i'

## threshold voltages
offset = 20.*mV
e_thresh_str = '(v>(theta - offset + v_thresh_e )) and (timer>refrac_e)' # I don't recall the raison d'etre of offset in the architecture
i_thresh_str = 'v>v_thresh_i'


#Equations for processing neural layers
tc_theta = 1e7 * ms #time constant for theta
eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt (unless refractory)
        I_synE = ge * nS *         -v                               : amp
        I_synI = gi * nS * (-100.*mV-v)                             : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        dtheta/dt = -theta / (tc_theta)                             : volt
        dtimer/dt = 0.1                                             : second
        '''

eqs_i = '''
        dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (10*ms)   : volt (unless refractory)
        I_synE = ge * nS *         -v                               : amp
        I_synI = gi * nS * (-85.*mV-v)                              : amp
        dge/dt = -ge/(1.0*ms)                                       : 1
        dgi/dt = -gi/(2.0*ms)                                       : 1
        '''


#For excitatory to excitatory connection, in this case from input to excitatory neurons
eqs_stdp_ee = '''
                post2before                            : 1
                dpre/dt   =  -pre/(tc_pre_ee)          : 1 (event-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
            '''
eqs_stdp_pre_ee = 'pre = 1.; w = clip(w + nu_ee_pre * post1, 0, wmax_ee)'
eqs_stdp_post_ee = 'post2before = post2; w = clip(w + nu_ee_post * pre * post2before, 0, wmax_ee); post1 = 1.; post2 = 1.'



plt.ion() # Interactive mode on
fig_num = 1 # this might be causing issues in plotting
result_monitor = np.zeros((update_interval,n_e))


MNIST_data_path = os.path.join(os.path.dirname(notebook_path), "Datasets\\")
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename) and False:
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

### Just be in the doing, forget everything else about life, that's how you've been happy in the past

start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print('time needed to load training set:', end - start)

start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print('time needed to load test set:', end - start)

####### STUFF TO PUT INTO PREVIOUS STUFF
input_dim = 28*28
epoch_num = 3 #100 chosen to be in a similar order of magnitude as the number of training samples used in [1] (3 epochs of 60000 sized MNIST data)
epoch_sample_count = 60000
sample_count = epoch_sample_count * epoch_num
input_intensity = 64.
weight['ee_input'] = 0.3

update_interval = 5000 # How often the assignments are updated. THIS WAS 10000 in [1]
weight_update_interval = 20 # how often the plotted weights are updated, just uninteresting plot variable
result_monitor = np.zeros((update_interval,n_e))

###### STUFF THAT SHOULD BE ELSEWHERE ENDS

### 
### CREATING NETWORK OBJECTS
###

start_scope()
#Create neuron groups
e_neurons = NeuronGroup(n_e, eqs_e, threshold= e_thresh_str, refractory= refrac_e, reset= e_reset_str, method='euler')
i_neurons = NeuronGroup(n_i, eqs_i, threshold= i_thresh_str, refractory= refrac_i, reset= i_reset_str, method='euler')

#set start voltages
e_neurons.v = v_rest_e - 40*mV
i_neurons.v = v_rest_e - 40*mV

#set theta values
e_neurons.theta = np.ones((n_e)) * 20.*mV

## Create connections between the excitatory and inhibitory neurons
# First excitatory to inhibitory synapses
#e_weightMatrix = np.random.rand(n_e, n_i)
e_weightMatrix = np.zeros((n_e,n_i))
np.fill_diagonal(e_weightMatrix, np.random.rand(n_e))

model = 'w : 1'
pre = 'ge_post += w'
post = ''

ei_synapses = Synapses(e_neurons, i_neurons, model=model, on_pre=pre, on_post=post)
ei_synapses.connect(True) #Creates all-to-all connection
ei_synapses.w = e_weightMatrix[ei_synapses.i, ei_synapses.j] #.i and .j refer to the pre and post neuron in the synapse


# inhibitory to excitatory synapses
pre = 'gi_post += w'
i_weightMatrix = np.random.rand(n_i, n_e) * weight['ie']
np.fill_diagonal(i_weightMatrix, 0)

ie_synapses = Synapses(i_neurons, e_neurons, model=model, on_pre=pre, on_post=post)
ie_synapses.connect(True) #Creates all-to-all connection
ie_synapses.w = i_weightMatrix[ei_synapses.i, ei_synapses.j] #.i and .j refer to the pre and post neuron in the synapse


### The creation of Poisson inputs from vec data
input_weightMatrix = np.random.rand(input_dim, n_e) * weight['ee_input'] #initially random weights
input_neurons = PoissonGroup(input_dim, 0*Hz)


# creation of synapse from input to excitatory
model = 'w : 1' + eqs_stdp_ee
pre = 'ge_post += w; ' + eqs_stdp_pre_ee
post = eqs_stdp_post_ee

input_synapses = Synapses(input_neurons, e_neurons, model=model, on_pre=pre, on_post=post)

min_delay = delay['ee_input'][0]
max_delay = delay['ee_input'][1]
delta_delay = max_delay - min_delay
# TODO: test this ## That was note from brian2 translator, I wonder why, hmm

input_synapses.connect(True) # all-to-all connection
input_synapses.delay = 'min_delay + rand() * delta_delay'
input_synapses.w = input_weightMatrix[input_synapses.i, input_synapses.j]


### Creation of monitors
#e_rate_monitor = PopulationRateMonitor(e_neurons)
#i_rate_monitor = PopulationRateMonitor(i_neurons)
#input_rate_monitor = PopulationRateMonitor(input_neurons)
#rate_monitors = [e_rate_monitor, i_rate_monitor, input_rate_monitor]

e_spike_monitor = SpikeMonitor(e_neurons)
i_spike_monitor = SpikeMonitor(i_neurons)
spike_monitors = [e_spike_monitor, i_spike_monitor]

input_intensity = 64. # maximum Hz of input (gets increased if not generating enough output)
start_input_intensity = input_intensity # resets to this after each input, if it was increased



def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/perf_update_interval)
    start_num = current_example_num - perf_update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_labels[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(perf_update_interval) * 100
    return performance



###
### Running stuff
###


previous_spike_count = np.zeros(n_e)
#assignments = np.tile(np.linspace(0,25,26), neurons_per_categories)
assignments = np.zeros(n_e)
input_labels = [0] * sample_count
outputNumbers = np.zeros((sample_count, 10))
#performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)
perf_update_interval = 100
performance = np.zeros(int(sample_count/perf_update_interval))
results_data_path = os.path.join(os.path.dirname(notebook_path), "Results\\MNIST_accuracy")
#print("plot performance ran")
j=0
while j < sample_count:
    #if j%150 == 0:
    #    clear_output()
    
    spike_rates = training['x'][j%epoch_sample_count,:,:].reshape((input_dim)) / 8. *  input_intensity
    input_neurons.rates = spike_rates * Hz
    #print('prior norm', input_synapses.w)
    normalize_input_weights()
    #print(input_synapses.w[500:525], input_synapses.w[1500:1525])
    #print('post norm', input_synapses.w)
    run(input_sample_time, report = 'text') #, report = 'text'
    #print("DID run")
    
    if j % update_interval == 0 and j > 0:
        #print("new assignments got on sample" + j)
        assignments = get_new_assignments(result_monitor[:], input_labels[j-update_interval : j])
    

    current_spike_count = np.asarray(e_spike_monitor.count[:]) - previous_spike_count
    previous_spike_count = np.copy(e_spike_monitor.count[:])
    if np.sum(current_spike_count) < 5:
        print('skip at ', j)
        
        print('current label', training['y'][j%epoch_sample_count][0])
        plot(e_spike_monitor)
        print('current spike rates ', training['x'][j%epoch_sample_count,:] / 8. * input_intensity)
        print('actual_curr_spik_rates', spike_rates)
        input_intensity += 3 #The oroginial paper had 1 but there are so many skips that I think 3 is warranted
        input_neurons.rates = 0 * Hz
        run(inter_input_time)        
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        input_labels[j] = training['y'][j%epoch_sample_count][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        #if j % 100 == 0 and j > 0:
        #    print('runs done:', j, 'of', int(sample_count))
            
        ####

        if j % perf_update_interval == 0:
            performance = get_current_performance(performance, j)
            print('')
            print('current classification performance ' + str(performance[:int((j/float(perf_update_interval))+1)]))
            print('')
        if j % update_interval == 0 and j > 0:
            #print(input_labels[j:])
            #print(output_numbers)
            np.save((results_data_path + str(j)), performance[:int((j/float(perf_update_interval))+1)])
            #unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
            print('At run ', int(j/update_interval), 'of', int(sample_count/update_interval))
            print('')
            print('Classification performance ', performance[:int((j/float(perf_update_interval))+1)])
            print('')
            print('Assignments ', assignments)
            #print('Last 26 outputs', outputNumbers[j-26:j,:])
            print('')
            print('Last 10 guesses (top 2) ', outputNumbers[j-10:j,0:2])
            print('')
            print('')
            print('Run reports:')
            
        ####
        
        input_neurons.rates = 0 * Hz
        run(inter_input_time)
        input_intensity = start_input_intensity
        j += 1