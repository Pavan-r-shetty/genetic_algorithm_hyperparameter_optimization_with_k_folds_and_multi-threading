import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import numpy as np
import tensorflow.keras.backend as kb
import ml_util_4_head
import time
from scipy import signal
import random
import importlib
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from keras.callbacks import Callback
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor as Executor
import pandas as pd
import wandb
import numpy as np
import random
import itertools


# Experiment_options
importlib.reload(ml_util_4_head)
window_size = 200
num_snapshots_in_sequence = 300

all_subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Function to generate splits
def generate_splits(subjects):
    splits = []
    for i in range(len(subjects)):
        train_subjects = subjects[i:i+7]  # Get 7 subjects for training
        if len(train_subjects) < 7:
            train_subjects += subjects[0:(7-len(train_subjects))]
        valid_subject = subjects[(i+7) % len(subjects)]
        test_subject = subjects[(i+8) % len(subjects)]
        splits.append((train_subjects, valid_subject, test_subject))
    return splits

# Generate and print the splits
splits = generate_splits(all_subjects)
# for train_subjects, valid_subject, test_subject in splits:
#     print(f"train_subjects = {train_subjects}, valid_subject = {valid_subject}, test_subject = {test_subject}")


sides = ['LEFT', 'RIGHT']
trial_nums = [1,2,3,4,5,6,7,8,9,10,11,12]
# trial_nums = [1]

root_folder = "latest_dataset_for_4_head"
sequence_len = num_snapshots_in_sequence + window_size - 1
training_instances = np.empty(shape=[0,sequence_len, 12], dtype=np.float32)


def construct_model_2023v2(window_size, filter_sizes, kernel_sizes, dilations, num_channels=8, batch_norm_insertion_pts=[2], sp_dense_sizes=[20, 10], ss_dense_sizes=[20, 10], v_dense_sizes=[20, 10], r_dense_sizes=[20, 10], do_fix_input_dim=False):
    if len(filter_sizes) != len(kernel_sizes) + 1:
        raise ValueError('Must provide one more filter size than kernel size--last kernel size is calculated')
    current_output_size = window_size  # Track for final conv layer

    if do_fix_input_dim:
        input_layer = tf.keras.layers.Input(shape=(window_size, num_channels), name='my_input_layer')
    else:
        input_layer = tf.keras.layers.Input(shape=(None, num_channels), name='my_input_layer')

    z = input_layer

    # Iterating over each Conv1D block, ensuring unique layer names
    for block in ['a', 'b', 'c', 'd']:  # Assuming 4 blocks as per the provided structure
        current_output_size = window_size
        z = input_layer
        for layer_idx, (filter_size, kernel_size, dilation) in enumerate(zip(filter_sizes[:-1], kernel_sizes, dilations)):
            z = tf.keras.layers.Conv1D(filters=filter_size, kernel_size=kernel_size, dilation_rate=dilation, activation='relu', name=f'conv1d_{block}_{layer_idx}')(z)
            if layer_idx in batch_norm_insertion_pts:
                z = tf.keras.layers.BatchNormalization(name=f'batch_norm_{block}_{layer_idx}')(z)
            current_output_size -= (kernel_size - 1) * dilation

        if current_output_size < 1:
            raise ValueError('layers shrink the cnn too much')
        else:
            z = tf.keras.layers.Conv1D(filters=filter_sizes[-1], kernel_size=current_output_size, activation='relu', name=f'conv1d_final_{block}')(z)

        # Adding dense layers specific to each block
        if block == 'a':
            for idx, num_neurons in enumerate(sp_dense_sizes):
                z = tf.keras.layers.Dense(num_neurons, activation='relu', name=f'dense_{block}_{num_neurons}_{idx}')(z)
            output_stance_phase = tf.keras.layers.Dense(1, name='stance_phase_output')(z)

        elif block == 'b':
            for idx,num_neurons in enumerate(ss_dense_sizes):
                z = tf.keras.layers.Dense(num_neurons, activation='relu', name=f'dense_{block}_{num_neurons}_{idx}')(z)
            output_stance_swing = tf.keras.layers.Dense(1, activation='sigmoid', name='stance_swing_output')(z)

        elif block == 'c':
            for idx,num_neurons in enumerate(v_dense_sizes):
                z = tf.keras.layers.Dense(num_neurons, activation='relu', name=f'dense_{block}_{num_neurons}_{idx}')(z)
            velocity = tf.keras.layers.Dense(1, name='velocity_output')(z)

        elif block == 'd':
            for idx,num_neurons in enumerate(r_dense_sizes):
                z = tf.keras.layers.Dense(num_neurons, activation='relu', name=f'dense_{block}_{num_neurons}_{idx}')(z)
            ramp = tf.keras.layers.Dense(1, name='ramp_output')(z)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_stance_phase, output_stance_swing, velocity, ramp])

    return model


class TQDMProgressBar(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_progress = tqdm(total=self.params['steps'],
                                   desc=f'Epoch {self.current_epoch}/{self.epochs}',
                                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}',
                                   leave=False)  # Set leave=False so progress bar disappears after epoch

    def on_batch_end(self, batch, logs=None):
        # Update the description with the current losses
        desc = f'Epoch {self.current_epoch}/{self.epochs} - loss: {logs["loss"]:.4f}'
        for output in ['stance_phase_output', 'stance_swing_output', 'velocity_output', 
                       'ramp_output']:
            if f'{output}_loss' in logs:
                desc += f' - {output}_loss: {logs[f"{output}_loss"]:.4f}'

        self.epoch_progress.set_description(desc)
        self.epoch_progress.update(1)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress.close()

# Define the hyperparameter space



# Define the hyperparameter space
hyperparam_space = {
    "filter_sizes": list(itertools.product(range(5, 50),  repeat = 1)),
    
    "kernel_sizes": list(itertools.product(range(5, 50),  repeat = 1)),

    "cnn_depth" : list(itertools.product(range(2, 5), repeat = 1)),
    
    # Two dilations each ranging from 1 to 8
    "dilations": list(itertools.product(range(1, 8), repeat=1)),
  


    "dense_depth" : list(itertools.product(range(1, 5),  repeat = 1)), 
    "dense_width" : list(itertools.product(range(10, 50),  repeat = 1)), 


}

    # "filter_sizes": list(itertools.combinations(range(5, 50),2 )),
    
    # "kernel_sizes": list(itertools.combinations(range(5, 50), 1)),
    
    # # Two dilations each ranging from 1 to 8
    # "dilations": list(itertools.product(range(1, 8), repeat=1)),
  

    # # Two batch norm insertion points each ranging from 0 to 3
    # # "batch_norm_insertion_pts": list(itertools.combinations(range(0, 4), 2)),

  

    # # Three values for each dense size parameter, ranging from 15 to 45
    # # "sp_dense_sizes": list(itertools.combinations(range(5, 50), 3)),
    # # "ss_dense_sizes": list(itertools.combinations(range(5, 50), 3)),
    # # "v_dense_sizes": list(itertools.combinations(range(5, 50), 3)),
    # "r_dense_sizes": list(itertools.combinations(range(5, 50), 3))


# Initialize W&B
wandb.init(project="genetic_algorithm_optimization", entity="shetty-pa")

head_rmse_means = []

# Initialize a population
def initialize_population(pop_size):
    print('initialize population')
    population = []
    for _ in range(pop_size):
        individual = {}
        for param, value in hyperparam_space.items():
     
            individual[param] = random.choice(value)
        population.append(individual)
    return population


def load_train_data_for_subjects(train_subjects):

    sequence_len = num_snapshots_in_sequence + window_size - 1
    training_instances = np.empty(shape=[0,sequence_len, 12], dtype=np.float32)
    files_to_train_with = ml_util_4_head.get_files_to_use(root_folder, train_subjects, sides, trial_nums)
    for myfile in files_to_train_with:
        data = ml_util_4_head.load_file(myfile)
        
        ss_col = data[:,-4]
        ss_col = ml_util_4_head.manipulate_ss_col(ss_col)
        data[:,-4] = ss_col

        # MAX 
        ramp_col = data[:,-2]
        ramp_col[ss_col==0]=-100
        data[:,-2] = ramp_col

        num_rows, num_cols = data.shape
        num_rows_to_drop = num_rows % sequence_len

        data = data[0:-num_rows_to_drop]
        new_num_rows, num_cols = data.shape
        num_sequences = new_num_rows/sequence_len
        new_data_shape = (int(num_sequences), sequence_len, num_cols)
        new_instances = data.reshape(new_data_shape)
        training_instances = np.append(training_instances, new_instances, axis=0)


    shuffled_training_instances = tf.random.shuffle(training_instances) 
    num_channels = 8
    x = shuffled_training_instances[:, :, :num_channels]
    y_v = shuffled_training_instances[:, window_size-1:,-1]
    y_r = shuffled_training_instances[:, window_size-1:,-2]
    y_sp = shuffled_training_instances[:, window_size-1:,-4]
    y_ss = shuffled_training_instances[:, window_size-1:,-3]

    return x, y_sp, y_ss, y_v, y_r


def load_valid_data_for_subject(valid_subject):

    sequence_len = num_snapshots_in_sequence + window_size - 1
    valid_instances = np.empty(shape=[0,sequence_len, 12], dtype=np.float32)
    files_to_valid_with = ml_util_4_head.get_files_to_use(root_folder, valid_subject, sides, trial_nums)
    for myfile in files_to_valid_with:
        data = ml_util_4_head.load_file(myfile)
        
        ss_col = data[:,-4]
        ss_col = ml_util_4_head.manipulate_ss_col(ss_col)
        data[:,-4] = ss_col

        # MAX 
        ramp_col = data[:,-2]
        ramp_col[ss_col==0]=-100
        data[:,-2] = ramp_col

        num_rows, num_cols = data.shape
        num_rows_to_drop = num_rows % sequence_len

        data = data[0:-num_rows_to_drop]
        new_num_rows, num_cols = data.shape
        num_sequences = new_num_rows/sequence_len
        new_data_shape = (int(num_sequences), sequence_len, num_cols)
        new_instances = data.reshape(new_data_shape)
        valid_instances = np.append(valid_instances, new_instances, axis=0)


    shuffled_valid_instances = tf.random.shuffle(valid_instances) 
    num_channels = 8
    x = shuffled_valid_instances[:, :, :num_channels]
    y_v = shuffled_valid_instances[:, window_size-1:,-1]
    y_r = shuffled_valid_instances[:, window_size-1:,-2]
    y_sp = shuffled_valid_instances[:, window_size-1:,-4]
    y_ss = shuffled_valid_instances[:, window_size-1:,-3]

    return x, y_sp, y_ss, y_v, y_r


children_in_gen_info = {}
parents_info = []
parents_selected_info = []

# Fitness function (model training and evaluation)
def evaluate_individual(individual, ind_num, generation):
    print(f"Training individual {ind_num} in generation {generation} with hyperparameters: {individual}")

    # Clear any existing TensorFlow graph
    tf.keras.backend.clear_session()
    try:
        # Build the model with the individual's hyperparameters
        dense_list =  [individual['dense_width'][0]] * individual['dense_depth'][0]

        filter_list = [individual["filter_sizes"][0]] * individual["cnn_depth"][0]

        kernel_list = [individual["kernel_sizes"][0]] * (individual["cnn_depth"][0] - 1)

        dilations_list = [individual["dilations"][0]] * (individual["cnn_depth"][0] - 1)

        print(f"Training individual {ind_num} in generation {generation} with filter_list: {filter_list}, \
        kernel_list: {kernel_list}, dilations_list: {dilations_list}, dense_list: {dense_list}")


        model = construct_model_2023v2(window_size=window_size,
                                    filter_sizes=filter_list,
                                    kernel_sizes=kernel_list,
                                    dilations=dilations_list,
                                    num_channels=8,
                                    batch_norm_insertion_pts=[0,1],
                                    sp_dense_sizes=dense_list,
                                    ss_dense_sizes=dense_list,
                                    v_dense_sizes=dense_list,
                                    r_dense_sizes=dense_list,
                                    do_fix_input_dim=False)
        # Compile the model with the individual's learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=[ml_util_4_head.custom_loss, 'binary_crossentropy',
                            ml_util_4_head.custom_loss, ml_util_4_head.custom_loss_for_ramp],
                    loss_weights=[4,1, 0.3, 0.1],
                    optimizer=optimizer)
        try:
            # Create dummy data for error checking
            window_size_dummy = 200  # Adjust according to your needs
            num_channels_dummy = 8  
            dummy_data = np.random.rand(window_size_dummy, num_channels_dummy)
            x_test_dummy = tf.expand_dims(dummy_data[:, :num_channels_dummy], axis=0)
            x_test_dummy = tf.cast(x_test_dummy, dtype=tf.float32)

            # Attempt a dummy prediction to test model viability
            model_outputs_dummy = model.predict(x_test_dummy)
        except Exception as e:
            print("Failed to create the model due to error:", e)
            return float('inf')  # Indicate failure to the genetic algorithm


        # Ensure the models directory exists
        models_dir = 'models'

        try:
            os.makedirs(models_dir)
        except FileExistsError:
            pass  # Directory already exists, so do nothing


        # Ensure a subdirectory for the current generation exists
        generation_dir = os.path.join(models_dir, f'generation_{generation}')

        try:
            os.makedirs(generation_dir)
        except FileExistsError:
            pass  # Directory already exists, so do nothing

        mean_score_list = []
        for train_subjects, valid_subject, test_subject in splits:
            # Load data for the selected subjects
            x_train, y_sp_train, y_ss_train, y_v_train, y_r_train = load_train_data_for_subjects(train_subjects)
            x_valid, y_sp_valid, y_ss_valid, y_v_valid, y_r_valid = load_valid_data_for_subject([valid_subject])
            # Define callbacks
            filename = os.path.join(generation_dir, f'individual_{ind_num}_{test_subject}.h5')
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
            mc = tf.keras.callbacks.ModelCheckpoint(filename, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
            tqdm_callback = TQDMProgressBar()
            print(f" subjects in training {train_subjects}")
            print(f" subjects used for validation {valid_subject}")
            print(f" subjects used for testing {test_subject}")

            history = model.fit(x=x_train, 
                            y=[y_sp_train, y_ss_train, y_v_train, y_r_train],
                            batch_size=32, 
                            epochs=50, 
                            validation_data=(x_valid, [y_sp_valid, y_ss_valid, y_v_valid, y_r_valid]),
                            callbacks=[es, mc, tqdm_callback],  # Add the custom callback
                            verbose=0)  # Set verbose to 0 to disable the default progress bar
            rmses = []
            rmses_vel = []
            rmses_isp = []
            rmses_ramp = []



            num_channels = 8

            files_to_test_with = ml_util_4_head.get_files_to_use(root_folder, [test_subject], sides, trial_nums)
            for myfile in files_to_test_with:
                data = ml_util_4_head.load_file(myfile)

                ss_col_test = data[window_size-1:,-4]


                # MAX 
                ramp_col_test = data[window_size-1:,-2]
                ramp_col_test[ss_col_test==0]=-100
                data[window_size-1:,-2] = ramp_col_test
            


                x_test = tf.expand_dims(data[:,:num_channels], axis=0)
                x_test=tf.cast(x_test, dtype = tf.float32)
                y_sp_test = data[window_size-1:,-4]
                y_ss_test = data[window_size-1:,-3]
                y_v_test = data[window_size-1:,-1]
                y_r_test = data[window_size-1:,-2]


                model_outputs = model.predict(x=x_test)    
                y_sp_predict, y_ss_predict, y_v_predict, y_r_predict = tf.squeeze(model_outputs)

                # Keep the original slicing if test_subject is not 1 and filter where y_sp_test != -1
                valid_indices = (y_sp_test[7000:-900] != -1) & (y_ss_test[7000:-900] == True)
                err = y_sp_predict[7000:-900][valid_indices] - y_sp_test[7000:-900][valid_indices]

                rmse = np.sqrt(np.mean(np.square(err))) 
                rmses.append(rmse)


                err_isp = y_ss_predict[7000:-900]-y_ss_test[7000:-900]
            
                # Mask nan values
                err_isp = err_isp[~np.isnan(err_isp)]
                rmse_isp = np.sqrt(np.mean(np.square(err_isp))) 
                rmses_isp.append(rmse_isp)
        


                err_vel = y_v_predict - y_v_test
                err_vel = err_vel[y_v_test!=-1].numpy()
                rmse_vel = np.sqrt(np.mean(np.square(err_vel)))
                rmses_vel.append(rmse_vel)


                # Mask where y_r_test is not -100
                mask = y_r_test != -100

                # Apply mask to calculate RMSE
                err_ramp = y_r_predict[mask] - y_r_test[mask]
                rmse_ramp = np.sqrt(np.mean(np.square(err_ramp)))
                rmses_ramp.append(rmse_ramp)


            mean_score_per_model = 4*np.mean(rmses) + np.mean(rmses_isp) + 0.3*np.mean(rmses_vel) + 0.1*np.mean(rmses_ramp)
            # Calculate the mean of each RMSE list
            mean_rmse = np.mean(rmses)
            mean_rmse_isp = np.mean(rmses_isp)
            mean_rmse_vel = np.mean(rmses_vel)
            mean_rmse_ramp = np.mean(rmses_ramp)

            # Create a DataFrame
            head_rmse_means.append({
                'generation' : [generation],
                'individual_number' : [ind_num],
                'individual': [individual],
                'test_subject' : [test_subject],
                'RMSE': [mean_rmse],
                'RMSE_ISP': [mean_rmse_isp],
                'RMSE_VEL': [mean_rmse_vel],
                'RMSE_RAMP': [mean_rmse_ramp]
            })



            mean_score_list.append(mean_score_per_model)

            print(f"Individual {ind_num} in generation {generation} with test subject {test_subject} achieved mean_score: {mean_score_per_model}")
            # Log the individual's performance
            wandb.log({"generation": generation, 
                    "individual_num": ind_num, 
                    "test_subject": test_subject,
                    "mean_score_per_model": mean_score_per_model, 
                    **individual})
            

        mean_score = np.mean(mean_score_list)




        children_in_gen_info[ind_num, generation] = {
            'generation': generation,
            'individual': ind_num,
            'params': individual,
            'mean_score' : mean_score,
            'mean_score_list': mean_score_list,  
        }



        print(f"Individual {ind_num} in generation {generation} achieved AVERAGE mean_score: {mean_score}")
        return mean_score 
    
    except ValueError as e:
        if str(e) == 'layers shrink the cnn too much':
            print(f"Configuration for individual {ind_num} in generation {generation} is not viable (CNN layers shrink too much). Assigning high validation loss.")
            return float('inf')  # Assign an infinite loss to indicate a poor configuration
        else:
            raise  # Re-raise the exception if it's not the specific one we're catching

    
          



# Selection
def select_parents(population, fitness_scores, num_parents):
    print('select parents')
    fitness_scores_array = np.array(fitness_scores)
    sorted_indices = np.argsort(fitness_scores_array)[:num_parents]


    parents = np.array(population)[sorted_indices].tolist()
    print(parents)
    return parents

# Crossover
def crossover(parent1, parent2):
    #print('crossover')
    child = {}
    for param in hyperparam_space:
        if param == "dense_sizes":
            child[param] = {k: random.choice([parent1[param][k], parent2[param][k]]) for k in parent1[param]}
        else:
            child[param] = random.choice([parent1[param], parent2[param]])
    #print('child', child)
    return child

def mutate(individual):
    param_to_mutate = random.choice(list(hyperparam_space.keys()))

    try:
        # Handle 'dense_sizes' separately
        if param_to_mutate == "dense_sizes":
            for key in hyperparam_space["dense_sizes"]:
                individual[param_to_mutate][key] = random.sample(hyperparam_space[param_to_mutate][key], 3)
        else:
            individual[param_to_mutate] = random.choice(hyperparam_space[param_to_mutate])

    except KeyError as e:
        # Log the error and relevant information
        print(f"KeyError occurred with key: {param_to_mutate}")
        print(f"Error details: {e}")
        print(f"Current individual: {individual}")
        # Handle the error appropriately

    return individual

# Genetic algorithm
def genetic_algorithm(pop_size, generations, num_parents):
    population = initialize_population(pop_size)

    best_individuals_per_generation = []
    for gen in range(generations):
        print(f"Starting generation {gen + 1}")

        # Evaluate individuals in parallel and preserve order
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all individuals for evaluation
            futures = {executor.submit(evaluate_individual, ind, ind_num, gen + 1): ind_num for ind_num, ind in enumerate(population)}
            
            # Collect fitness scores in the original order
            fitness_scores = [None] * len(population)  # Initialize a list to hold fitness scores in the original order
            for future in concurrent.futures.as_completed(futures):
                ind_num = futures[future]  # Retrieve the original index from the future
                fitness_scores[ind_num] = future.result()  # Place the result in the corresponding position

        max_accuracy = np.nanmin(fitness_scores)
        # best_individual = population[np.argmin(fitness_scores)]
        best_individual = population[np.nanargmin(fitness_scores)]

        print(f"Best individual in generation {gen + 1}: {best_individual} with rmse_average: {max_accuracy}")

        best_individuals_per_generation.append({
            'generation': gen + 1,
            'best_fitness': max_accuracy,
            'parameters': best_individual
        })

        parents = select_parents(population, fitness_scores, num_parents)


        parents_info.append({
            "generation" : gen + 1,
            "parents" : parents,
            "generation" : gen+1
        })

        next_generation = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(parents, 2)

            parents_selected_info.append({
                "parents_selected" : [parent1, parent2],
                "child_info" : _+1,
                "generation" : gen+1
            })

            child = crossover(parent1, parent2)
            next_generation.append(mutate(child))

        population = next_generation
        print(f"Generation {gen + 1} completed.\n")

    return best_individuals_per_generation

best_individuals_per_generation = genetic_algorithm(pop_size=12, generations=8, num_parents=6)



# Assuming best_individuals_per_generation is a list of dictionaries
df = pd.DataFrame(best_individuals_per_generation)
csv_file_path = 'best_individuals_per_generation.csv'
df.to_csv(csv_file_path, index=False)
print(f'Saved optimization details to {csv_file_path}')


df_means = pd.DataFrame(head_rmse_means)
csv_file_path = '4head_means.csv'
df_means.to_csv(csv_file_path, index=False)
print(f'Saved head performance to {csv_file_path}')


df_parents_info = pd.DataFrame(parents_info)
csv_file_path = 'parents_info.csv'
df_parents_info.to_csv(csv_file_path, index=False)
print(f'Saved parents_info details to {csv_file_path}')

df_parents_selected_info = pd.DataFrame(parents_selected_info)
csv_file_path = 'parents_selected_info.csv'
df_parents_selected_info.to_csv(csv_file_path, index=False)
print(f'Saved parents_selected_info details to {csv_file_path}')


# Convert dictionary to DataFrame for easy CSV writing
df_mean_score_info = pd.DataFrame.from_dict(children_in_gen_info, orient='index')
# Save to CSV
df_mean_score_info.to_csv('mean_score_info_with_generation.csv')







