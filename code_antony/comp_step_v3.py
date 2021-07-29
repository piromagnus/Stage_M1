import random
import sys
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import sklearn
import random
import os
import tensorflow as tf
import time
import psutil
from collections import deque


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
	
        tf.config.experimental.set_virtual_device_configuration(
            gpus[int(sys.argv[1])], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(sys.argv[2]))])
        tf.config.experimental.set_visible_devices(gpus[int(sys.argv[1])], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(10)
        # Virtual devices must be set before GPUs have been initialized
        print(e)

process = psutil.Process(os.getpid())

# Explication de l'output console après chaque epoch :
# Numéro de l'epochs et temps
# Epoch : 0 en 53 secondes
# Accuracy sur le jeu de train
# [99, 100, 97, 98, 81, 0]
# Accuracy sur le jeu de test
# [96, 98, 93, 97, 78, 0]
# Accuracy chiffre par chiffre sur le jeu de train
# [[99, 99, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100, 100], [100, 100, 100, 99, 98, 98, 99], [100, 100, 100, 100, 99, 99, 100], [100, 98, 98, 99, 85, 98, 100], [22, 4, 7, 3, 2, 6, 31]]
# Accuracy chiffre par chiffre sur le jeu de test
# [[98, 96, 100, 100, 100, 100, 100], [100, 100, 98, 99, 99, 100, 100], [100, 100, 99, 95, 99, 98, 100], [100, 100, 100, 100, 97, 100, 100], [100, 97, 97, 95, 86, 97, 100], [25, 11, 6, 4, 8, 6, 31]]
# Distribution des étapes pour l'apprentissage actif
# [0.08733333 0.08333333 0.09533333 0.09133333 0.15933333 0.48333333]

nb_step_early_stopping = 50 # Nombre d'epoch pendant lequels on verfie l'augmentation de l'accuracy
accuracy_early_stopping = 0.5 # Valeur d'accuracy à partir de laquelle on regarde l'early stopping

step_between_save = 50 # Nombre d'étape entre chaque sauvegarde des poids

load_weight = False # Chargement des poids pour poursuivre l'apprentissage (par exemple après un crash)
weight_file_name = None # Nom du fichier à charger
epoch_done = None # Nombre d'epoch déjà exécuté (on fera donc seuelemnt nb_epoch - epoch_done epoch d'apprentissage)
last_seed = None # Seed random pour l'apprentissage précédent afin d'avoir les mêmes valeurs.

batch_size = 10  # Taille des batchs utilisés pour le training

save_weight_at_best_epoch = True # Enregistrement des poids ou non lors de la meilleur accuracy sur le jeu de validation

nb_epochs = int(sys.argv[3])
train_set_size = 100000
test_set_size = 10000
validation_set_size = 1000
epochs_actif = 1  # Nombre d'epoch (entre chaque modification du jeu d'apprentissage par l'apprentissage actif)
latent_dim = 500  # Dimension latente des réseaux réccurent
operand_size = int(sys.argv[4]) # La taille maximal des opérandes
restricted = int(sys.argv[5])  # Si le resultat peut faire une taille 2 * operand_size ou seulement 2 * operand_size - 1
val_app = 0.5  # Valeur pour l'apprentissage actif

if not load_weight:
    random_seed = random.randint(0, 100000) # A définir pour la reproductibilité, A modifier si on veut tester plusieurs fois les mêmes paramètres
    start_epoch = 0
else:
    random_seed = last_seed
    start_epoch = epoch_done


with_transfer = 0
nb_step = operand_size + 1

if not restricted:
    max_operand = 10 ** operand_size
    result_size = 2 * operand_size
else:
    result_size = 2 * operand_size - 1
    max_operand = max_operand = int(10 ** operand_size * 0.31622776601683793319988935)
seq_size = nb_step * (result_size + 1)

name = "comp_step_{}digits_{}".format(result_size, random_seed)

if len(sys.argv) > 6:
    name += "_" + sys.argv[6]

log_name = "log_" + name + ".txt"
weight_name = "weight_" + name + ".h5"
weight_tmp_name = "weight_" + name + "_tmp.h5"
weight_best_name = "weight_" + name + "_best.h5"

if load_weight:
    open_way = "a+"
else:
    open_way = "w+"
with open(log_name, open_way) as l:
    l.write(
        "Log pour l'apprentissage de multiplication {} par {} chiffres avec résultat de {} chiffre en transfert \n\n\n\n".format(
            operand_size, operand_size, result_size))
    l.write("Avec une dimension latente de {}, et {} epochs\n".format(latent_dim, nb_epochs))
    l.write("et une random seed de {}\n\n".format(random_seed))


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 20))
encoder_inputs2 = Masking(mask_value=2)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs2)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 20))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(20, activation="sigmoid")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy',
              metrics=['accuracy'])

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

if load_weight:
    model.load_weights(weight_file_name)

def generate_data_addition(x, y, size=2, height=2):
    """
    Permet de générer les valeurs d'input et d'output pour l'addition de x et y
    :param x: premier opérande
    :param y: deuxième opérande
    :param size: largeur du support
    :param height: hauteur du support
    :return: valeur pour l'input et valeur pour l'output
    """
    x_str = str(x)
    y_str = str(y)
    encoder_in = []
    decoder_in = []
    for i in range(size):
        inp = []
        if i < len(x_str):
            inp.append(x_str[-i - 1])
        else:
            inp.append("0")
        if i < len(y_str):
            inp.append(y_str[-i - 1])
        else:
            inp.append("0")
        encoder_in.append(inp)
        if i == 0:
            decoder_in.append([(int(inp[0]) + int(inp[1])) // 10, (int(inp[0]) + int(inp[1])) % 10])
        else:
            # pass
            decoder_in.append([(int(inp[0]) + int(inp[1]) + int(decoder_in[-1][0])) // 10,
                               (int(inp[0]) + int(inp[1]) + int(decoder_in[-1][0])) % 10])
    encoder_in.append(["endl", "endl"])

    return encoder_in, decoder_in


def get_support_mult(x, y, size=4, max_len=2):
    """
    Permet d'obtenir le support pour la mutliplication complétement remplis pour pouvoir facilement récupérer
    les information d'input et d'output de chaque étape
    :param x: premier opérande
    :param y: deuxième opérande
    :param size: largeur du support
    :param max_len: nombre de chiffre maximal des opérande
    :return: le support intégralement rempli
    """
    support = np.zeros((2 * max_len + 4, size))
    x_str = str(x)
    y_str = str(y)
    for i in range(size):
        if i < len(x_str):
            support[0, -i - 1] = int(x_str[-1 - i])
        else:
            support[0, -i - 1] = 0
        if i < len(y_str):
            support[1, -i - 1] = int(y_str[-1 - i])
        else:
            support[1, -i - 1] = 0
    for k in range(max_len):
        for i in range(size):
            if i < k:
                support[(k + 1) * 2, -1 - i] = 0
                support[(k + 1) * 2 + 1, -1 - i] = 0
            elif (i - k) == 0:
                support[(k + 1) * 2, -1 - i] = support[0, - 1 - k] * support[1, - 1 - (i - k)] // 10
                support[(k + 1) * 2 + 1, -1 - i] = support[0, - 1 - k] * support[1, - 1 - (i - k)] % 10
            else:
                support[(k + 1) * 2, -1 - i] = (support[0, - 1 - k] * support[1, - 1 - (i - k)] + support[
                    (k + 1) * 2, - k - (i - k)]) // 10
                support[(k + 1) * 2 + 1, -1 - i] = (support[0, - 1 - k] * support[1, - 1 - (i - k)] + support[
                    (k + 1) * 2, - k - (i - k)]) % 10
        for i in range(size):
            val = 0
            for k in range(max_len):
                val += support[(k + 1) * 2 + 1, -1 - i]
            if i > 0:
                val += support[-2, -i]
            support[-1, -1 - i] = val % 10
            support[-2, -1 - i] = val // 10
    return support


def generate_data_multiplication(x, y, size=4, max_len=2):
    """
    Génère les valeurs d'input et d'output pour la mutliplication (avec les retenues pour chacunes des étapes)
    :param x: premier opérande
    :param y: deuxième opérande
    :param size: largeur du support
    :param max_len: nombre de chiffre max des opérande
    :return: liste des input et output pour la multiplication
    """
    x_str = str(x)
    y_str = str(y)
    support = get_support_mult(x, y, size, max_len)
    encoder_in_list = []
    decoder_in_list = []
    for i in range(max_len + 1):
        encoder_in = []
        decoder_in = []
        for k in range((i + 1) * (size + 1)):
            inp = []
            if k % (size + 1) == size:
                inp.append("endl")
                inp.append("endl")
            else:
                inp.append(int(support[(k // (size + 1)) * 2, - 1 - k % (size + 1)]))
                inp.append(int(support[(k // (size + 1)) * 2 + 1, - 1 - k % (size + 1)]))
            encoder_in.append(inp)
        for k in range(size):
            out = []
            out.append(int(support[(i + 1) * 2, - 1 - k]))
            out.append(int(support[(i + 1) * 2 + 1, - 1 - k]))
            decoder_in.append(out)
        # decoder_in.append(["endl", "endl"])
        encoder_in_list.append(encoder_in)
        decoder_in_list.append(decoder_in)

    return encoder_in_list, decoder_in_list


def get_acc(input_seq, output_seq, max_step, verbose=0):
    """
    Permet d'obtenir l'accuracy d'une étape donnée
    :param input_seq: sequence d'input de l'encoder
    :param output_seq: séquence d'output
    :param max_step: longueur max de la séquence d'output
    :param verbose: verbose
    :return: liste de 0 et 1 pour chaque caractère d'output, 0 ou 1 suivant si l'ensemble de la séquence et juste (1) ou non (0)
    """
    # Encode the input as state vectors.
    decoded_sentence, out = get_result(input_seq, max_step)
    acc = [0] * len(output_seq[0])
    result = 0
    for i in range(len(output_seq)):
        acc_tmp = [0] * len(output_seq[0])
        for index in range(len(output_seq[0])):
            if index < len(decoded_sentence) and output_seq[i][index][0] == decoded_sentence[index][0][i] and output_seq[i][index][1] == decoded_sentence[index][1][i]:
                acc_tmp[index] += 1
        acc = [a + b for a, b in zip(acc_tmp, acc)]
        if min(acc_tmp) == 1:
            result += 1
    if verbose:
        print(input_seq)
        print(output_seq)
        print(decoded_sentence)
    return acc, result


def get_result(input_seq, max_len=5, decimal=False):
    """
    Permet d'obtenir la séquence d'output pour une séquence d'input donnée
    :param input_seq: la séquence d'input
    :param max_len: longueur max de la séquence d'input
    :return: séquence de sortie sous forme de liste, liste des output du réseau
    """
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    batch_size = input_seq.shape[0]
    # Generate empty target sequence of length 1.
    if decimal:
        target_seq = np.zeros((batch_size, 1, 22))
    else:
        target_seq = np.zeros((batch_size, 1, 20))
    # Populate the first character of target sequence with the start character.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1)00.
    stop_condition = False
    decoded_sentence = []
    out = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        if sum(output_tokens[0, -1]) > 5:
            stop_condition = True
        else:
            if decimal:
                sampled_token_index = np.argmax(output_tokens[:, -1, :11], -1)
                sampled_token_index2 = np.argmax(output_tokens[:, -1, 11:], -1)
            else:
                sampled_token_index = np.argmax(output_tokens[:, -1, :10], -1)
                sampled_token_index2 = np.argmax(output_tokens[:, -1, 10:], -1)
            # sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence.append([sampled_token_index, sampled_token_index2])
            out.append(output_tokens)
            if decimal:
                target_seq = np.zeros((batch_size, 1, 22))
                target_seq[range(batch_size), 0, sampled_token_index] = 1.
                target_seq[range(batch_size), 0, sampled_token_index2 + 11] = 1.
            else:
                target_seq = np.zeros((batch_size, 1, 20))
                target_seq[range(batch_size), 0, sampled_token_index] = 1.
                target_seq[range(batch_size), 0, sampled_token_index2 + 10] = 1.

        # Exit condition: either hit max length
        # or find stop character.
        if (len(decoded_sentence) > max_len):
            stop_condition = True

        # Update the target sequence (of length 1).


        # Update states
        states_value = [h, c]

    return decoded_sentence, out


def get_final_result(list_x, list_y, step, size, fixed_len=None, max_result=20, decimal=False):
    """
    Permet d'obtenir le résultat final d'une multiplication à paritir des 2 opérandes (et du nombre d'étapes, de la taille du support)
    :param x: premier opérande
    :param y: deuxième opérande
    :param step: nombre d'étape de calcul (par exemple 3 pour une multiplication 2 chiffres par 2 chiffres)
    :param size: Largeur du support
    :param fixed_len: Si l'input doit avoir une taille fixe (si entraîné avec du padding, parfois nécéssaire)
    :param max_result: longueur max de la séquence d'output
    :return: La liste des résultats de chaque étape et le résultat final
    """
    if decimal:
        in_len = 22
        char_len = 11
    else:
        in_len = 20
        char_len = 10
    if fixed_len is not None:
        input_sequence = np.zeros((len(list_x), fixed_len, in_len))
    else:
        input_sequence = np.zeros((len(list_x), size + 1, in_len))
    result_list = []
    for x, y, index in zip(list_x, list_y, range(len(list_x))):
        x_str = str(x)
        y_str = str(y)
        for i in range(size):
            if i < len(x_str):
                if decimal and x_str[- 1 - i] == ".":
                    input_sequence[index, - (size + 1) + i, 10] = 1
                else:
                    input_sequence[index, - (size + 1) + i, int(x_str[- 1 - i])] = 1
            else:
                input_sequence[index, - (size + 1) + i, 0] = 1
            if i < len(y_str):
                if decimal and y_str[- 1 - i] == ".":
                    input_sequence[index, - (size + 1) + i, 21] = 1
                else:
                    input_sequence[index, - (size + 1) + i, char_len + int(y_str[- 1 - i])] = 1
            else:
                input_sequence[index, - (size + 1) + i, char_len] = 1
    input_sequence[:, -1] = np.ones(in_len)
    old_input_sequence = input_sequence
    # input_sequence[1] = input_sequence[0]
    result = get_result(input_sequence, max_result, decimal=decimal)[0]
    result_list.append(result)
    for k in range(1, step):
        if fixed_len is not None:
            input_sequence = np.zeros((len(list_x), fixed_len, in_len))
        else:
            input_sequence = np.zeros((len(list_x), (size + 1) * (k + 1), in_len))
        input_sequence[:, -(size + 1) * (k + 1): - (size + 1)] = old_input_sequence[:, - (size + 1) * k:]
        for i in range(size):
            if i < len(result):
                input_sequence[range(len(list_x)), - (size + 1) + i, result[i][0]] = 1
                input_sequence[range(len(list_x)), - (size + 1) + i, char_len + result[i][1]] = 1
        input_sequence[:, -1] = np.ones(in_len)
        old_input_sequence = input_sequence
        result = get_result(input_sequence, max_result, decimal=decimal)[0]
        result_list.append(result)
    val = [""] * len(list_x)
    for i in reversed(result_list[-1]):
        for k in range(len(list_x)):
            if i[1][k] < 10:
                val[k] += str(i[1][k])
            else:
                val[k] += "."
    return result_list, val


def get_specific_data(encoder_in_, decoder_in_, decoder_out_, prob, nb_step, batch=True):
    """
    Permet d'obtenir une seule étape pour chaque calcul au lieu de toute selon la distribution passé en paramètre
    :param data: le jeu de données avec toutes les étapes de tous les calcul
    :param prob: la distribution à utiliser
    :param nb_step: le nombre d'étape possible
    :return: Le nouveau jeu de données
    """
    # print(choice)
    if batch:
        encoder_in = np.zeros((encoder_in_.shape[1], encoder_in_.shape[2], encoder_in_.shape[3]))
        decoder_in = np.zeros((decoder_in_.shape[1], decoder_in_.shape[2], decoder_in_.shape[3]))
        decoder_out = np.zeros((decoder_out_.shape[1], decoder_out_.shape[2], decoder_out_.shape[3]))
        size = encoder_in_.shape[1]
    else:
        encoder_in = []
        decoder_in = []
        decoder_out = []
        size = len(encoder_in_)
    choice = np.random.choice(range(nb_step), size, p=prob)
    for i in range(size):
        if batch:
            encoder_in[i] = encoder_in_[choice[i], i]
            decoder_in[i] = decoder_in_[choice[i], i]
            decoder_out[i] = decoder_out_[choice[i], i]
        else:
            encoder_in.append(encoder_in_[i][choice[i]])
            decoder_in.append(decoder_in_[i][choice[i]])
            decoder_out.append(decoder_out_[i][choice[i]])
    return encoder_in, decoder_in, decoder_out


def get_step_acc(input_seq, output_seq, max_step, nb_data, verbose=0, seq_index=0):
    acc = [0] * input_seq.shape(0)
    for i in range(len(acc)):
        for k in range(nb_data):
            acc[i] += get_acc(input_seq[i, k: k + 1], output_seq[i, k], max_step, verbose=0, seq_index=0)
    return acc


def generate_data(list_x, list_y, nb_step, seq_in, seq_out, max_len, size, addition=False, batch=True, in_size=20,
                  add_total=0, space=False):
    """
    Permet de générer les données nécessaire pour l'apprentissage dans le cas ou on lit les chiffres 2 par 2 et
    qu'on écrit les retenues
    :param list_x: Liste des premiers opérandes
    :param list_y: liste des seconds opérandes
    :param nb_step: Nombre d'étapes de l'opération (utile pour la multiplication)
    :param seq_in: Taille de la sequence d'input
    :param seq_out: Taille de la séquence d'output
    :param max_len: Nombre de chiffres max des opérandes
    :param size: Largeur du support
    :param addition: SI c'est une addition
    :param batch: Si on veux pouvoir faire de l'entraînement par batch, il faut que toutes les sequences fassent la même taille.
    On doit donc ajouter du padding au départ des séquences les plus courtes pour les ralonger.
    Si True : le retour sera simplement une np array, sinon ce sera des listes
    :param in_size: Taille de l'input
    :param add_total: 0 par defaut, 1 si on veut ajouter une étape supplémentaire pour la prédiciton finale sans les étapes intermédiares
    :param space: Si add_total=1, permet de choisir si la prédicion finale note les retenues de l'addition (False) ou non (True)
    :return:
    """
    encode_in_list = []
    decode_in_list = []
    if addition:
        encoder_input = np.zeros((len(list_x), seq_in, in_size))
        decoder_input = np.zeros((len(list_x), seq_out, in_size))
        decoder_output = np.zeros((len(list_x), seq_out, in_size))
    else:
        if batch:
            encoder_input = 2 * np.ones((nb_step + add_total, len(list_x), seq_in, in_size), dtype=np.uint8)
            for x in range(nb_step):
                encoder_input[x, :, - (x + 1) * (size + 1):, :] = np.zeros((1, len(list_x), (x + 1) * (size + 1), in_size))
            if add_total == 1:
                encoder_input[-1, :, :, :] = np.zeros((1, len(list_x), seq_in, in_size), dtype=np.uint8)
            decoder_input = np.zeros((nb_step + add_total, len(list_x), seq_out, in_size), dtype=np.uint8)
            decoder_output = np.zeros((nb_step + add_total, len(list_x), seq_out, in_size), dtype=np.uint8)
        else:
            encoder_input = []
            decoder_input = []
            decoder_output = []
    for x, y, index in zip(list_x, list_y, range(len(list_x))):
        if addition:
            e, d = generate_data_addition(x, y, size)
            if not batch:
                encoder_input.append(np.zeros((1, len(e), 20)))
                decoder_input.append(np.zeros((1, len(d) + 1, 20)))
                decoder_output.append(np.zeros((1, len(d) + 1, 20)))
            encode_in_list.append(e)
            decode_in_list.append(d)
            for i, v in enumerate(e):
                if v[0] == "endl":
                    if batch:
                        encoder_input[index, i] = np.ones(20)
                    else:
                        encoder_input[index][0, i] = np.ones(20)
                else:
                    if batch:
                        encoder_input[index, i, int(v[0])] = 1
                        encoder_input[index, i, int(v[1]) + 10] = 1
                    else:
                        encoder_input[index][0, i, int(v[0])] = 1
                        encoder_input[index][0, i, int(v[1]) + 10] = 1

            for i, v in enumerate(d):
                if batch:
                    decoder_input[index, i + 1, int(v[0])] = 1
                    decoder_input[index, i + 1, int(v[1]) + 10] = 1
                    decoder_output[index, i, int(v[0])] = 1
                    decoder_output[index, i, int(v[1]) + 10] = 1
                else:
                    decoder_input[index][0, i + 1, int(v[0])] = 1
                    decoder_input[index][0, i + 1, int(v[1]) + 10] = 1
                    decoder_output[index][0, i, int(v[0])] = 1
                    decoder_output[index][0, i, int(v[1]) + 10] = 1
            if batch:
                decoder_output[index, -1] = np.ones(20)
            else:
                decoder_output[index][0, -1] = np.ones(20)
        else:
            en, de = generate_data_multiplication(x, y, size=size, max_len=max_len)
            encode_in_list.append(en)
            decode_in_list.append(de)
            if not batch:
                encoder_input.append([])
                decoder_input.append([])
                decoder_output.append([])
            for e, d, k in zip(en, de, range(len(en))):
                if not batch:
                    encoder_input[index].append(np.zeros((1, len(e), 20)))
                    decoder_input[index].append(np.zeros((1, len(d) + 1, 20)))
                    decoder_output[index].append(np.zeros((1, len(d) + 1, 20)))
                pad = seq_in - len(e)
                for i, v in enumerate(e):
                    if v[0] == "endl":
                        if batch:
                            if add_total == 1 and k == 0:
                                encoder_input[-1, index, i] = np.ones(20)
                            encoder_input[k, index, pad + i] = np.ones(20)
                        else:
                            encoder_input[index][k][0, i] = np.ones(20)
                    else:
                        if batch:
                            if add_total == 1 and k == 0:
                                encoder_input[-1, index, i, int(v[0])] = 1
                                encoder_input[-1, index, i, int(v[1]) + 10] = 1
                            encoder_input[k, index, pad + i, int(v[0])] = 1
                            encoder_input[k, index, pad + i, int(v[1]) + 10] = 1
                        else:
                            encoder_input[index][k][0, i, int(v[0])] = 1
                            encoder_input[index][k][0, i, int(v[1]) + 10] = 1

                for i, v in enumerate(d):
                    if batch:
                        if add_total == 1 and k == (nb_step - 1):
                            if not space:
                                decoder_input[-1, index, i + 1, int(v[0])] = 1
                                decoder_output[-1, index, i, int(v[0])] = 1
                            decoder_input[-1, index, i + 1, int(v[1]) + 10] = 1
                            decoder_output[-1, index, i, int(v[1]) + 10] = 1

                        decoder_input[k, index, i + 1, int(v[0])] = 1
                        decoder_input[k, index, i + 1, int(v[1]) + 10] = 1
                        decoder_output[k, index, i, int(v[0])] = 1
                        decoder_output[k, index, i, int(v[1]) + 10] = 1
                    else:
                        decoder_input[index][k][0, i + 1, int(v[0])] = 1
                        decoder_input[index][k][0, i + 1, int(v[1]) + 10] = 1
                        decoder_output[index][k][0, i, int(v[0])] = 1
                        decoder_output[index][k][0, i, int(v[1]) + 10] = 1
                if batch:
                    decoder_output[k, index, -1] = np.ones(20)
                else:
                    decoder_output[index][k][0, -1] = np.ones(20)
            decoder_output[-1, index, -1] = np.ones(20)
            for z in range(nb_step):
                encoder_input[-1, index, (z + 1) * (seq_in // nb_step) - 1] = np.ones(20)
    return encode_in_list, decode_in_list, encoder_input, decoder_input, decoder_output


def get_transfer_result(list_x, list_y, size, leng):
    input_sequence = np.zeros((len(list_x), size, 20))
    for x, y, index in zip(list_x, list_y, range(len(list_x))):
        x_str = str(x)
        y_str = str(y)
        for i in range(leng):
            if i < len(x_str):
                input_sequence[index, i, int(x_str[- 1 - i])] = 1
            else:
                input_sequence[index, i, 0] = 1
            if i < len(y_str):
                input_sequence[index, i, 10 + int(y_str[- 1 - i])] = 1
            else:
                input_sequence[index, i, 10] = 1
    input_sequence[:, leng] = np.ones(20)
    for i in range(1, size//(leng + 1) + 1):
        input_sequence[:, i * (leng + 1) - 1] = np.ones(20)
    result = get_result(input_sequence, 20)[0]
    val = [""] * len(list_x)
    for i in reversed(result):
        for k in range(len(list_x)):
            if i[1][k] < 10:
                val[k] += str(i[1][k])
            else:
                val[k] += "."
    return val

np.random.seed(random_seed)

liste_A = []
liste_B = []
liste_C = []

while len(liste_A)<train_set_size:
    if (1+len(liste_A))%10000==0:
        print("Data :",1+len(liste_A))
    n1 = np.random.randint(max_operand)
    n2 = np.random.randint(max_operand)
    while (n1, n2) in liste_A:
        n1 = np.random.randint(max_operand)
        n2 = np.random.randint(max_operand)
    liste_A.append((n1, n2))
while len(liste_B)<test_set_size:
    if (1+len(liste_B))%1000==0:
        print("Data :", 1+len(liste_B))
    n1 = np.random.randint(max_operand)
    n2 = np.random.randint(max_operand)
    while (n1, n2) in liste_A or (n1, n2) in liste_B:
        n1 = np.random.randint(max_operand)
        n2 = np.random.randint(max_operand)
    liste_B.append((n1, n2))
while len(liste_C)<validation_set_size:
    if (1+len(liste_C))%1000==0:
        print("Data :", 1+len(liste_C))
    n1 = np.random.randint(max_operand)
    n2 = np.random.randint(max_operand)
    while (n1, n2) in liste_A or (n1, n2) in liste_B or (n1, n2) in liste_C:
        n1 = np.random.randint(max_operand)
        n2 = np.random.randint(max_operand)
    liste_C.append((n1, n2))
x_list = [x[0] for x in liste_A]
y_list = [x[1] for x in liste_A]


encode_in, decode_in_list, encoder_input, decoder_input, decoder_output = generate_data(x_list, y_list, nb_step, seq_size, result_size + 1, operand_size, result_size, add_total=with_transfer)

x_list_test = [x[0] for x in liste_B]
y_list_test = [x[1] for x in liste_B]
encode_in_test, decode_in_list_test, encoder_input_test, decoder_input_test, decoder_output_test = generate_data(x_list_test, y_list_test, nb_step, seq_size, result_size + 1, operand_size, result_size, add_total=with_transfer)

x_list_validation = [x[0] for x in liste_C]
y_list_validation = [x[1] for x in liste_C]
encode_in_validation, decode_in_list_validation, encoder_input_validation, decoder_input_validation, decoder_output_validation = generate_data(x_list_validation, y_list_validation, nb_step, seq_size, result_size + 1, operand_size, result_size, add_total=with_transfer)

accuracy_queue = deque(maxlen=nb_step_early_stopping)

din_list = np.array(decode_in_list)
din_list_test = np.array(decode_in_list_test)
din_list_validation = np.array(decode_in_list_validation)
batch = True
accuracy_evolution = []
attention = []
accuracy_evolution_train = []
accuracy_evolution_test = []
prob_uniforme = np.full(nb_step + with_transfer, 1 / (nb_step + with_transfer))
proba = prob_uniforme

best_accuracy = 0
best_accuracy_epoch = 0

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times = time.time() - self.epoch_time_start
time_callback = TimeHistory()

for i in range(start_epoch, nb_epochs):
    print(i)
    e, di, do = get_specific_data(encoder_input, decoder_input, decoder_output, proba, nb_step + with_transfer, batch=True)
    e, di, do = sklearn.utils.shuffle(e, di, do)
    model.fit([e, di], do,
              batch_size=batch_size,
              epochs=1, verbose=1, callbacks=[time_callback])
    print(int(time_callback.times))
    val_train = [random.randint(0,train_set_size - 1) for i in range(1000)]
    acc_list_test = []
    acc_list_train = []
    detail_list_train = []
    detail_list_test = []
    accuracy_validation = 0
    for ik in range(nb_step + with_transfer):
        if ik < nb_step:
            rep, acc_train = get_acc(encoder_input[ik, val_train], din_list[val_train, ik], result_size + 1)
        else:
            rep, acc_train = get_acc(encoder_input[ik, val_train], din_list[val_train, nb_step - 1], result_size + 1)
        acc_list_train.append(acc_train)
        detail_list_train.append(rep)
        if ik < nb_step:
            rep, acc_validation = get_acc(encoder_input_validation[ik], din_list_validation[range(din_list_validation.shape[0]), ik], result_size + 1)
        else:
            rep, acc_validation = get_acc(encoder_input_validation[ik], din_list_validation[range(din_list_validation.shape[0]), nb_step - 1], result_size + 1)
        acc_list_test.append(acc_validation)
        detail_list_test.append(rep)
    x_list = [x[0] for x in liste_C]
    y_list = [x[1] for x in liste_C]
    a = get_final_result(x_list, y_list, nb_step, result_size)[1]
    b = [x * y for x, y in liste_C]
    a = [int(x) for x in a]
    accuracy_validation = sum([x == y for x, y in zip(a, b)])/len(liste_C)
    print(acc_list_train)
    print(acc_list_test)
    print(detail_list_train)
    print(detail_list_test)
    if accuracy_validation > best_accuracy:
        best_accuracy = accuracy_validation
        best_accuracy_epoch = i
        if save_weight_at_best_epoch:
            model.save_weights(weight_best_name)
    accuracy_evolution_train.append(acc_list_train)
    accuracy_evolution_test.append(acc_list_test)
    error = [len(val_train) - a for a in acc_list_train]
    if np.sum(np.array(error)) == 0:
        proba = prob_uniforme
    else:
        error_prob = np.array(error) / np.sum(np.array(error))
        proba = val_app * error_prob + (1 - val_app) * prob_uniforme
    print(proba)

    with open(log_name, "a") as l:
        l.write(
            "\n\nEpoch : {} exécuté en {} secondes\n {} Mo de RAM utilisé\nAccuracy on train set : {}\nAccuracy on validation set : {}\nDetail Accuracy on train set : {}\nDetail Accuracy on validation set : {}\nActiv learning distribution : {}\nAccuracy on validation set for global operation (step by step) : {}\n\n".format(
                str(i), str(int(time_callback.times)),str(process.memory_info().rss/1000000), str(acc_list_train), str(acc_list_test),
                str(detail_list_train), str(detail_list_test), str(proba), str(accuracy_validation)))
    if i > nb_step_early_stopping and accuracy_validation > accuracy_early_stopping and i - best_accuracy_epoch > nb_step_early_stopping:
        break
    accuracy_queue.append(accuracy_validation)
    if (i + 1) % step_between_save == 0:
        model.save_weights(weight_tmp_name)

model.load_weights(weight_best_name)

with open(log_name, "a") as l:
    l.write("\n\nAccuracy evolution on train set : {}".format(str(accuracy_evolution_train)))
    l.write("\n\nAccuracy evolution on test set : {}\n\n\n".format(accuracy_evolution_test))
model.save_weights(weight_name)


acc_step = 0
for x in range(len(liste_B)// 1000):
    x_list = [x[0] for x in liste_B[x * 1000: (x + 1) * 1000]]
    y_list = [x[1] for x in liste_B[x * 1000: (x + 1) * 1000]]

    a = get_final_result(x_list, y_list, nb_step, result_size)[1]
    # print(a)
    b = [x * y for x,y in liste_B[x * 1000: (x + 1) * 1000]]
    a = [int(x) for x in a]
    acc_step += sum([x == y for x, y in zip(a, b)])
acc_step = acc_step / (len(liste_B) // 1000 * 1000)
print("\nAccuracy pour l'opération step by step : {}\n".format(str(acc_step)))
with open(log_name, "a") as l:
    l.write("\nAccuracy pour l'opération step by step : {}\n".format(str(acc_step)))


os.rename(log_name, "log_{}_{}.txt".format(name, str(acc_step)))
os.rename(weight_name, "weight_{}_{}.h5".format(name, str(acc_step)))
