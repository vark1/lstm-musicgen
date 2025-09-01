import numpy as np
import math

from music21 import stream, midi, tempo
from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, ZeroPadding2D, Dense, LSTM, RepeatVector

from preprocess import get_corpus_data, get_musical_data
from qa import prune_grammar, prune_notes, clean_up_notes
from grammar import unparse_grammar
from model import predict_and_sample

# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv2D):
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPooling2D):
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        if (type(layer) == ZeroPadding2D):
            descriptors.append(layer.padding)
        if (type(layer) == Dense):
            descriptors.append(layer.activation.__name__)
        if (type(layer) == LSTM):
            descriptors.append(layer.input_shape)
            descriptors.append(layer.activation.__name__)
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
        result.append(descriptors)
    return result

def load_music_utils(file_path):

    chords, abstract_grammars = get_musical_data(file_path)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)

# converts a midi note to freq. using the standard 440Hz concertA pitch
def nmidi_to_freq(note, concert_A=440.0):
  return (2.0 ** ((note - 69) / 12.0)) * concert_A

def ticks_to_ms(ticks, tempo, mid):
    tick_ms = math.ceil((60000.0 / tempo) / mid.ticks_per_beat)
    return ticks * tick_ms

def mid2wav(file):
    mid = MidiFile(file)
    output = AudioSegment.silent(mid.length * 1000.0)
    tempo = 130

    for track in mid.tracks:
        # position of rendering in ms
        current_pos = 0.0
        current_notes = defaultdict(dict)

        for msg in track:
            current_pos += ticks_to_ms(msg.time, tempo, mid)
            if msg.type == 'note_on':
                if msg.note in current_notes[msg.channel]:
                    current_notes[msg.channel][msg.note].append((current_pos, msg))
                else:
                    current_notes[msg.channel][msg.note] = [(current_pos, msg)]


            if msg.type == 'note_off':
                start_pos, start_msg = current_notes[msg.channel][msg.note].pop()

                duration = math.ceil(current_pos - start_pos)
                # signal_generator = Sine(nmidi_to_freq(msg.note, 500))
                signal_generator = Sine(nmidi_to_freq(msg.note))
                rendered = signal_generator.to_audio_segment(duration=duration-50, volume=-20).fade_out(100).fade_in(30)

                output = output.overlay(rendered, start_pos)

    output.export("./output/rendered.wav", format="wav")

# cut the corpus into semi-redundant sequences of Tx values
def data_processing(corpus, values_indices, m = 60, Tx = 30):
    Tx = Tx 
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=bool)
    Y = np.zeros((m, Tx, N_values), dtype=bool)
    for i in range(m):
        random_idx = np.random.choice(len(corpus)-Tx)
        corp_data = corpus[random_idx:(random_idx+Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j!=0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = np.swapaxes(Y,0,1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values 

# generates an audio stream to save music using a trained model
def generateAudioStream(inference_model, indices_tones, chords):

    n_a = 64
    x_init = np.zeros((1, 1, 90))
    a_init = np.zeros((1, n_a))
    c_init = np.zeros((1, n_a))

    out_stream = stream.Stream()
    
    offset = 0.0
    num_chords = int(len(chords)/3)     # num of diff set of chords
    for i in range(1, num_chords):
        curr_chords = stream.Voice()

        for j in chords[i]:
            curr_chords.insert((j.offset % 4), j)
        
        _, indices = predict_and_sample(inference_model, x_init, a_init, c_init)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        predicted_tones += pred[-1]

        # post processing
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')
        predicted_tones = prune_grammar(predicted_tones)    # smoothing measure
        sounds = unparse_grammar(predicted_tones, curr_chords)
        sounds = prune_notes(sounds)        # removing repeated and too close together sounds
        sounds = clean_up_notes(sounds)     # qa: cleanup sounds

        for m in sounds:
            out_stream.insert(offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(offset + mc.offset, mc)

        offset += 4.0
        
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    return out_stream

X, Y, num_unique_musical_val, indices_dict, chords = load_music_utils('data/training_sample_midi.mid')