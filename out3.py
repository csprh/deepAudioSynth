import tensorflow as tf
from musicnn_keras import configuration as config
from musicnn_keras.extractor import batch_data
import librosa
import numpy as np
from IPython.display import Image, display
import sounddevice as sd

model = 'MTT_musicnn'
input_length = 3
input_overlap = False
extract_features = True

file_name = './audio/joram-moments_of_clarity-08-solipsism-59-88.mp3'
labels = config.MTT_LABELS
num_classes = len(labels)

try:
    keras_model = tf.keras.models.load_model('./musicnn_keras/keras_checkpoints/{}.h5'.format(model))
except:
    raise ValueError('Unknown model')

# select labels
if 'MTT' in model:
    labels = config.MTT_LABELS
elif 'MSD' in model:
    labels = config.MSD_LABELS

if 'vgg' in model and input_length != 3:
    raise ValueError('Set input_length=3, the VGG models cannot handle different input lengths.')

# convert seconds to frames
n_frames = librosa.time_to_frames(input_length, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP) + 1
if not input_overlap:
    overlap = n_frames
else:
    overlap = librosa.time_to_frames(input_overlap, sr=config.SR, n_fft=config.FFT_SIZE, hop_length=config.FFT_HOP)

feature_nums = 50
#feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

batch, spectrogram = batch_data(file_name, n_frames, overlap)
batch1 = batch[0, :, :]

print('Total number of feature channels:', feature_nums)
# start with a gray image with a little noise
upper = np.max(batch1)
img_noise = np.random.uniform(high = upper, size = (187, 96))
thisInput = np.float32(img_noise)

thisInput = batch1#batch1 = thisInput
total_variation_weight = 0.000001
#layerOut = y
#channel = 0

#names = ['tf_op_layer_Max','tf_op_layer_moments/Squeeze','tf_op_layer_transpose_2','tf_op_layer_transpose_1','tf_op_layer_transpose','tf_op_layer_concat','dense_1']
names = ['tf_op_layer_concat']

#names = ['dense_1']
layers = [keras_model.get_layer(name).output for name in names]

# Create the feature extraction model
#dream_model = tf.keras.Model(inputs=keras_model.input, outputs=layers)
dream_model = tf.keras.Model(keras_model.input, layers)


def calc_loss(thisInput, model):
  channel = 3
  # Pass forward the image through the model to retrieve the activations.
  # Converts the image into a batch of size 1.
  input_batch = tf.expand_dims(thisInput, axis=0)
  input_batch = tf.expand_dims(input_batch, 3)
  layer_activations = model(input_batch)
  this_class = layer_activations[0,408:,0]
  #print ("Layer_activtations{}  a".format(np.shape(this_class)))
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act[0])
    #loss = tf.math.abs(this_class)

    losses.append(loss)
  #return this_class
  #return  tf.reduce_sum(tf.math.abs(this_class))
  return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
  def __init__(self, model):
    self.model = model

  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  def __call__(self, thisInput, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          # This needs gradients relative to `img`
          # `GradientTape` only watches `tf.Variable`s by default
          tape.watch(thisInput)
          loss = calc_loss(thisInput, self.model)
          loss = loss+total_variation_weight*tf.image.total_variation(tf.expand_dims(thisInput, axis=2))

        # Calculate the gradient of the loss with respect to the pixels of the input image.
        gradients = tape.gradient(loss, thisInput)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
        # You can update the image by directly adding the gradients (because they're the same shape!)
        thisInput = thisInput + gradients*step_size
        #thisInput = tf.clip_by_value(thisInput, 0, 1.5*upper)
        print ("Step {}, loss {}, step_size {}".format(n, loss, step_size))
      return loss, thisInput

deepdream = DeepDream(dream_model)

def run_deep_dream_simple(thisInput, steps=100, step_size=0.005):
  # Convert from uint8 to the range expected by the model.

  thisInput = tf.convert_to_tensor(thisInput)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, thisInput = deepdream(thisInput, run_steps, tf.constant(step_size))

    #display.clear_output(wait=True)
    #show(deprocess(img))
    print ("Step {}, loss {}, stepsize {}".format(step, loss, step_size))


  result = thisInput

  return result

dream_img = run_deep_dream_simple(thisInput,100, 0.005)

#dream_img = batch[0,:,:]
audio_rep = (np.power(10.0, dream_img)-1.0)/10000.0


audio_out1 = librosa.feature.inverse.mel_to_audio(M=audio_rep.T,
                                               sr=config.SR,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE)
batch1 = batch[0,:,:]

batch1Out = (np.power(10.0, batch1)-1.0)/10000.0

audio_out2 = librosa.feature.inverse.mel_to_audio(M=batch1Out.T,
                                               sr=config.SR,
                                               hop_length=config.FFT_HOP,
                                               n_fft=config.FFT_SIZE)

librosa.output.write_wav('audio_out1.wav', audio_out1, config.SR)
librosa.output.write_wav('audio_out2.wav', audio_out2, config.SR)
