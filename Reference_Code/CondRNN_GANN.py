import os
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline
from IPython.display import display

from .CondRNNGenerator import CondRNNGenerator
from .CondRNNDiscriminator import CondRNNDiscriminator


class CondRNN_GANN:
	cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

	def __init__(self, dataSet: tf.data.Dataset, batchSize: int, seqLength: int, numFeatures: int, latentDim: int,
				 conditionDim: int, lstmCellDim: int, learning_rate: float, trainEpochs: int, trainIter: int, chkPointLoc: str):
		"""
		Parameters
		----------
			:param dataSet: The dataset object used to access the data, must implement TensorFlow Dataset API
			:param batchSize: The number of input values that will be processed in a batch
			:param seqLength: The length of the input sequence, i.e. the number of samples in the time series
			:param numFeatures: The number of features that each sample in the batch contains of size seqLength
			:param latentDim: The number of dimensions in the latent variable used as input for synthetic generation
			:param conditionDim: The number of dimensions the condition variable is to have
			:param lstmCellDim: The number of hidden layers the lstm will have when the model is constructed
			:param learning_rate: The learning rate for the optimization of the weights in the model
			:param trainEpochs: Number of training epochs to execute per call to train
			:param trainIter: Number of iterations of training per epoch.
			:param chkPointLoc: The location to store checkpoints for this GANN.
		"""

		self._dataIterator = iter(dataSet)
		self._trainEpochs = trainEpochs
		self._batchSize = batchSize
		self._learning_rate = learning_rate
		self._trainIter = trainIter
		self._seqLength = seqLength
		self._numFeatures = numFeatures
		self._latentDim = latentDim
		self._conditionDim = conditionDim
		self.loss_output_dir = chkPointLoc
		self.reg = 0.01
		self.G_round = 1
		self.D_round = 5


		# Construct a generator class and the output sample tensor
		self._generator = CondRNNGenerator(seqLength, numFeatures, latentDim,
										   conditionDim, lstmCellDim)

		# Construct a discriminator class and the real/generated data descriminator output
		self._discriminator = CondRNNDiscriminator(seqLength, numFeatures, conditionDim, lstmCellDim)

		# self._generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
		self._generator_optimizer = tf.keras.optimizers.Adam()
		# self._discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
		self._discriminator_optimizer = tf.keras.optimizers.SGD(self._learning_rate)
        
		if not os.path.exists(chkPointLoc):
			os.mkdir(chkPointLoc)

		self._chkpnt = tf.train.Checkpoint(step=tf.Variable(1), generator_optimizer=self._generator_optimizer,
										   discriminator_optimizer=self._generator_optimizer, generator=self._generator,
										   discriminator=self._discriminator)
		self._chkPointMgr = tf.train.CheckpointManager(self._chkpnt, chkPointLoc, max_to_keep=None) #  max_to_keep=3

		self._chkpnt.restore(self._chkPointMgr.latest_checkpoint)
		if self._chkPointMgr.latest_checkpoint:
			print("Restored from {}".format(self._chkPointMgr.latest_checkpoint))
		else:
			print("Initializing from scratch.")

	def get_generated_sample(self, condition_val):
		gen_values = self._generator(self._get_fake_samples(), training=False)
		return gen_values

	def train_bak(self):
		trace_idx = self.loss_output_dir.split("_")[2]
		f_dir = self.loss_output_dir.split("_")[0][:-5] + '/loss_traces/loss_trace_' + trace_idx + '.txt'
		print(f_dir)
		f_out = open(f_dir, 'w+')
		f_out.write("step\tgen_loss\tdisc_loss\n")
		for step in range(self._trainEpochs):
			disc_loss = None
			for i, sample_batch in enumerate(self._dataIterator):
				if i > self._trainIter:
					break
				# Slice off the last column of each sample, it is assumed to be the condition label
				condition_vals = tf.slice(sample_batch, [0, 0, self._numFeatures], [-1, -1, self._conditionDim])
				sliced_sample = tf.slice(sample_batch, [0, 0, 0], [-1, -1, self._numFeatures])
				gen_loss, disc_loss = self._train_step(sliced_sample, condition_vals)

				self._chkpnt.step.assign_add(1)
			self._chkpnt.step.assign_add(1)
			if step % 10 == 0:
				save_path = self._chkPointMgr.save()
				print("Saved checkpoint for step {}: {}".format(int(self._chkpnt.step), save_path))
			if disc_loss is not None:
				print("** step - {:d} : gen loss= {:1.2f}, disc loss= {:1.2f}, chkpnt = {:d}".format(step, gen_loss.numpy(), disc_loss.numpy(), int(self._chkpnt.step)))
				f_out.write(str(step) + '\t' + str(gen_loss.numpy()) + '\t' + str(disc_loss.numpy()) + '\n')
				f_out.flush()
		f_out.close()
    
	def train(self):
		trace_idx = self.loss_output_dir.split("_")[2]
		f_dir = self.loss_output_dir.split("_")[0][:-5] + 'loss_traces/loss_trace_' + trace_idx + '.txt'
		print(f_dir)
		f_out = open(f_dir, 'w+')
		f_out.write("step\tgen_loss\tdisc_loss\n")
		for step in range(self._trainEpochs):
			# train
			gen_loss, disc_loss = self._train_step(step)
			self._chkpnt.step.assign_add(1)
			if step % 5 == 0:
				save_path = self._chkPointMgr.save()
				print("Saved checkpoint for step {}: {}".format(int(self._chkpnt.step), save_path))
			if disc_loss is not None:
				print("** step - %d : gen loss= %1.2f, disc loss= %1.2f"%(step, gen_loss.numpy(), disc_loss.numpy()))
				f_out.write(str(step) + '\t' + str(gen_loss.numpy()) + '\t' + str(disc_loss.numpy()) + '\n')
				f_out.flush()
		f_out.close()
        
	def _get_real_samples(self): 
		# Slice off the last column of each sample, it is assumed to be the condition label
		sample_batch = self._dataIterator.get_next()
		condition_vals = tf.slice(sample_batch, [0, 0, self._numFeatures], [-1, -1, self._conditionDim])
		sliced_sample = tf.slice(sample_batch, [0, 0, 0], [-1, -1, self._numFeatures])
# 		for i in range(self._batchSize):    
# 			plt.plot(range(self._seqLength), sliced_sample[i])
# 			plt.show()
		return sliced_sample, condition_vals
    
	def _get_fake_samples(self):
		latent_var = tf.random.normal([self._batchSize, self._seqLength, self._latentDim])
		# latent_var = tf.random.uniform([self._batchSize, self._seqLength, self._latentDim], minval=-1, maxval=1) # uniform
		# each step of the generator takes one time step of the signal to evaluate +
		# its conditional embedding, so we need to concat the condition value onto our latent variable tensor
		condition_vals = tf.ones([self._batchSize, self._seqLength, self._conditionDim])
		latent_var_input = tf.concat([latent_var, condition_vals], 2)
		# latent_var_input: batch_size * seqLength * (latentDim+1)
		return latent_var_input
        
	def _train_step(self, curr_step):
		for i in range(self._trainIter):
			# train disc
			for j in range(self.D_round):
				with tf.GradientTape() as disc_tape:
					generated_samples = self._generator(self._get_fake_samples(), training=True)
					fake_output = self._discriminator(generated_samples, training=True)
					real_output = self._discriminator(self._get_real_samples(), training=True)
					disc_loss = self._discriminator_loss(real_output, fake_output)
# 				var_names = [v.name for v in self._discriminator.trainable_variables()]
				gradients_of_discriminator = disc_tape.gradient(disc_loss, self._discriminator.trainable_variables)
# 				self._discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))
# 				tf.keras.optimizers.SGD(self._learning_rate).apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))
				tf.keras.optimizers.Adam(self._learning_rate).apply_gradients(zip(gradients_of_discriminator, self._discriminator.trainable_variables))
				# tf.keras.optimizers.SGD(self._learning_rate).minimize(disc_loss_curr, self._discriminator.trainable_variables)

			# train gen
			for j in range(self.G_round):
				with tf.GradientTape() as gen_tape:
					generated_samples = self._generator(self._get_fake_samples(), training=True)
					fake_output = self._discriminator(generated_samples, training=True)
					gen_loss = self._generator_loss(fake_output)
# 				var_names = [v.name for v in self._generator.trainable_variables()]
# 				print(var_names)
				gradients_of_generator = gen_tape.gradient(gen_loss, self._generator.trainable_variables)
# 				self._generator_optimizer.apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))
				tf.keras.optimizers.Adam(self._learning_rate).apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))
# 				tf.keras.optimizers.Adam().minimize(gen_loss_curr, self._generator.trainable_variables)# apply_gradients(zip(gradients_of_generator, self._generator.trainable_variables))

		if curr_step%3 ==0:
			print("Viewing samples at %d:"%(curr_step))
			print(generated_samples.shape)
			generated_samples = self._generator(self._get_fake_samples(), training=False)
			fig, ax = plt.subplots(2, 3, figsize=(13, 6))
			for i in range(2):
				for j in range(3):
					ax1 = ax[i, j]
					ax1.plot(range(self._seqLength), generated_samples[np.random.randint(0, self._batchSize), :, 0], c='r', lw=0.5)
			plt.show()
		return gen_loss, disc_loss

	def _discriminator_loss(self, real_output, fake_output):
		real_loss = CondRNN_GANN.cross_entropy(tf.ones_like(real_output), real_output)
		fake_loss = CondRNN_GANN.cross_entropy(tf.zeros_like(fake_output), fake_output)
# 		real_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output))
# 		fake_loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output))
		total_loss = real_loss + fake_loss # + reg_loss
		return total_loss

	def _generator_loss(self, fake_output):
		# print("cal gen loss ...")
# 		loss = tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output))
		loss = CondRNN_GANN.cross_entropy(tf.ones_like(fake_output), fake_output)
		# loss = tf.math.reduce_mean(-tf.math.log(tf.clip_by_value(fake_output, -0.7, 0.7)))
		# reg_loss = 0.5 * self.reg * np.sum(self._generator._weight_out ** 2)
		return loss #  + reg_loss

		return gen_loss, disc_loss

	def save_checkpoint(self):
		save_path = self._chkPointMgr.save()
		print("Saved checkpoint for step {}: {}".format(int(self._chkpnt.step), save_path))

	def load_checkpoint(self, path=None):
		if path is None:
			self._chkpnt.restore(self._chkPointMgr.latest_checkpoint)
			if self._chkPointMgr.latest_checkpoint:
				print("Restored from {}".format(self._chkPointMgr.latest_checkpoint))
			else:
				print("Initializing from scratch.")
		else:
			self._chkpnt.restore(path)# .assert_consumed()
			print("Checkpoint {} Restored".format(path))
