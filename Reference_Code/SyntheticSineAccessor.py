import numpy as np
import pandas as pd

class SyntheticSineAccessor:

	def __init__(self, seqLength, numFeatures):

		self._seqLength = seqLength
		self._numFeatures = numFeatures

	def __iter__(self):
		return self

	def __next__(self):
		"""
		Returns the next data item and its label
		:rtype: tuple (data, label)
		"""
		dataItem = self._get_sine_wave() # (self._get_sine_wave(), 1)

		return dataItem

	def _get_sine_wave(self,  freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, **kwargs):
		ix = np.arange(self._seqLength) + 1
		# signals = []
		# for i in range(self._numFeatures):
		f = np.random.uniform(low=freq_high, high=freq_low)  # frequency
		A = np.random.uniform(low=amplitude_high, high=amplitude_low)  # amplitude
			# offset
		offset = np.random.uniform(low=-np.pi, high=np.pi)
			# signals.append(A * np.sin(2 * np.pi * f * ix / float(self._seqLength) + offset))
		signals = A * np.sin(2 * np.pi * f * ix / float(self._seqLength) + offset)
		signals = np.array(signals).T
		tmp_df = pd.DataFrame(signals)
		tmp_df.insert(self._numFeatures, 'LABEL', np.ones(len(tmp_df)))
		# the shape of the samples is seq_length x num_signals
		return tmp_df
