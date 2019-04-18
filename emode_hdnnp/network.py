"""
This version of the Behler-Parinello is aperiodic,non-sparse.
It's being developed to explore alternatives to symmetry functions. It's not for production use.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import random
import sys
import os
import pickle
import numpy as np
import tensorflow as tf

from .element_data import ELEMENT_MODES_NEW, ele_U_wb97xd
import NL
from tensorflow.python.client import timeline

class EMHDNNP(object):
	"""
	The 0.2 model chemistry.
	"""
	def __init__(self, mol_set_name=None, max_num_atoms=None, name=None):
		"""
		Args:
			mol_set (tensormol.MSet object): a class which holds the training data
			name (str): a name used to recall this network

		Notes:
			if name != None, attempts to load a previously saved network, otherwise assumes a new network
		"""
		self.tf_precision = eval("tf.float32")
		self.hidden_layers = [512, 512, 512]
		self.learning_rate = 0.0001
		self.weight_decay = None
		self.max_steps = 1000
		self.batch_size = 64
		self.max_checkpoints = 3
		self.path = "./"
		self.train_gradients = True
		self.activation_function = self.shifted_softplus
		self.test_ratio = 0.1
		self.validation_ratio = 0.1
		self.element_codes = ELEMENT_MODES_NEW
		self.codes_shape = self.element_codes.shape[1]
		self.profiling = False

		#Reloads a previous network if name variable is not None
		if name != None:
			self.name = name
			self.load_network()
			self.network_directory = self.path+self.name
			return

		#Data parameters
		self.mol_set_name = mol_set_name
		self.step = 0
		self.test_freq = 5
		self.network_type = "EModeHDNNP"
		self.name = self.network_type+"_"+time.strftime("%a_%b_%d_%H.%M.%S_%Y")
		self.network_directory = self.path+self.name
		return

	def __getstate__(self):
		state = self.__dict__.copy()
		remove_vars = ["mol_set", "activation_function", "xyz_data", "Z_data", "energy_data", "charges_data",
						"num_atoms_data", "gradient_data"]
		for var in remove_vars:
			try:
				del state[var]
			except:
				pass
		return state

	def set_symmetry_function_params(self):
		self.element_pairs = np.array([
			[self.elements[i], self.elements[j]] for i in range(len(self.elements)) for j in range(i, len(self.elements))])
		self.zeta = 8.0
		self.eta = 4.0

		#Define radial grid parameters
		num_radial_rs = 32
		self.radial_cutoff = 4.6
		self.radial_rs = self.radial_cutoff * np.linspace(0, (num_radial_rs - 1.0) / num_radial_rs, num_radial_rs)

		#Define angular grid parameters
		num_angular_rs = 8
		num_angular_theta_s = 8
		self.angular_cutoff = 3.1
		self.theta_s = np.pi * np.linspace(0, (num_angular_theta_s - 1.0) / num_angular_theta_s, num_angular_theta_s)
		self.angular_rs = self.angular_cutoff * np.linspace(0, (num_angular_rs - 1.0) / num_angular_rs, num_angular_rs)
		return

	def shifted_softplus(self, x):
		return tf.nn.softplus(x) - tf.cast(tf.log(2.0), self.tf_precision)

	def start_training(self):
		self.load_data_to_scratch()
		self.set_symmetry_function_params()
		self.compute_normalization()
		self.save_network()
		self.train_prepare()
		self.train()

	def restart_training(self):
		self.load_data()
		self.train_prepare(restart=True)
		self.train()

	# def train(self):
	# 	for i in range(self.max_steps):
	# 		self.step += 1
	# 		self.train_step(self.step)
	# 		if self.step%self.test_freq==0:
	# 			validation_loss = self.validation_step(self.step)
	# 			if self.step == self.test_freq:
	# 				self.best_loss = validation_loss
	# 				self.save_checkpoint(self.step)
	# 			elif validation_loss < self.best_loss:
	# 				self.best_loss = validation_loss
	# 				self.save_checkpoint(self.step)
	# 	print("Performing final test on independent test set")
	# 	self.test_step(self.step)
	# 	self.sess.close()
	# 	return

	def train(self):
		for i in range(self.max_steps):
			self.step += 1
			train_loss = self.run_step("train")
			if self.step%self.test_freq==0:
				validation_loss = self.run_step("validation")
				if self.step == self.test_freq:
					self.best_loss = validation_loss
					self.save_checkpoint(self.step)
				elif validation_loss < self.best_loss:
					self.best_loss = validation_loss
					self.save_checkpoint(self.step)
		print("Performing final test on independent test set")
		self.run_step("test")
		self.sess.close()
		return

	def save_checkpoint(self, step):
		checkpoint_file = os.path.join(self.network_directory,self.name+'-checkpoint')
		print("----- Saving checkpoint file at step {} -----".format(step))
		self.saver.save(self.sess, checkpoint_file, global_step=step)
		return

	def save_network(self):
		print("Saving TFInstance")
		f = open(self.network_directory+".tfn","wb")
		pickle.dump(self, f, protocol=2)
		f.close()
		return

	def load_network(self):
		print("Loading TFInstance")
		network = pickle.load(open(self.path+"/"+self.name+".tfn","rb"))
		self.__dict__.update(network.__dict__)
		return

	def load_data(self):
		print("----- Opening {:s} to load training data -----".format(self.mol_set_name))
		with open(self.mol_set_name, "rb") as f:
			data = pickle.load(f)
			self.xyz_data = data['xyz']
			self.Z_data = data['Zs']
			self.energy_data = data['energy']
			self.gradient_data = data['gradients']
			self.charges_data = data['charges']
			self.num_atoms_data = data['natoms']
		return

	def load_data_to_scratch(self):
		"""
		Reads built training data off disk into scratch space.
		Divides training and test data.
		Normalizes inputs and outputs.
		note that modifies my MolDigester to incorporate the normalization
		Initializes pointers used to provide training batches.

		Args:
			random: Not yet implemented randomization of the read data.

		Note:
			Also determines mean stoichiometry
		"""
		self.load_data()
		elements, counts = np.unique(self.Z_data, return_counts=True)
		self.elements, counts = elements[1:], counts[1:]
		(self.num_molecules, self.max_num_atoms) = np.shape(self.Z_data)

		self.num_validation_cases = int(self.validation_ratio * self.num_molecules)
		self.num_test_cases = int(self.test_ratio * self.num_molecules)
		num_validation_test = self.num_validation_cases + self.num_test_cases
		self.num_train_cases = int(self.num_molecules - num_validation_test)
		case_idxs = np.arange(int(self.num_molecules))
		np.random.shuffle(case_idxs)
		self.validation_idxs = case_idxs[int(self.num_molecules - self.num_validation_cases):]
		self.test_idxs = case_idxs[int(self.num_molecules - num_validation_test):int(self.num_molecules - self.num_validation_cases)]
		self.train_idxs = case_idxs[:int(self.num_molecules - num_validation_test)]
		self.train_pointer, self.test_pointer, self.validation_pointer = 0, 0, 0

		print("----- Data set cases -----")
		print("Total number of cases:              {:11,d}".format(self.num_molecules))
		print("Number of training cases:           {:11,d}".format(self.num_train_cases))
		print("Number of validation cases:         {:11,d}".format(self.num_validation_cases))
		print("Number of test cases:               {:11,d}".format(self.num_test_cases))
		print("Total number of atoms in data set:  {:11,d}".format(np.sum(self.num_atoms_data)))
		print("Average number of atoms per case:   {:11.2f}".format(np.mean(self.num_atoms_data)))
		print("Max number of atoms per case:       {:11,d}".format(self.max_num_atoms))
		print("----- Elements with counts in data set -----")
		num_elements = len(self.elements)
		n_cols = 2
		n_rows = int(np.ceil(num_elements / n_cols))
		for i in range(n_rows):
			strng = ""
			for j in range(n_cols):
				if (i * n_cols + j) >= num_elements:
					break
				strng += "AN:  {:2d}  Count: {:11,d}      ".format(
					self.elements[i*n_cols+j], counts[i*n_cols+j])
			print(strng.strip())
		if self.batch_size > self.num_train_cases:
			raise Exception("Insufficent data to fill a training batch.\n"\
					+str(self.num_train_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		if self.batch_size > self.num_validation_cases:
			raise Exception("Insufficent data to fill a validation batch.\n"\
					+str(self.num_validation_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		if self.batch_size > self.num_test_cases:
			raise Exception("Insufficent data to fill a test batch.\n"\
					+str(self.num_test_cases)+" cases in dataset with a batch size of "+str(self.batch_size))
		return

	def compute_normalization(self):
		idxs = self.train_idxs
		elements = self.elements.tolist()
		max_atomic_num = elements[-1]
		self.energy_fit = np.zeros((max_atomic_num+1))
		for element in elements:
			self.energy_fit[element] = ele_U_wb97xd[element]
		atomization_energies = self.energy_data[idxs] - np.sum(self.energy_fit[self.Z_data[idxs]], axis=1)
		gradients = self.gradient_data[idxs]
		charges = self.charges_data[idxs]
		self.energy_mean = np.mean(atomization_energies)
		self.energy_std = np.std(atomization_energies)
		grads_mean = np.mean(gradients)
		grads_std = np.std(gradients)
		self.charge_mean = np.zeros((max_atomic_num+1))
		self.charge_std = np.zeros((max_atomic_num+1))
		for element in self.elements:
			element_idxs = np.where(np.equal(self.Z_data, element))
			element_charges = self.charges_data[element_idxs]
			self.charge_mean[element] = np.mean(element_charges)
			self.charge_std[element] = np.std(element_charges)

		print("----- Statistics of label data -----")
		print("Mean       Atomization energy:  {:>9.6f}  Forces:  {:>9.6f}".format(self.energy_mean, grads_mean))
		print("Std. Dev.  Atomization energy:  {:>9.6f}  Forces:  {:>9.6f}".format(self.energy_std, grads_std))
		print("Max        Atomization energy:  {:>9.6f}  Forces:  {:>9.6f}".format(
			np.amax(atomization_energies), np.amax(gradients)))
		print("Min        Atomization energy:  {:>9.6f}  Forces:  {:>9.6f}".format(
			np.amin(atomization_energies), np.amin(gradients)))
		for element in elements:
			print("AN:  {:2d}    Charge mean:  {:>9.6f}    Charge Std. Dev.:  {:>9.6f}".format(
				element, self.charge_mean[element], self.charge_std[element]))
		self.embed_shape = self.codes_shape * (self.radial_rs.shape[0] + self.angular_rs.shape[0] * self.theta_s.shape[0])
		self.label_shape = 1
		return

	def get_batch_data(self, mode):
		if mode == "train":
			if self.train_pointer + self.batch_size >= self.num_train_cases:
				np.random.shuffle(self.train_idxs)
				self.train_pointer = 0
			self.train_pointer += self.batch_size
			idxs = self.train_idxs
			pointer = self.train_pointer
		elif mode == "validation":
			if self.validation_pointer + self.batch_size >= self.num_validation_cases:
				self.validation_pointer = 0
			self.validation_pointer += self.batch_size
			idxs = self.validation_idxs
			pointer = self.validation_pointer
		elif mode == "test":
			if self.test_pointer + self.batch_size >= self.num_test_cases:
				self.test_pointer = 0
			self.test_pointer += self.batch_size
			idxs = self.test_idxs
			pointer = self.test_pointer
		batch_xyzs = self.xyz_data[idxs[pointer-self.batch_size:pointer]]
		batch_Zs = self.Z_data[idxs[pointer-self.batch_size:pointer]]
		batch_energies = self.energy_data[idxs[pointer-self.batch_size:pointer]]
		batch_gradients = self.gradient_data[idxs[pointer-self.batch_size:pointer]]
		batch_charges = self.charges_data[idxs[pointer-self.batch_size:pointer]]
		batch_num_atoms = self.num_atoms_data[idxs[pointer-self.batch_size:pointer]]
		nn_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = NL.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
		batch_data = [batch_xyzs, batch_Zs, batch_energies, batch_gradients,
			batch_charges, batch_num_atoms, nn_pairs, nn_triples, coulomb_pairs]
		return batch_data

	def fill_feed_dict(self, batch_data):
		"""
		Fill the tensorflow feed dictionary.

		Args:
			batch_data: a list of numpy arrays containing inputs, bounds, matrices and desired energies in that order.
			and placeholders to be assigned. (it can be longer than that c.f. TensorMolData_BP)

		Returns:
			Filled feed dictionary.
		"""
		pl_list = [self.xyzs_pl, self.Zs_pl, self.energy_pl, self.gradients_pl,
			self.charges_pl, self.num_atoms_pl, self.nn_pairs_pl, self.nn_triples_pl,
			self.coulomb_pairs_pl]
		feed_dict={i: d for i, d in zip(pl_list, batch_data)}
		return feed_dict

	def variable_with_weight_decay(self, shape, stddev, weight_decay, name = None):
		"""
		Creates random tensorflow variable from a truncated normal distribution with weight decay

		Args:
			name: name of the variable
			shape: list of ints
			stddev: standard deviation of a truncated Gaussian
			wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

		Returns:
			Variable Tensor

		Notes:
			Note that the Variable is initialized with a truncated normal distribution.
			A weight decay is added only if one is specified.
		"""
		variable = tf.Variable(tf.truncated_normal(shape, stddev = stddev, dtype = self.tf_precision), name = name)
		if weight_decay is not None:
			weightdecay = tf.multiply(tf.nn.l2_loss(variable), weight_decay, name='weight_loss')
			tf.add_to_collection('energy_losses', weightdecay)
		return variable

	def train_prepare(self, restart=False):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
		        continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.energy_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size])
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.charges_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision)
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
			charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = self.tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.mol_nn_energy = tf.reduce_sum(self.atom_nn_energy, axis=1)
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
			with tf.name_scope('charge_inference'):
				atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
				atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
				self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
				self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
				self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
				self.charge_loss = 0.1 * self.loss_op(self.charges - self.charge_labels) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
				tf.add_to_collection('total_loss', self.charge_loss)
			self.total_energy = self.mol_nn_energy + self.mol_coulomb_energy
			self.energy_loss = 100.0 * self.loss_op(self.total_energy - self.energy_pl) / tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision)
			tf.add_to_collection('total_loss', self.energy_loss)
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)
				self.gradient_loss = self.loss_op(self.gradients - self.gradient_labels) / (3 * tf.cast(tf.reduce_sum(self.num_atoms_pl), self.tf_precision))
				if self.train_gradients:
					tf.add_to_collection('total_loss', self.gradient_loss)
			self.total_loss = tf.add_n(tf.get_collection('total_loss'))
			self.train_op = self.optimizer(self.total_loss, self.learning_rate)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			if restart:
				self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
			else:
				init = tf.global_variables_initializer()
				self.sess.run(init)
			if self.profiling:
				self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				self.run_metadata = tf.RunMetadata()
		return

	def eval_prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[self.batch_size, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[self.batch_size])
			self.energy_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size])
			self.gradients_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms, 3])
			self.charges_pl = tf.placeholder(self.tf_precision, shape=[self.batch_size, self.max_num_atoms])
			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
			charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			embed = self.tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(embed, atom_codes, padding_mask)
				self.atom_nn_energy_tmp = self.atom_nn_energy * energy_std
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.gather(energy_fit, self.Zs_pl), axis=1)
				self.mol_nn_energy += mol_energy_fit
			with tf.name_scope('charge_inference'):
				atom_nn_charges, charge_variables = self.charge_inference(embed, atom_codes, padding_mask)
				atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
				self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
				self.atom_nn_charges = self.charge_equalization(self.atom_nn_charges, self.num_atoms_pl, self.Zs_pl)
				dxyzs, q1q2, scatter_coulomb = self.gather_coulomb(self.xyzs_pl, self.Zs_pl, self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
				self.charges = tf.gather_nd(self.atom_nn_charges, padding_mask)
				self.charge_labels = tf.gather_nd(self.charges_pl, padding_mask)
			self.total_energy = self.mol_nn_energy + self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.xyz_grad = tf.gradients(self.total_energy, self.xyzs_pl)[0]
				self.gradients = tf.gather_nd(self.xyz_grad, padding_mask)
				self.gradient_labels = tf.gather_nd(self.gradients_pl, padding_mask)

			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return

	def tf_sym_func_element_codes(self, xyzs, Zs, pairs, triples, element_codes, radial_gauss,
			radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta):
		"""
		A tensorflow implementation of the AN1 symmetry function for a set of molecule.
		Args:
			xyzs: nmol X maxnatom X 3 tensor of coordinates.
			Zs: nmol X max_n_atom tensor of atomic numbers.
			elements: a neles X 1 tensor of elements present in the data.
			element_pairs: a nelepairs X 2 X 12tensor of elements pairs present in the data.
			element_codes: n_elements x 4 tensor of codes for embedding elements
			radial_gauss: A symmetry function parameter of radius part
			radial_cutoff: Radial Cutoff of radius part
			angular_gauss: A symmetry function parameter of angular part
			angular_cutoff: Radial Cutoff of angular part
		Returns:
			Digested Mol. In the shape nmol X maxnatom X (Dimension of radius part + Dimension of angular part)
		"""
		dxyzs, pair_Zs = self.sparse_pairs(xyzs, Zs, pairs)
		radial_embed = self.tf_radial_sym_func(dxyzs, radial_gauss, radial_cutoff, eta)
		coded_radial_embed = self.tf_radial_code_channel_sym_func(radial_embed, pair_Zs, element_codes)
		dtxyzs, triples_Zs, scatter_idx = self.sparse_triples(xyzs, Zs, triples)
		angular_embed = self.tf_angular_sym_func(dtxyzs, angular_gauss, thetas, angular_cutoff, zeta, eta)
		padding_mask = tf.where(tf.not_equal(Zs, 0))
		coded_angular_embed = self.tf_angular_code_channel_sym_func(angular_embed, triples_Zs, element_codes, padding_mask, scatter_idx)
		embed = tf.concat([coded_radial_embed, coded_angular_embed], axis=-1)
		return embed

	def sparse_pairs(self, xyzs, Zs, pairs):
		padding_mask = tf.where(tf.not_equal(Zs, 0))
		central_atom_coords = tf.gather_nd(xyzs, padding_mask)
		pairs = tf.gather_nd(pairs, padding_mask)
		padded_pairs = tf.equal(pairs, -1)
		tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
		gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
		pair_coords = tf.gather_nd(xyzs, gather_pairs)
		dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
		pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
		dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), self.tf_precision)
		pair_Zs = tf.gather_nd(Zs, gather_pairs)
		pair_Zs *= pair_mask
		return dxyzs, pair_Zs

	def sparse_triples(self, xyzs, Zs, triples):
		padding_mask = tf.where(tf.not_equal(Zs, 0))
		central_atom_coords = tf.gather_nd(xyzs, padding_mask)
		xyzs = tf.gather(xyzs, padding_mask[:,0])
		Zs = tf.gather(Zs, padding_mask[:,0])
		triples = tf.gather_nd(triples, padding_mask)
		valid_triples = tf.cast(tf.where(tf.not_equal(triples[...,0], -1)), tf.int32)
		triples_idx = tf.gather_nd(triples, valid_triples)
		gather_inds = tf.stack([tf.stack([valid_triples[...,0], triples_idx[:,0]], axis=-1), tf.stack([valid_triples[...,0], triples_idx[:,1]], axis=-1)], axis=-2)
		triples_coords = tf.gather_nd(xyzs, gather_inds)
		central_atom_coords = tf.gather(central_atom_coords, valid_triples[...,0])
		triples_Zs = tf.gather_nd(Zs, gather_inds)
		dtxyzs = tf.expand_dims(central_atom_coords, axis=-2) - triples_coords
		return dtxyzs, triples_Zs, valid_triples

	def tf_radial_sym_func(self, dxyzs, radial_gauss, radial_cutoff, eta):
		"""
		Tensorflow implementation of the ANI-1 radial symmetry functions.

		Args:
			dxyzs: n_case X max_neighbors X 3 tensor of coordinates.
			Zs: n_case X max_neighbors tensor of atomic numbers.
			element_codes: n_elements x 4 tensor of codes for embedding element type
			radial_gauss: n_gauss tensor of radial gaussian centers
			radial_cutoff: radial cutoff distance
			eta: radial gaussian width parameter
		Returns:
			radial_embed: n_case X 4 x n_gauss tensor of atoms embeded into central atoms environment
		"""
		dist_tensor = tf.norm(dxyzs+1.e-16, axis=-1)
		exponent = tf.square(tf.expand_dims(dist_tensor, axis=-1) - radial_gauss)
		exponent *= -1.0 * eta
		gauss = tf.exp(exponent)
		cutoff = 0.5 * (tf.cos(np.pi * dist_tensor / radial_cutoff) + 1.0)
		radial_embed = gauss * tf.expand_dims(cutoff, axis=-1)
		return radial_embed

	def tf_radial_code_channel_sym_func(self, radial_embed, pair_Zs, element_codes):
		pair_codes = tf.gather(element_codes, pair_Zs)
		pad_mask = tf.where(tf.equal(pair_Zs, 0), tf.zeros_like(pair_Zs, dtype=self.tf_precision),
					tf.ones_like(pair_Zs, dtype=self.tf_precision))
		pair_codes *= tf.expand_dims(pad_mask, axis=-1)
		radial_embed = tf.expand_dims(radial_embed, axis=-2) * tf.expand_dims(pair_codes, axis=-1)
		radial_embed = tf.reduce_sum(radial_embed, axis=1)
		return radial_embed

	def tf_angular_sym_func(self, dtxyzs, angular_gauss, thetas, angular_cutoff, zeta, eta):
		"""
		Tensorflow implementation of the ANI-1 angular symmetry functions.

		Args:
			R: a nmol X maxnatom X 3 tensor of coordinates.
			Zs : nmol X maxnatom X 1 tensor of atomic numbers.
			eleps_: a nelepairs X 2 tensor of element pairs present in the data.
			SFP: A symmetry function parameter tensor having the number of elements
			as the SF output. 4 X nzeta X neta X thetas X nRs. For example, SFPs_[0,0,0,0,0]
			is the first zeta parameter. SFPs_[3,0,0,0,1] is the second R parameter.
			R_cut: Radial Cutoff
			AngtriEle: angular triples within the cutoff. m, i, j, k, l
			prec: a precision.
		Returns:
			Digested Mol. In the shape nmol X maxnatom X nelepairs X nZeta X nEta X nThetas X nRs
		"""
		dist_jk_tensor = tf.norm(dtxyzs+1.e-16, axis=-1)
		dij_dik = dist_jk_tensor[...,0] * dist_jk_tensor[...,1]
		ij_dot_ik = tf.reduce_sum(dtxyzs[...,0,:] * dtxyzs[...,1,:], axis=-1)
		cos_angle = ij_dot_ik / dij_dik
		cos_angle = tf.where(tf.greater(cos_angle, 1.0 - 1.e-6), tf.ones_like(cos_angle) - 1.e-6, cos_angle)
		cos_angle = tf.where(tf.less(cos_angle, -1.0 + 1.e-6), -1.0 * tf.ones_like(cos_angle) + 1.e-6, cos_angle)
		theta_ijk = tf.acos(cos_angle)
		dtheta = tf.expand_dims(theta_ijk, axis=-1) - thetas
		cos_factor = tf.cos(2 * dtheta)
		exponent = tf.expand_dims(tf.reduce_sum(dist_jk_tensor, axis=-1) / 2.0, axis=-1) - angular_gauss
		dist_factor = tf.exp(-eta * tf.square(exponent))
		cutoffj = 0.5 * (tf.cos(np.pi * dist_jk_tensor[...,0] / angular_cutoff) + 1.0)
		cutoffk = 0.5 * (tf.cos(np.pi * dist_jk_tensor[...,1] / angular_cutoff) + 1.0)
		cutoff = cutoffj * cutoffk
		angular_embed = tf.expand_dims(tf.pow(1.0 + cos_factor, zeta), axis=-1) * tf.expand_dims(dist_factor, axis=-2)
		angular_embed *= tf.pow(tf.cast(2.0, self.tf_precision), 1.0 - zeta)
		angular_embed *= tf.expand_dims(tf.expand_dims(cutoff, axis=-1), axis=-1)
		return angular_embed

	def tf_angular_code_channel_sym_func(self, angular_embed, triples_Zs, element_codes, padding_mask, triples_scatter):
		angular_embed = (tf.expand_dims(angular_embed, axis=-3)
							* tf.expand_dims(tf.expand_dims(tf.reduce_prod(tf.gather(element_codes,
							triples_Zs), axis=-2), axis=-1), axis=-1))
		scatter_shape = [tf.shape(padding_mask)[0], tf.reduce_max(triples_scatter[:,1]) + 1, tf.shape(element_codes)[1], 8, 8]
		angular_embed = tf.reduce_sum(tf.scatter_nd(triples_scatter, angular_embed, scatter_shape), axis=1)
		angular_embed = tf.reshape(angular_embed, [tf.shape(angular_embed)[0], tf.shape(element_codes)[1], -1])
		return angular_embed

	def energy_inference(self, embed, atom_codes, indices):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		variables=[]
		with tf.variable_scope("energy_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel1", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape), dtype=self.tf_precision)
			variables.append(code_kernel1)
			variables.append(code_kernel2)
			coded_weights = tf.matmul(atom_codes, code_kernel1)
			coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
			coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
								stddev=np.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(coded_embed, weights) + biases)
						variables.append(weights)
						variables.append(biases)
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=np.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(activations, weights) + biases)
						variables.append(weights)
						variables.append(biases)
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=np.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
				variables.append(weights)
				variables.append(biases)
				atom_nn_energy = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_energy, variables

	def charge_inference(self, embed, atom_codes, indices):
		"""
		Builds a Behler-Parinello graph

		Args:
			inp: a list of (num_of atom type X flattened input shape) matrix of input cases.
			index: a list of (num_of atom type X batchsize) array which linearly combines the elements
		Returns:
			The BP graph output
		"""
		variables=[]
		with tf.variable_scope("charge_network", reuse=tf.AUTO_REUSE):
			code_kernel1 = tf.get_variable(name="CodeKernel", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
			code_kernel2 = tf.get_variable(name="CodeKernel2", shape=(self.codes_shape, self.codes_shape),dtype=self.tf_precision)
			variables.append(code_kernel1)
			variables.append(code_kernel2)
			coded_weights = tf.matmul(atom_codes, code_kernel1)
			coded_embed = tf.einsum('ijk,ij->ijk', embed, coded_weights)
			coded_embed = tf.reshape(tf.einsum('ijk,jl->ilk', coded_embed, code_kernel2), [tf.shape(embed)[0], -1])
			embed = tf.reshape(embed, [tf.shape(embed)[0], -1])
			for i in range(len(self.hidden_layers)):
				if i == 0:
					with tf.name_scope('hidden1'):
						weights = self.variable_with_weight_decay(shape=[self.embed_shape, self.hidden_layers[i]],
								stddev=np.sqrt(2.0 / float(self.embed_shape)), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(embed, weights) + biases)
						variables.append(weights)
						variables.append(biases)
				else:
					with tf.name_scope('hidden'+str(i+1)):
						weights = self.variable_with_weight_decay(shape=[self.hidden_layers[i-1], self.hidden_layers[i]],
								stddev=np.sqrt(2.0 / float(self.hidden_layers[i-1])), weight_decay=self.weight_decay, name="weights")
						biases = tf.Variable(tf.zeros([self.hidden_layers[i]], dtype=self.tf_precision), name='biases')
						activations = self.activation_function(tf.matmul(activations, weights) + biases)
						variables.append(weights)
						variables.append(biases)
			with tf.name_scope('regression_linear'):
				weights = self.variable_with_weight_decay(shape=[self.hidden_layers[-1], 1],
						stddev=np.sqrt(2.0 / float(self.hidden_layers[-1])), weight_decay=self.weight_decay, name="weights")
				biases = tf.Variable(tf.zeros([1], dtype=self.tf_precision), name='biases')
				outputs = tf.squeeze(tf.matmul(activations, weights) + biases, axis=1)
				variables.append(weights)
				variables.append(biases)
				atom_nn_charges = tf.scatter_nd(indices, outputs, [self.batch_size, self.max_num_atoms])
		return atom_nn_charges, variables

	def charge_equalization(self, atom_nn_charges, num_atoms, Zs):
		excess_charge = tf.reduce_sum(atom_nn_charges, axis=1)
		atom_nn_charges -= tf.expand_dims(excess_charge / tf.cast(num_atoms, self.tf_precision), axis=-1)
		mask = tf.where(tf.equal(Zs, 0), tf.zeros_like(Zs, dtype=self.tf_precision),
				tf.ones_like(Zs, dtype=self.tf_precision))
		atom_nn_charges = atom_nn_charges * mask
		return atom_nn_charges

	def alchem_charge_equalization(self, atom_nn_charges, num_alchem_atoms, alchem_switch):
		excess_charge = tf.reduce_sum(atom_nn_charges, axis=1)
		atom_nn_charges -= (excess_charge / num_alchem_atoms) * alchem_switch
		return atom_nn_charges

	def gather_coulomb(self, xyzs, Zs, atom_charges, pairs):
		padding_mask = tf.where(tf.logical_and(tf.not_equal(Zs, 0), tf.reduce_any(tf.not_equal(pairs, -1), axis=-1)))
		central_atom_coords = tf.gather_nd(xyzs, padding_mask)
		central_atom_charge = tf.gather_nd(atom_charges, padding_mask)
		pairs = tf.gather_nd(pairs, padding_mask)
		padded_pairs = tf.equal(pairs, -1)
		tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
		gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
		pair_coords = tf.gather_nd(xyzs, gather_pairs)
		dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
		pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
		dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), self.tf_precision)
		pair_charges = tf.gather_nd(atom_charges, gather_pairs)
		pair_charges *= tf.cast(pair_mask, self.tf_precision)
		q1q2 = tf.expand_dims(central_atom_charge, axis=-1) * pair_charges
		return dxyzs, q1q2, padding_mask

	def alchem_gather_coulomb(self, xyzs, atom_charges, pairs):
		padding_mask = tf.where(tf.reduce_any(tf.not_equal(pairs, -1), axis=-1))
		central_atom_coords = tf.gather_nd(xyzs, padding_mask)
		central_atom_charge = tf.gather_nd(atom_charges, padding_mask)
		pairs = tf.gather_nd(pairs, padding_mask)
		padded_pairs = tf.equal(pairs, -1)
		tmp_pairs = tf.where(padded_pairs, tf.zeros_like(pairs), pairs)
		gather_pairs = tf.stack([tf.cast(tf.tile(padding_mask[:,:1], [1, tf.shape(pairs)[1]]), tf.int32), tmp_pairs], axis=-1)
		pair_coords = tf.gather_nd(xyzs, gather_pairs)
		dxyzs = tf.expand_dims(central_atom_coords, axis=1) - pair_coords
		pair_mask = tf.where(padded_pairs, tf.zeros_like(pairs), tf.ones_like(pairs))
		dxyzs *= tf.cast(tf.expand_dims(pair_mask, axis=-1), self.tf_precision)
		pair_charges = tf.gather_nd(atom_charges, gather_pairs)
		pair_charges *= tf.cast(pair_mask, self.tf_precision)
		q1q2 = tf.expand_dims(central_atom_charge, axis=-1) * pair_charges
		return dxyzs, q1q2, padding_mask

	def calculate_coulomb_energy(self, dxyzs, q1q2, scatter_idx):
		"""
		Polynomial cutoff 1/r (in BOHR) obeying:
		kern = 1/r at SROuter and LRInner
		d(kern) = d(1/r) (true force) at SROuter,LRInner
		d**2(kern) = d**2(1/r) at SROuter and LRInner.
		d(kern) = 0 (no force) at/beyond SRInner and LROuter

		The hard cutoff is LROuter
		"""
		srange_inner = tf.constant(4.0*1.889725989, dtype=self.tf_precision)
		srange_outer = tf.constant(6.5*1.889725989, dtype=self.tf_precision)
		lrange_inner = tf.constant(13.0*1.889725989, dtype=self.tf_precision)
		lrange_outer = tf.constant(15.0*1.889725989, dtype=self.tf_precision)
		a, b, c, d, e, f, g, h = -9.83315, 4.49307, -0.784438, 0.0747019, -0.00419095, 0.000138593, -2.50374e-6, 1.90818e-8
		dist = tf.norm(dxyzs+1.e-16, axis=-1)
		dist *= 1.889725989
		dist = tf.where(tf.less(dist, srange_inner), tf.ones_like(dist) * srange_inner, dist)
		dist = tf.where(tf.greater(dist, lrange_outer), tf.ones_like(dist) * lrange_outer, dist)
		dist2 = dist * dist
		dist3 = dist2 * dist
		dist4 = dist3 * dist
		dist5 = dist4 * dist
		dist6 = dist5 * dist
		dist7 = dist6 * dist
		kern = (a + b*dist + c*dist2 + d*dist3 + e*dist4 + f*dist5 + g*dist6 + h*dist7) / dist
		mrange_energy = tf.reduce_sum(kern * q1q2, axis=1)
		lrange_energy = tf.reduce_sum(q1q2, axis=1) / lrange_outer
		coulomb_energy = (mrange_energy - lrange_energy) / 2.0
		return tf.reduce_sum(tf.scatter_nd(scatter_idx, coulomb_energy, [self.batch_size, self.max_num_atoms]), axis=-1)

	def optimizer(self, loss, learning_rate, variables=None):
		"""
		Sets up the training Ops.
		Creates a summarizer to track the loss over time in TensorBoard.
		Creates an optimizer and applies the gradients to all trainable variables.
		The Op returned by this function is what must be passed to the
		`sess.run()` call to cause the model to train.

		Args:
			loss: Loss tensor, from loss().
			learning_rate: the learning rate to use for gradient descent.

		Returns:
			train_op: the tensorflow operation to call for training.
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)
		return train_op

	def loss_op(self, error):
		loss = tf.nn.l2_loss(error)
		return loss

	def run_step(self, mode):
		start_time = time.time()
		total_loss, total_energy_loss, total_gradient_loss, total_charge_loss =  0.0, 0.0, 0.0, 0.0
		num_atoms_epoch = 0
		energy_labels, energy_outputs = [], []
		force_labels, force_outputs = [], []
		charge_labels, charge_outputs = [], []
		if mode == "train":
			num_batches = int(0.25 * self.num_train_cases/self.batch_size)
		elif mode == "validation":
			num_batches = int(self.num_validation_cases/self.batch_size)
		elif mode == "test":
			num_batches = int(self.num_test_cases/self.batch_size)
		for ministep in range(num_batches):
			batch_data = self.get_batch_data(mode)
			feed_dict = self.fill_feed_dict(batch_data)
			if mode == "train":
				if self.train_gradients:
					(_, loss, energy_loss, gradient_loss, charge_loss) = self.sess.run([
						self.train_op, self.total_loss, self.energy_loss, self.gradient_loss,
						self.charge_loss], feed_dict=feed_dict)
					total_gradient_loss += gradient_loss
				else:
					(_, loss, energy_loss, charge_loss) = self.sess.run([self.train_op,
						self.total_loss, self.energy_loss, self.charge_loss], feed_dict=feed_dict)
			elif (mode == "validation") or (mode == "test"):
				(total_energies, batch_energy_labels, gradients, batch_gradient_labels, charges,
					batch_charge_labels, loss, energy_loss, gradient_loss, charge_loss,
					num_atoms) = self.sess.run([self.total_energy, self.energy_pl,
					self.gradients, self.gradient_labels, self.charges, self.charge_labels,
					self.total_loss, self.energy_loss, self.gradient_loss, self.charge_loss,
					self.num_atoms_pl],  feed_dict=feed_dict)
				energy_labels.append(batch_energy_labels)
				energy_outputs.append(total_energies)
				force_labels.append(-1.0 * batch_gradient_labels)
				force_outputs.append(-1.0 * gradients)
				charge_labels.append(batch_charge_labels)
				charge_outputs.append(charges)
				total_gradient_loss += gradient_loss
				num_atoms_epoch += np.sum(num_atoms)
			total_loss += loss
			total_energy_loss += energy_loss
			total_charge_loss += charge_loss
		total_loss /= num_batches
		total_energy_loss /= num_batches
		total_gradient_loss /= num_batches
		total_charge_loss /= num_batches
		duration = time.time() - start_time
		self.print_epoch(duration, total_loss, total_energy_loss,
			total_gradient_loss, total_charge_loss, mode=mode)
		if (mode == "validation") or (mode == "test"):
			self.print_error_stats(energy_labels, energy_outputs, force_labels, force_outputs,
				charge_labels, charge_outputs, num_batches, num_atoms_epoch)
		return total_loss

	def print_epoch(self, duration, loss, energy_loss, gradient_loss, charge_loss, mode="train"):
		print("step: {:>4d}  duration: {:>5.1f}  {:>10s} loss: {:>12.8f}  energy loss: {:>12.8f}  gradient loss: {:>12.8f}  charge loss: {:>12.8f}".format(
			self.step, duration, mode, loss, energy_loss, gradient_loss, charge_loss))
		return

	def print_error_stats(self, energy_labels, energy_outputs, force_labels,
			force_outputs, charge_labels, charge_outputs, num_batches, num_atoms_epoch):
		energy_labels = np.concatenate(energy_labels)
		energy_outputs = np.concatenate(energy_outputs)
		energy_errors = energy_labels - energy_outputs
		force_labels = np.concatenate(force_labels)
		force_outputs = np.concatenate(force_outputs)
		force_errors = force_labels - force_outputs
		charge_labels = np.concatenate(charge_labels)
		charge_outputs = np.concatenate(charge_outputs)
		charge_errors = charge_labels - charge_outputs
		print("----- Sample labels and predictions from data subset -----")
		for i in [random.randint(0, num_batches * self.batch_size - 1) for _ in range(10)]:
			print("Energy label: {:>12.6f}  Energy output: {:>12.6f}".format(
				energy_labels[i], energy_outputs[i]))
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
			print("Charge label: {:>12.6f}  Charge output: {:>12.6f}".format(
				charge_labels[i], charge_outputs[i]))
		for i in [random.randint(0, num_atoms_epoch - 1) for _ in range(10)]:
			print("Forces label: {:>9.6f} {:>9.6f} {:>9.6f}  Forces output: {:>9.6f} {:>9.6f} {:>9.6f}".format(
				force_labels[i,0], force_labels[i,1], force_labels[i,2],
				force_outputs[i,0], force_outputs[i,1], force_outputs[i,2]))
		print("----- Error statistics from data subset -----")
		print("MAE    Energy: {:>9.6f}    Forces: {:>9.6f}    Charges: {:>9.6f}".format(
			np.mean(np.abs(energy_errors)), np.mean(np.abs(force_errors)),
			np.mean(np.abs(charge_errors))))
		print("MSE    Energy: {:>9.6f}    Forces: {:>9.6f}    Charges: {:>9.6f}".format(
			np.mean(energy_errors), np.mean(force_errors),
			np.mean(charge_errors)))
		print("RMSE   Energy: {:>9.6f}    Forces: {:>9.6f}    Charges: {:>9.6f}".format(
			np.sqrt(np.mean(np.square(energy_errors))), np.sqrt(np.mean(np.square(force_errors))),
			np.sqrt(np.mean(np.square(charge_errors)))))
		print("MaxE   Energy: {:>9.6f}    Forces: {:>9.6f}    Charges: {:>9.6f}".format(
			np.amax(np.abs(energy_errors)), np.amax(np.abs(force_errors)),
			np.amax(np.abs(charge_errors))))
		return

	def alchem_prepare(self):
		"""
		Get placeholders, graph and losses in order to begin training.
		Also assigns the desired padding.

		Args:
			continue_training: should read the graph variables from a saved checkpoint.
		"""
		with tf.Graph().as_default():
			self.xyzs_pl = tf.placeholder(self.tf_precision, shape=[None, self.max_num_atoms, 3])
			self.Zs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms])
			self.nn_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.nn_triples_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None, 2])
			self.coulomb_pairs_pl = tf.placeholder(tf.int32, shape=[None, self.max_num_atoms, None])
			self.num_atoms_pl = tf.placeholder(tf.int32, shape=[None])
			self.delta_pl = tf.placeholder(self.tf_precision, shape=[1])

			radial_gauss = tf.Variable(self.radial_rs, trainable=False, dtype = self.tf_precision)
			angular_gauss = tf.Variable(self.angular_rs, trainable=False, dtype = self.tf_precision)
			thetas = tf.Variable(self.theta_s, trainable=False, dtype = self.tf_precision)
			radial_cutoff = tf.Variable(self.radial_cutoff, trainable=False, dtype = self.tf_precision)
			angular_cutoff = tf.Variable(self.angular_cutoff, trainable=False, dtype = self.tf_precision)
			zeta = tf.Variable(self.zeta, trainable=False, dtype = self.tf_precision)
			eta = tf.Variable(self.eta, trainable=False, dtype = self.tf_precision)
			self.element_codes = tf.Variable(self.element_codes, trainable=True, dtype=self.tf_precision, name="element_codes")
			energy_fit = tf.Variable(self.energy_fit, trainable=False, dtype=self.tf_precision)
			charge_mean = tf.Variable(self.charge_mean, trainable=False, dtype=self.tf_precision)
			charge_std = tf.Variable(self.charge_std, trainable=False, dtype=self.tf_precision)

			padding_mask = tf.where(tf.not_equal(self.Zs_pl, 0))
			alchem_padding_mask = tf.where(tf.reduce_any(tf.not_equal(self.Zs_pl, 0), axis=0, keepdims=True))
			_, max_atom_idx = tf.nn.top_k(self.num_atoms_pl)
			self.alchem_xyzs = tf.gather(self.xyzs_pl, max_atom_idx)
			self.alchem_switch = tf.where(tf.not_equal(self.Zs_pl, 0), tf.stack([tf.tile(1.0 - self.delta_pl,
								[self.max_num_atoms]), tf.tile(self.delta_pl, [self.max_num_atoms])]),
								tf.zeros_like(self.Zs_pl, dtype=self.tf_precision))
			embed = tf_sym_func_element_codes(self.xyzs_pl, self.Zs_pl, self.nn_pairs_pl, self.nn_triples_pl,
				self.element_codes, radial_gauss, radial_cutoff, angular_gauss, thetas, angular_cutoff, zeta, eta)
			reconst_embed = tf.scatter_nd(padding_mask, embed, [tf.cast(tf.shape(self.Zs_pl)[0], tf.int64),
				self.max_num_atoms, self.codes_shape, int(self.embed_shape / self.codes_shape)])
			alchem_embed = tf.reduce_sum(tf.stack([reconst_embed[0] * (1.0 - self.delta_pl),
				reconst_embed[1] * self.delta_pl], axis=0), axis=0)
			atom_codes = tf.gather(self.element_codes, tf.gather_nd(self.Zs_pl, padding_mask))
			reconst_atom_codes = tf.scatter_nd(padding_mask, atom_codes,
				[tf.cast(tf.shape(self.Zs_pl)[0], tf.int64), self.max_num_atoms, self.codes_shape])
			alchem_atom_codes = tf.reduce_sum(tf.stack([reconst_atom_codes[0] * (1.0 - self.delta_pl),
				reconst_atom_codes[1] * self.delta_pl], axis=0), axis=0)
			with tf.name_scope('energy_inference'):
				self.atom_nn_energy, variables = self.energy_inference(alchem_embed,
					alchem_atom_codes, alchem_padding_mask)
				self.atom_nn_energy *= tf.reduce_sum(self.alchem_switch, axis=0)
				self.mol_nn_energy = (tf.reduce_sum(self.atom_nn_energy, axis=1) * energy_std) + energy_mean
				mol_energy_fit = tf.reduce_sum(tf.reduce_sum(tf.gather(energy_fit,
					self.Zs_pl) * self.alchem_switch, axis=0), axis=0)
				# self.mol_nn_energy += mol_energy_fit
			with tf.name_scope('charge_inference'):
				atom_nn_charges, charge_variables = self.charge_inference(alchem_embed,
					alchem_atom_codes, alchem_padding_mask)
				atom_charge_mean, atom_charge_std = tf.gather(charge_mean, self.Zs_pl), tf.gather(charge_std, self.Zs_pl)
				atom_charge_mean *= self.alchem_switch
				atom_charge_std *= self.alchem_switch
				atom_charge_mean = tf.reduce_sum(atom_charge_mean, axis=0)
				atom_charge_std = tf.reduce_sum(atom_charge_std, axis=0)
				self.atom_nn_charges = (atom_nn_charges * atom_charge_std) + atom_charge_mean
				num_alchem_atoms = tf.reduce_sum(self.alchem_switch)
				self.atom_nn_charges = self.alchem_charge_equalization(self.atom_nn_charges,
					num_alchem_atoms, tf.reduce_sum(self.alchem_switch, axis=0))
				dxyzs, q1q2, scatter_coulomb = self.alchem_gather_coulomb(self.alchem_xyzs,
					self.atom_nn_charges, self.coulomb_pairs_pl)
				self.mol_coulomb_energy = self.calculate_coulomb_energy(dxyzs, q1q2, scatter_coulomb)
			self.total_energy = self.mol_nn_energy + self.mol_coulomb_energy
			with tf.name_scope('gradients'):
				self.gradients = tf.reduce_sum(tf.gradients(self.total_energy, self.xyzs_pl)[0], axis=0)
			self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
			self.saver = tf.train.Saver(max_to_keep = self.max_checkpoints)
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.network_directory))
		return


	def get_eval_batch(self):
		batch_xyzs = self.xyz_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]]
		batch_Zs = self.Z_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]]
		nn_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = NL.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, 19.0, self.max_num_atoms, False, False)
		batch_data = []
		batch_data.append(batch_xyzs)
		batch_data.append(batch_Zs)
		batch_data.append(nn_pairs)
		batch_data.append(nn_triples)
		batch_data.append(coulomb_pairs)
		batch_data.append(self.num_atoms_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]])
		batch_data.append(self.energy_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]])
		batch_data.append(self.gradient_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]])
		batch_data.append(self.charges_data[self.train_idxs[self.train_pointer -self.batch_size:self.train_pointer]])
		return batch_data

	def evaluate_set(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		energy_true, energy_pred = [], []
		gradients_true, gradient_preds = [], []
		charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = NL.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			total_energy, energy_label, gradients, gradient_labels, charges, charge_labels = self.sess.run([self.total_energy,
				self.energy_pl, self.gradients, self.gradient_labels, self.charges, self.charge_labels], feed_dict=feed_dict)
			energy_true.append(energy_label)
			energy_pred.append(total_energy)
			gradients_true.append(gradient_labels)
			gradient_preds.append(gradients)
			charges_true.append(charge_labels)
			charge_preds.append(charges)
		energy_true = np.concatenate(energy_true)
		energy_pred = np.concatenate(energy_pred)
		gradients_true = np.concatenate(gradients_true)
		gradient_preds = np.concatenate(gradient_preds)
		energy_errors = energy_true - energy_pred
		gradient_errors = gradients_true - gradient_preds
		charges_true = np.concatenate(charges_true)
		charge_preds = np.concatenate(charge_preds)
		charge_errors = charges_true - charge_preds
		return energy_errors, gradient_errors, charge_errors

	def evaluate_mol(self, mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.train_prepare(restart=True)
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		xyz_data[0][:mol.NAtoms()] = mol.coords
		Z_data[0][:mol.NAtoms()] = mol.atoms
		num_atoms_data[0] = mol.NAtoms()
		nn_pairs = NL.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
		nn_triples = NL.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
		coulomb_pairs = NL.Make_NLTensor(xyz_data, Z_data, 19.0, self.max_num_atoms, False, False)
		feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
					self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
		energy, atme, nne, ce, gradients, charges = self.sess.run([self.total_energy, self.atom_nn_energy, self.mol_nn_energy, self.mol_coulomb_energy, self.gradients, self.atom_nn_charges], feed_dict=feed_dict)
		return energy, atme, nne, ce, -gradients, charges

	def evaluate_alchem_mol(self, mols):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = max([mol.NAtoms() for mol in mols])
			self.batch_size = 1
			self.alchem_prepare()
		xyz_data = np.zeros((len(mols), self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((len(mols), self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((len(mols)), dtype = np.int32)
		max_alchem_atoms = np.argmax(num_atoms_data)
		def alchem_energy_force(mols, delta, return_forces=True):
			for i, mol in enumerate(mols):
				xyz_data[i][:mol.NAtoms()] = mols[i].coords
				Z_data[i][:mol.NAtoms()] = mols[i].atoms
				num_atoms_data[i] = mol.NAtoms()
			nn_pairs = NL.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = NL.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = NL.Make_NLTensor(xyz_data[max_alchem_atoms:max_alchem_atoms+1],
							Z_data[max_alchem_atoms:max_alchem_atoms+1], 15.0, self.max_num_atoms, True, False)
			delta = np.array(delta).reshape(1)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs,
						self.num_atoms_pl:num_atoms_data, self.delta_pl:delta}
			energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
			return energy[0], -gradients
		return alchem_energy_force

	def get_energy_force_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.eval_prepare()
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def EF(xyz_, DoForce=True):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = NL.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = NL.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = NL.Make_NLTensor(xyz_data, Z_data, 15.0, self.max_num_atoms, True, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.coulomb_pairs_pl:coulomb_pairs, self.num_atoms_pl:num_atoms_data}
			if (DoForce):
				energy, gradients = self.sess.run([self.total_energy, self.gradients], feed_dict=feed_dict)
				return energy[0], -JOULEPERHARTREE*gradients
			else:
				energy = self.sess.run(self.total_energy, feed_dict=feed_dict)
				return energy[0]
		return EF

	def get_charge_function(self,mol):
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mol.NAtoms()
			self.batch_size = 1
			self.train_prepare(restart=True)
		num_mols = 1
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		num_atoms_data[0] = mol.NAtoms()
		Z_data[0][:mol.NAtoms()] = mol.atoms
		def QF(xyz_):
			xyz_data[0][:mol.NAtoms()] = xyz_
			nn_pairs = NL.Make_NLTensor(xyz_data, Z_data, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = NL.Make_TLTensor(xyz_data, Z_data, self.angular_cutoff, self.max_num_atoms, False)
			feed_dict = {self.xyzs_pl:xyz_data, self.Zs_pl:Z_data, self.nn_pairs_pl:nn_pairs,
						self.nn_triples_pl:nn_triples, self.num_atoms_pl:num_atoms_data}
			charges = self.sess.run(self.charges, feed_dict=feed_dict)
			return charges
		return QF

	def get_atom_energies(self, mset):
		"""
		Takes coordinates and atomic numbers from a manager and feeds them into the network
		for evaluation of the forces

		Args:
			xyzs (np.float): numpy array of atomic coordinates
			Zs (np.int32): numpy array of atomic numbers
		"""
		try:
			self.sess
		except AttributeError:
			self.sess = None
		if self.sess is None:
			self.assign_activation()
			self.max_num_atoms = mset.MaxNAtom()
			self.batch_size = 100
			self.eval_prepare()
		atom_energy_data = []
		num_mols = len(mset.mols)
		xyz_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype = np.float64)
		Z_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.int32)
		charges_data = np.zeros((num_mols, self.max_num_atoms), dtype = np.float64)
		num_atoms_data = np.zeros((num_mols), dtype = np.int32)
		energy_data = np.zeros((num_mols), dtype = np.float64)
		gradient_data = np.zeros((num_mols, self.max_num_atoms, 3), dtype=np.float64)
		for i, mol in enumerate(mset.mols):
			xyz_data[i][:mol.NAtoms()] = mol.coords
			Z_data[i][:mol.NAtoms()] = mol.atoms
			charges_data[i][:mol.NAtoms()] = mol.properties["charges"]
			energy_data[i] = mol.properties["energy"]
			gradient_data[i][:mol.NAtoms()] = mol.properties["gradients"]
			num_atoms_data[i] = mol.NAtoms()
		eval_pointer = 0
		# energy_true, energy_pred = [], []
		# gradients_true, gradient_preds = [], []
		# charges_true, charge_preds = [], []
		for ministep in range(int(num_mols / self.batch_size)):
			eval_pointer += self.batch_size
			batch_xyzs = xyz_data[eval_pointer - self.batch_size:eval_pointer]
			batch_Zs = Z_data[eval_pointer - self.batch_size:eval_pointer]
			nn_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, self.radial_cutoff, self.max_num_atoms, True, True)
			nn_triples = NL.Make_TLTensor(batch_xyzs, batch_Zs, self.angular_cutoff, self.max_num_atoms, False)
			coulomb_pairs = NL.Make_NLTensor(batch_xyzs, batch_Zs, 15.0, self.max_num_atoms, True, False)
			batch_data = []
			batch_data.append(batch_xyzs)
			batch_data.append(batch_Zs)
			batch_data.append(nn_pairs)
			batch_data.append(nn_triples)
			batch_data.append(coulomb_pairs)
			batch_data.append(num_atoms_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(energy_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(gradient_data[eval_pointer - self.batch_size:eval_pointer])
			batch_data.append(charges_data[eval_pointer - self.batch_size:eval_pointer])
			feed_dict = self.fill_feed_dict(batch_data)
			batch_atom_energies = self.sess.run(self.atom_nn_energy_tmp,
				feed_dict=feed_dict)
			atom_energy_data.append(batch_atom_energies)
		atom_energy_data = np.concatenate(atom_energy_data)
		Z_data_tmp = Z_data[:np.shape(atom_energy_data)[0]]
		elements = mset.AtomTypes().tolist()
		element_energy_data = []
		for element in elements:
			element_idxs = np.where(np.equal(Z_data_tmp, element))
			atom_energies = atom_energy_data[element_idxs]
			element_energy_data.append(atom_energies)
		return element_energy_data
