from torch.utils.data.dataset import Dataset
import torch
import os
import pickle
from lie_algebra import SO3

class HUSKYDataset(Dataset):
	def __init__(self, args):
		self.name = args.dataset_name
		self.path_data_save = args.path_data_save
		self.path_results = args.path_results
		self.path_temp = args.path_temp
		self.train_sequences = args.train_sequences
		self.test_sequences = args.test_sequences
		self.cross_validation_sequences = args.cross_validation_sequences
		self.get_datasets()
		self.set_normalize_factors(args)

	def get_datasets(self):
		self.datasets = []
		for dataset in os.listdir(self.path_data_save):
			self.datasets += [dataset[:-2]] # take just name .p
		self.divide_datasets()

	def divide_datasets(self):
		self.datasets_test = self.test_sequences
		self.datasets_validation = self.cross_validation_sequences
		self.datasets_train = self.train_sequences

	def dataset_name(self, i):
		return self.datasets[i]

	def get_filter_data(self, i):
		if type(i) != int:
			i = self.datasets.index(i)
		pickle_dict =  self[i]
		t = pickle_dict['t']
		chi0 = pickle_dict['chi'][0]
		Rot0 = chi0[:3, :3]

		angles = SO3.to_rpy(Rot0.unsqueeze(dim=0))
		p0 = chi0[:3, 3]
		u_odo_fog = pickle_dict['u_odo_fog']
		y_imu = pickle_dict['u_imu']
		x0 = torch.zeros(9)
		x0[:3] = p0
		x0[3:6] = angles
		return t, x0, u_odo_fog, y_imu

	def get_ground_truth_data(self, i):
		pickle_dict =  self[self.datasets.index(i) if type(i) != int else i]
		return pickle_dict['t'], pickle_dict['chi']

	def get_test_data(self, i, gp_name):
		var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
		dataset = self.datasets_test[i] if type(i) == int else i
		pickle_dict =  self[self.datasets.index(dataset)]
		u = pickle_dict["u_" + var]
		y = pickle_dict["y_" + var]
		u = self.normalize(u, "u_" + var)
		return u, y

	def get_validation_data(self, i, gp_name):
		var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
		dataset = self.datasets_validation[i] if type(i) == int else i
		pickle_dict =  self[self.datasets.index(dataset)]
		u = pickle_dict["u_" + var]
		y = pickle_dict["y_" + var]
		u = self.normalize(u, "u_" + var)
		return u, y

	def get_train_data(self, i, gp_name):
		var = "odo_fog" if gp_name == "GpOdoFog" else "imu"
		dataset = self.datasets_train[i] if type(i) == int else i
		pickle_dict =  self[self.datasets.index(dataset)]
		u = pickle_dict["u_" + var]
		y = pickle_dict["y_" + var]
		u = self.normalize(u, "u_" + var)
		return u, y

	def __getitem__(self, i):
		with open(self.path_data_save + self.datasets[i] + '.p', "rb") as file_pi:
			mondict = pickle.load(file_pi)
		return mondict

	def __len__(self):
		return len(self.datasets)

	def set_normalize_factors(self, args):
		"""
		Compute mean and variance of input data using only training data
		"""
		# first mean
		self.num_data = 0
		for i, dataset in enumerate(self.datasets_train):
			with open(self.path_data_save + dataset + '.p', "rb") as file_pi:
				pickle_dict = pickle.load(file_pi)
			u_odo_fog = pickle_dict['u_odo_fog']
			u_imu = pickle_dict['u_imu']
			if i == 0:
				u_odo_fog_loc = u_odo_fog.mean(dim=0).mean(dim=0)
				u_imu_loc = u_imu.mean(dim=0).mean(dim=0)
			else:
				u_odo_fog_loc += u_odo_fog.mean(dim=0).mean(dim=0)
				u_imu_loc += u_imu.mean(dim=0).mean(dim=0)
			self.num_data += u_imu.shape[0]
		u_odo_fog_loc = u_odo_fog_loc/len(self.datasets_train)
		u_imu_loc = u_imu_loc/len(self.datasets_train)

		# second standard deviation
		u_length = 0
		for i, dataset in enumerate(self.datasets_train):
			with open(self.path_data_save + dataset + '.p', "rb") as file_pi:
				pickle_dict = pickle.load(file_pi)
			u_odo_fog = pickle_dict['u_odo_fog']
			u_imu = pickle_dict['u_imu']
			if i == 0:
				u_odo_fog_std = ((u_odo_fog-u_odo_fog_loc)**2).sum(dim=0).sum(dim=0)
				u_imu_std = ((u_imu-u_imu_loc)**2).sum(dim=0).sum(dim=0)
			else:
				u_odo_fog_std += ((u_odo_fog - u_odo_fog_loc)**2).sum(dim=0).sum(dim=0)
				u_imu_std += ((u_imu - u_imu_loc)**2).sum(dim=0).sum(dim=0)
			u_length += u_odo_fog.shape[0]*u_odo_fog.shape[1]
			u_odo_fog_std = (u_odo_fog_std/u_length).sqrt()
			u_imu_std = (u_imu_std/u_length).sqrt()

		# for constant measurements, set standard deviation to 1
		u_odo_fog_std[u_odo_fog_std == 0] = 1
		u_imu_std[u_imu_std == 0] = 1
		self.normalize_factors = {
							 'u_odo_fog_loc': u_odo_fog_loc,
							 'u_imu_loc': u_imu_loc,
							 'u_odo_fog_std': u_odo_fog_std,
							 'u_imu_std': u_imu_std,
							 }

		pickle_dict = {'normalize_factors': self.normalize_factors}
		with open(self.path_temp + "normalize_factors.p", "wb") as file_pi:
			pickle.dump(pickle_dict, file_pi)

	def normalize(self, x, var="u_odo_fog"):
		x_loc = self.normalize_factors[var + "_loc"]
		x_std = self.normalize_factors[var + "_std"]
		x_normalized = (x-x_loc)/x_std
		return x_normalized