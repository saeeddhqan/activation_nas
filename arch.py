import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random, numpy

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)


x = torch.linspace(-2*np.pi, 2*np.pi, 100).view(-1, 1)
y = torch.sin(x)

# Step 2: Define the MLP Model
class MLP(nn.Module):
	def __init__(self, act, nlayer=16):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(1, nlayer)
		self.act = act
		self.fc2 = nn.Linear(nlayer, 1)

	def forward(self, x):
		with torch.no_grad():
			x = x / 5
		x = self.act(self.fc1(x))
		return self.fc2(x)


def train(act, num_epochs=751):
	q = 0
	for nl in (16, 8, 4):
		for lr in (0.01, 0.05):
			out = 0
			model = MLP(act)
			criterion = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr=lr)
			for epoch in range(num_epochs):
				optimizer.zero_grad(set_to_none=True)
				output = model(x)
				loss = criterion(output, y)
				litem = loss.item()
				loss.backward()
				optimizer.step()
				if epoch >= 500 and epoch <= 750:
					out += litem
			q += abs(out / 250)
	return q / 6
