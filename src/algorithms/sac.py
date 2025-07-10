import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from .rl_utils import *
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision
import io
import requests
import os
import matplotlib.pyplot as plt
def get_background_image_tensor(initial_latent,port):
    buffer = io.BytesIO()
    torch.save(initial_latent.cpu(), buffer)
    buffer.seek(0)

    url = "http://localhost:{}/generate".format(port)
    files = {'initial_latent': ('initial_latent.pt', buffer, 'application/x-pytorch')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        response_buffer = io.BytesIO(response.content)
        generated_images = torch.load(response_buffer)
        return generated_images
    else:
        raise ValueError(f"Server returned an error: {response.status_code}, {response.text}")


class SAC(object):
	def __init__(self, obs_shape, action_shape, args):
		self.device=torch.device("cuda:{}".format(args.gpu))
		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq
		self.args=args
		shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).to(self.device)
		head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).to(self.device)
		actor_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		critic_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)

		self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).to(self.device)
		self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).to(self.device)
		self.critic_target = deepcopy(self.critic)

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)

		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)

		self.train()
		self.critic_target.train()
		self.initial_latent=torch.randn(1,4,5,5,dtype=torch.float16).to(torch.device("cuda:{}".format(args.gpu))).repeat(32,1,1,1)
		if not os.path.exists("{}_{}_{}_{}_{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag)):
			os.mkdir("{}_{}_{}_{}_{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag))
		if not os.path.exists("{}_{}_{}_{}_{}_saliency".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag)):
			os.mkdir("{}_{}_{}_{}_{}_saliency".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag))
	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def eval(self):
		self.train(False)

	@property
	def alpha(self):
		return self.log_alpha.exp()
		
	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).to(self.device)
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
		return mu.cpu().data.numpy().flatten()

	def select_action2(self, obs):
		with torch.no_grad():
			mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
		return mu
	
	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
		return pi.cpu().data.numpy().flatten()

	def sample_heatmap(self,obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			output = self.actor.show_headmap(_obs)
		return output.detach().cpu().numpy()

	def sample_latent(self,obs):
		output = self.actor.encoder(obs)

		return output
	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)



		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1,
								 target_Q) + F.mse_loss(current_Q2, target_Q)

		#semantic_loss=nn.MSELoss(reduction='sum')
		#semantic_distance=0.5*semantic_loss(self.critic.encoder(obs[self.args.batch_size:,:,:,:]).detach(),self.critic.encoder(obs[:self.args.batch_size,:,:,:]))
		#critic_loss-=self.args.adversarial_alpha*semantic_distance

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		_, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

	def soft_update_critic_target(self):
		utils.soft_update_params(
			self.critic.Q1, self.critic_target.Q1, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.Q2, self.critic_target.Q2, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.encoder, self.critic_target.encoder,
			self.encoder_tau
		)


	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()


		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

	def update2(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac(n=64)
		
		if self.args.mask_type == "binary":
			obs_grad = compute_attribution2(self.critic, obs, self.select_action2(obs).detach())
			mask = compute_attribution_mask(obs_grad)
		elif self.args.mask_type == "exp":
			obs_grad = compute_attribution2(self.critic, obs, self.select_action2(obs).detach())
			mask = compute_attribution_mask_exp(obs_grad, 0.3)
		elif self.args.mask_type == "log":
			obs_grad = compute_attribution2(self.critic, obs, self.select_action2(obs).detach())
			mask = compute_attribution_mask_log(obs_grad)



		background = get_background_image_tensor(self.initial_latent,port=self.args.port)
		background = background.to(self.device).to(torch.float32)

		background = torch.nn.functional.interpolate(background, size=(84, 84), mode='bilinear', align_corners=False)
		background = background.repeat(2, 3, 1, 1)


		aug_obs = obs.clone() * mask + (1.0-mask) * background
		threshold = 1e-6

		result = (mask.abs() < threshold).all(dim=3).all(dim=2)
		count = result.sum().item()

		obs_grad = compute_attribution2(self.critic, aug_obs, self.select_action2(aug_obs).detach())
		mask2 = compute_attribution_mask_exp(obs_grad, 0.3)

		if count != 0:
			print(count, step)

		if step%10000==0:
			if not os.path.exists("{}_{}_{}_{}_{}/{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step)):
				os.mkdir("{}_{}_{}_{}_{}/{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step))
			for i in range(aug_obs.shape[0]):
				for j in range(3):
					a=aug_obs[i,3*j:3*(j+1),:,:].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
					plt.imsave("{}_{}_{}_{}_{}/{}/{}_{}.png".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step,i,j),a)


		if step%10000==0:
			if not os.path.exists("{}_{}_{}_{}_{}_saliency/{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step)):
				os.mkdir("{}_{}_{}_{}_{}_saliency/{}".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step))
			for i in range(aug_obs.shape[0]):
				for j in range(3):
					a=mask[i,3*j:3*(j+1),:,:].permute(1,2,0).detach().cpu().numpy()
					plt.imsave("{}_{}_{}_{}_{}_saliency/{}/{}_{}.png".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step,i,j),a)

					a=mask2[i,3*j:3*(j+1),:,:].permute(1,2,0).detach().cpu().numpy()
					plt.imsave("{}_{}_{}_{}_{}_saliency/{}/{}_{}_3.png".format(self.args.algorithm,self.args.seed,self.args.domain_name,self.args.task_name,self.args.tag,step,i,j),a)

		obs = torch.cat([obs, aug_obs], dim=0)
		action = torch.cat([action, action], dim=0)
		reward = torch.cat([reward, reward], dim=0)
		next_obs = torch.cat([next_obs, next_obs], dim=0)
		not_done = torch.cat([not_done, not_done], dim=0)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()