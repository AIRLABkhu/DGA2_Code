import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
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

class SVEA_C(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.args=args
	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_conv(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea(n=128)


		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

	def update2(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea(n=64)

		if self.args.mask_type == "binary":
			obs_grad = compute_attribution2(self.critic, obs, self.select_action2(obs).detach())
			mask, _ = compute_attribution_mask3(obs_grad)
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
