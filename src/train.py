import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import matplotlib.pyplot as plt
from PIL import Image

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

# def evaluate(env, mode,agent, video, num_episodes, L, step, test_env=False):
#         episode_rewards = []
#         # aug_func1 = extract_low_freq(  
#         for i in range(num_episodes):
#                 obs = env.reset()
#                 video.init(enabled=(i==0))
#                 done = False
#                 episode_reward = 0
#                 while not done:
#                         with utils.eval_mode(agent):
#                                 # obs = aug_func1(obs.clone())
#                                 action = agent.select_action(obs)
#                         obs, reward, done, _ = env.step(action)
#                         video.record(env)
#                         episode_reward += reward

#                 if L is not None:
#                         _test_env = '_' + mode if test_env else ''
#                         video.save(f'{step}{_test_env}_{i}.mp4')
#                         L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
#                 episode_rewards.append(episode_reward)
#                 a=np.array(obs)[:3,:,:]
#                 a=np.transpose(a,(1,2,0))
#                 plt.imshow(a)
#                 plt.show()
#         return np.mean(episode_rewards)


def evaluate(env, mode, agent, video, num_episodes, L, step, video_dir, test_env=False):
    episode_rewards = []



    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        # GIF 저장을 위한 이미지 리스트
        images = []

        while not done:
            with utils.eval_mode(agent):
                # obs = aug_func1(obs.clone())  # 데이터 증강 시 사용 가능
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            # obs에서 첫 3채널을 깊은 복사로 가져옴
            a = obs[6:9, :, :].copy()
            a = np.transpose(a, (1, 2, 0))  # (3, H, W) -> (H, W, 3)

            # plt로 시각화하지 않고 바로 PIL 이미지로 변환
            img = (a - a.min()) / (a.max() - a.min()) * 255
            img = img.astype(np.uint8)
            image = Image.fromarray(img)
            images.append(image)  # 이미지 리스트에 추가


        # 평가 후 결과 저장
        if L is not None:
            _test_env = '_' + mode if test_env else ''
            # GIF로 저장
            gif_path = os.path.join(video_dir, f'{step}{_test_env}{i}.gif')
            if i%5==0:
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)


def main(args):
        # Set seed
        utils.set_seed_everywhere(args.seed)
        
        # Initialize environments
        gym.logger.set_level(40)
        env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='train'
        )


        test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='color_easy',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None
        test_env2 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+84,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='color_hard',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None


        test_env3 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+126,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='video_easy',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None

        test_env4 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+168,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='video_hard',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None    



        # Create working directory
        work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed)+'_'+args.tag)
        print('Working directory:', work_dir)
        if os.path.exists(work_dir):
            delete_option = input('working dir already exists, delete it? y or n :')
            if 'y' == delete_option:
                import shutil
                shutil.rmtree(work_dir)
            else:
                assert os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
        if not os.path.exists(work_dir):
                utils.make_dir(work_dir)
        model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
        video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
        video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
        # utils.write_info(args, os.path.join(work_dir, 'info.log'))

        #rint(torch.cuda.is_available())
        # Prepare agent
        assert torch.cuda.is_available(), 'must have cuda enabled'
        replay_buffer = utils.ReplayBuffer(
                obs_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                capacity=args.train_steps,
                batch_size=args.batch_size,
                args = args
        )
        cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
        print('Observations:', env.observation_space.shape)
        print('Cropped observations:', cropped_obs_shape)
        agent = make_agent(
                obs_shape=cropped_obs_shape,
                action_shape=env.action_space.shape,
                args=args
        )
        

        start_step, episode, episode_reward, done = 0, 0, 0, True
        L = Logger(work_dir)
        start_time = time.time()
        prev_episode_reward=0
        max_episode_reawrd=0
        for step in range(start_step, args.train_steps+1):
                if done:

                        if step > start_step:
                                L.log('train/duration', time.time() - start_time, step)
                                start_time = time.time()
                                L.dump(step)

                        # Evaluate agent periodically
                        if step % args.eval_freq == 0 and step>0:
                                print('Evaluating:', work_dir)
                                L.log('eval/episode', episode, step)
                                mode='train'
                                evaluate(env,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir)
                                if test_env is not None:
                                        mode='color_easy'
                                        evaluate(test_env,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='color_hard'
                                        evaluate(test_env2,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='video_easy'
                                        evaluate(test_env3,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='video_hard'
                                        evaluate(test_env4,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                L.dump(step)
                        # Save agent periodically
                        if step > start_step and step % args.save_freq == 0:
                              torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

                        L.log('train/episode_reward', episode_reward, step)

                        obs = env.reset()

                        if step<=1e5:

                                max_episode_reawrd=episode_reward
                                prev_episode_reward=episode_reward

                        else:
                               
                               if episode_reward>=max_episode_reawrd:
                                      max_episode_reawrd=episode_reward
                                      agent.initial_latent=torch.randn(1,4,5,5,dtype=torch.float16).to(torch.device("cuda:{}".format(args.gpu))).repeat(32,1,1,1)
                                      prev_episode_reward=episode_reward
                               else:
                                        if prev_episode_reward-episode_reward>=0:
                                                prev_episode_reward=episode_reward
                                                
                                        else:
                                                v_normalized=args.shift_ratio*(episode_reward-prev_episode_reward)/(max_episode_reawrd-prev_episode_reward)
                                                L.log('train/v_normalized', v_normalized, step)
                                                agent.initial_latent=(1/np.sqrt(v_normalized**2+1))*agent.initial_latent+(v_normalized/np.sqrt(v_normalized**2+1))*torch.randn(1,4,5,5,dtype=torch.float16).to(torch.device("cuda:{}".format(args.gpu))).repeat(32,1,1,1)
                                                prev_episode_reward=episode_reward

                        done = False
                        episode_reward = 0
                        episode_step = 0
                        episode += 1

                        L.log('train/episode', episode, step)

                # Sample action for data collection
                if step < args.init_steps:
                        action = env.action_space.sample()
                else:
                        with utils.eval_mode(agent):

                                action = agent.sample_action(obs)

                # Run training update
                if step >= args.init_steps:
                        num_updates = args.init_steps if step == args.init_steps else 1
                        if num_updates==1000 or step<=1e5:
                                for _ in range(num_updates):
                                        agent.update(replay_buffer, L, step)

                        else:
                               for _ in range(num_updates):
                                      agent.update2(replay_buffer,L,step)
                # Take step
                next_obs, reward, done, _ = env.step(action)
                done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
                replay_buffer.add(obs, action, reward, next_obs, done_bool)
                episode_reward += reward
                obs = next_obs

                episode_step += 1

        print('Completed training for', work_dir)


if __name__ == '__main__':
        args = parse_args()
        main(args)
