import sys
#sys.path.append('/home/pg/Desktop/DGA2_esj/src/algorithms')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from algorithms.diffuser import *
import utils
import algorithms.modules as m
from algorithms.rl_utils import *
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision
from flask import Flask, request, jsonify, send_file
import torch
import io

app = Flask(__name__)

generator=torch.manual_seed(42)

class PipelineWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "nota-ai/bk-sdm-small", torch_dtype=torch.float16, use_safetensors=False, safety_checker=None,
        ).to(self.device)

        self.pipeline.vae = AutoencoderTiny.from_pretrained(
            "sayakpaul/taesd-diffusers", torch_dtype=torch.float16, use_safetensors=False,
        ).to(self.device)

        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipeline.scheduler.config)
        self.negative_prompt_embeddings = torch.randn(1, 77, 768).to(self.device)
        self.prompt_embeddings = torch.randn(1, 77, 768).to(self.device)

    def generate_images(self, initial_latent):
        with torch.no_grad():
            generated_images = self.pipeline(
                num_inference_steps=3,
                generator=generator,
                num_images_per_prompt=32,
                width=40,
                height=40,
                prompt_embeds=self.prompt_embeddings,
                negative_prompt_embeds=self.negative_prompt_embeddings,
                output_type='pt',
                latents=initial_latent
            ).images

            background = (255 * generated_images).clamp(0, 255).cpu()

            buffer = io.BytesIO()
            torch.save(background, buffer)
            buffer.seek(0)

            return buffer

pipeline_wrapper = PipelineWrapper()

@app.route('/generate', methods=['POST'])
def generate():
    if 'initial_latent' not in request.files:
        return jsonify({"error": "No initial_latent file provided"}), 400

    # 클라이언트에서 받은 텐서를 로드
    initial_latent_file = request.files['initial_latent']
    initial_latent = torch.load(initial_latent_file, map_location='cuda').to(torch.float16)

    buffer = pipeline_wrapper.generate_images(initial_latent)

    # 생성된 텐서를 클라이언트로 전송
    return send_file(buffer, mimetype='application/x-pytorch')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--port',default=5000,type=int)
args = parser.parse_args()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=args.port)