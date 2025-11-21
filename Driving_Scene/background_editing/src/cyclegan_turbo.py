import os
import sys
import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, download_url
from diffusers.models.attention_processor import AttnProcessor
from my_utils.cross_att import CrossViewAttnProcessor

class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = (_vae.decode(x / _vae.config.scaling_factor).sample).clamp(-1, 1)
        return x_decoded

class CycleGAN_Turbo(torch.nn.Module):
    def __init__(self, input_length,view_length, weather_type, with_seg=False,tv_att=False, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_unet=8, lora_rank_vae=4):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")
        if tv_att:
            unet.set_attn_processor(
                        processor=CrossViewAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=input_length,view_num=view_length))
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        self.unet, self.vae = unet, vae
     
        sd = torch.load(pretrained_path)
        self.load_ckpt_from_state_dict(sd,weather_type,with_seg)
        self.timesteps = torch.tensor([999], device="cuda").long()
        self.caption = None
        self.direction = None

        self.vae_enc.cuda()
        self.vae_dec.cuda()
        self.unet.cuda()

    def set_vae(self):

        old_conv_in = self.vae.encoder.conv_in
        new_in_channels = old_conv_in.in_channels + 1 
        out_channels = old_conv_in.out_channels
        kernel_size = old_conv_in.kernel_size
        stride = old_conv_in.stride
        padding = old_conv_in.padding

        new_conv_in = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        with torch.no_grad():
            new_conv_in.weight[:, :old_conv_in.in_channels, :, :] = old_conv_in.weight
            new_conv_in.weight[:, old_conv_in.in_channels:, :, :] = old_conv_in.weight.mean(dim=1, keepdim=True)
            new_conv_in.bias = old_conv_in.bias

        self.vae.encoder.conv_in = new_conv_in


    def load_ckpt_from_state_dict(self, sd,weather_type,with_seg=False):

        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_encoder"], lora_alpha=sd["rank_unet"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_target_modules_decoder"], lora_alpha=sd["rank_unet"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["l_modules_others"], lora_alpha=sd["rank_unet"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                blended_weight = sd[f"sd_{weather_type}_encoder"][name_sd]

                p.data.copy_(blended_weight)
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                blended_weight = sd[f"sd_{weather_type}_decoder"][name_sd]

                p.data.copy_(blended_weight)                
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                blended_weight = sd[f"sd_{weather_type}_other"][name_sd]

                p.data.copy_(blended_weight)

        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
        print(with_seg)
        if with_seg:
            self.set_vae()
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        self.vae.decoder.gamma = 1
        self.vae_b2a = copy.deepcopy(self.vae)
        self.vae_enc = VAE_encode(self.vae, vae_b2a=self.vae_b2a)
        self.vae_enc.load_state_dict(sd["sd_vae_enc"])
        self.vae_dec = VAE_decode(self.vae, vae_b2a=self.vae_b2a)

        self.vae_dec.load_state_dict(sd["sd_vae_dec"])

        self.vae_enc.vae.set_adapter(["vae_skip"])
        self.vae_enc.vae_b2a.set_adapter(["vae_skip"])

        self.vae_dec.vae.set_adapter(["vae_skip"])
        self.vae_dec.vae_b2a.set_adapter(["vae_skip"])

    def load_ckpt_from_url(self, url, ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
        outf = os.path.join(ckpt_folder, os.path.basename(url))
        download_url(url, outf)
        sd = torch.load(outf)
        self.load_ckpt_from_state_dict(sd)

    @staticmethod
    def forward_with_networks(x, x_label, direction, vae_enc, unet, vae_dec, sched, timesteps, text_emb, alpha, total_length, mode, with_seg):

        B = x.shape[0]
        assert direction in ["a2b", "b2a"]

        if mode == 'train':
            if alpha == 0:
                unet.set_adapters(["rainy_encoder", "rainy_decoder", "rainy_others"])

            elif alpha == 1:
                unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others"])
            elif alpha == 0.5:
                unet.set_adapters(["foggy_encoder", "foggy_decoder", "foggy_others"])
            else:
                pass
        if with_seg:
            x_cat = torch.cat((x, x_label), dim=1)
        else:
            x_cat = x
        x_enc = vae_enc(x_cat, direction=direction).to(x.dtype)
        if mode !='train':
            text_emb =text_emb.repeat(total_length, 1, 1)
            timesteps = timesteps.repeat(total_length) 

        model_pred = unet(x_enc, timesteps, encoder_hidden_states=text_emb,).sample

        x_out = torch.stack([sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample for i in range(B)])
        x_out_decoded = vae_dec(x_out, direction=direction)
        x_out = x_out_decoded 
        return x_out

    @staticmethod
    def get_traininable_params(unet, vae_a2b, vae_b2a):
        # add all unet parameters
        params_gen = list(unet.conv_in.parameters())
        unet.conv_in.requires_grad_(True)
        
        unet.set_adapters(["snowy_encoder", "snowy_decoder", "snowy_others","rainy_encoder", "rainy_decoder", "rainy_others","foggy_encoder", "foggy_decoder", "foggy_others"])
        unet.lora_params = {
            'rainy': [p for n, p in unet.named_parameters() if "rainy" in n and "lora" in n],
            'snowy': [p for n, p in unet.named_parameters() if "snowy" in n and "lora" in n],
            'foggy': [p for n, p in unet.named_parameters() if "foggy" in n and "lora" in n]
        }
        params_gen += unet.lora_params['rainy'] + unet.lora_params['snowy'] +unet.lora_params['foggy'] 

 
        for n,p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        for n,p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())
        return params_gen

    def forward(self, x_t, x_t_label, alpha, total_length, mode=None, direction=None, caption=None, caption_emb=None, with_seg=False):

        if direction is None:
            assert self.direction is not None
            direction = self.direction
        if caption is None and caption_emb is None:
            assert self.caption is not None
            caption = self.caption
        if caption_emb is not None:
            caption_enc = caption_emb
        else:
            caption_tokens = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt").input_ids.to(x_t.device)
            

            caption_enc = self.text_encoder(caption_tokens)[0].detach().clone()

        return self.forward_with_networks(
            x_t, x_t_label, direction, self.vae_enc, self.unet, self.vae_dec, self.sched, self.timesteps, caption_enc, alpha, total_length, mode,with_seg)