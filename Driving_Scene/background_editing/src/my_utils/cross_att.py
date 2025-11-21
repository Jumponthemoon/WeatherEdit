import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND
import matplotlib.pyplot as plt
import os

def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    ref_frame_index = torch.tensor(ref_frame_index,device='cuda:0')
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)

    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")

    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")
    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross) 

    return hidden_states_ref_cross


class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2, view_num=5):
        self.unet_chunk_size = unet_chunk_size
        self.view_num = view_num
        self.self_attn_coeff = self_attn_coeff
        self.attention_maps = []  # Store all attention maps

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):
        hidden_chunks = torch.chunk(hidden_states, self.unet_chunk_size, dim=0)
        args = () if USE_PEFT_BACKEND else (scale,)
        if encoder_hidden_states is not None:
            enc_hidden_chunks = torch.chunk(encoder_hidden_states, self.unet_chunk_size, dim=0)
        else:
            enc_hidden_chunks = [None]*10
            split_size = hidden_states.shape[0] // 3

            first_part = hidden_states[:split_size]
            last_part = hidden_states[2*split_size:]
            first_result = first_part
            last_result = last_part

            key_last_frame = attn.to_k(last_result, *args)
            value_last_frame = attn.to_v(last_result, *args)
            
            key_last_frame = attn.head_to_batch_dim(key_last_frame)
            value_last_frame = attn.head_to_batch_dim(value_last_frame)
         
            key_first_frame = attn.to_k(first_result, *args)
            value_first_frame = attn.to_v(first_result, *args)
            
            key_first_frame = attn.head_to_batch_dim(key_first_frame)
            value_first_frame = attn.head_to_batch_dim(value_first_frame)

        final_result = []
        for i, (hidden_states,encoder_hidden_states) in enumerate(zip(hidden_chunks,enc_hidden_chunks)):

                batch_size, sequence_length, _ = ( 
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

                query = attn.to_q(hidden_states, *args)
                is_cross_attention = encoder_hidden_states is not None
  
        
                if encoder_hidden_states is None:
                    encoder_hidden_states = hidden_states
                elif attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

                key = attn.to_k(encoder_hidden_states, *args)
                value = attn.to_v(encoder_hidden_states, *args)

                query = attn.head_to_batch_dim(query) 

                
                if not is_cross_attention:
                    key_self = attn.head_to_batch_dim(key)
                    value_self = attn.head_to_batch_dim(value)
                    attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
                    hidden_states_self = torch.bmm(attention_probs, value_self)
      
                    video_length = key.size()[0] 
                    if self.view_num ==5:
                        ref0_frame_index = [0,0,0,1,2] 
                    elif self.view_num ==3: 
                        ref0_frame_index = [0,0,0]  
                    else:
                        ref0_frame_index =[1,0]       
                    hidden_states_ref_spatial = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask)

                    attention_probs_temporal_first = attn.get_attention_scores(query, key_first_frame, attention_mask)
                    hidden_states_temporal_first = torch.bmm(attention_probs_temporal_first, value_first_frame)

                    attention_probs_temporal_last = attn.get_attention_scores(query, key_last_frame, attention_mask)
                    hidden_states_temporal_last = torch.bmm(attention_probs_temporal_last, value_last_frame)

                    hidden_states_temporal = 0.5 * hidden_states_temporal_first + 0.5 * hidden_states_temporal_last
                    hidden_states_ref0 = 0.7 * hidden_states_ref_spatial + 0.3 * hidden_states_temporal


                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)
                hidden_states_ref4 = torch.bmm(attention_probs, value)


                hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff)* hidden_states_ref0 if not is_cross_attention else hidden_states_ref4 
                hidden_states = attn.batch_to_head_dim(hidden_states)

                hidden_states = attn.to_out[0](hidden_states, *args)
                hidden_states = attn.to_out[1](hidden_states)
                hidden_states = hidden_states / attn.rescale_output_factor

                final_result.append(hidden_states)

        hidden_states = torch.cat(final_result,dim=0)
        return hidden_states

