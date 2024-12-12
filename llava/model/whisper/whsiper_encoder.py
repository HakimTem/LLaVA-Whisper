import torch
import torch.nn as nn

from transformers import WhisperModel, WhisperProcessor, WhisperConfig


class WhisperAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_audio_tower', False):
            self.load_model()
        else:
            self.cfg_only = WhisperConfig.from_pretrained(self.audio_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.audio_tower_name))
            return

        self.audio_processor = WhisperProcessor.from_pretrained(self.audio_tower_name)
        self.audio_tower = WhisperModel.from_pretrained(self.audio_tower_name, device_map=device_map)
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        # if self.select_feature == 'patch':
        #     audio_features = audio_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     audio_features = audio_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return audio_features

    @torch.no_grad()
    def forward(self, samples):
        if type(samples) is list:
            audio_features = []
            for sample in samples:
                audio_forward_out = self.audio_tower(sample.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(sample.dtype)
                audio_features.append(audio_feature)
        else:
            audio_forward_outs = self.audio_tower(samples.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(samples.dtype)

        return audio_features

    # @property
    # def dummy_feature(self):
    #     return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only