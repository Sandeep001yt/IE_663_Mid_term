"""
model/TextImage.py

Drop-in replacement for AudioVideo.py for text+image modalities (Twitter15/17).

Architecture mirrors AVGBShareClassifier exactly:
  - AudioEncoder  → TextEncoder  (BERT-base, output projected to hidden_dim=512)
  - VideoEncoder  → ImageEncoder (ResNet-18, same as visual branch)
  - AVGBShareClassifier → TIGBShareClassifier (identical AUG classifier logic)

The AUG core (add_layer / classfier) is UNCHANGED — copy-pasted verbatim so
the boosting algorithm and adaptive classifier assignment work identically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


# ------------------------------------------------------------------ #
#  Modality Encoders                                                   #
# ------------------------------------------------------------------ #

class TextEncoder(nn.Module):
    """
    BERT encoder that produces a fixed-size 512-d vector per sample,
    matching the hidden_dim used by AudioEncoder's ResNet-18 output.

    Forward input:
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)
        token_type_ids : (B, seq_len)
    Forward output:
        (B, 512)  — projected [CLS] embedding
    """
    def __init__(self, config=None):
        super(TextEncoder, self).__init__()
        bert_path = config["dataset"].get("bert_path", "bert-base-uncased")
        self.bert = BertModel.from_pretrained(bert_path)
        bert_hidden = self.bert.config.hidden_size          # 768 for bert-base
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, 512),
            nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]      # (B, 768)
        return self.proj(cls_token)                         # (B, 512)


class ImageEncoder(nn.Module):
    """
    ResNet-50 visual encoder for single still images.
    Matches the paper which uses ResNet-50 as the image backbone.
    ResNet-50 output is 2048-d, projected down to 512-d to match hidden_dim.

    Forward input:  (B, 3, 224, 224)
    Forward output: (B, 512)
    """
    def __init__(self, config=None):
        super(ImageEncoder, self).__init__()
        import torchvision.models as models
        resnet = models.resnet50(pretrained=True)
        # Remove the final FC layer — keep up to avgpool
        self.image_net = nn.Sequential(*list(resnet.children())[:-1])
        # Project 2048 → 512 to match hidden_dim used by text encoder
        self.proj = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
        )

    def forward(self, image):
        v = self.image_net(image)       # (B, 2048, 1, 1)
        v = torch.flatten(v, 1)         # (B, 2048)
        v = self.proj(v)                # (B, 512)
        return v


# ------------------------------------------------------------------ #
#  Main Model — mirrors AVGBShareClassifier 1-to-1                    #
# ------------------------------------------------------------------ #

class TIGBShareClassifier(nn.Module):
    """
    Text-Image AUG classifier.

    Naming convention kept parallel to AVGBShareClassifier:
        audio_encoder → text_encoder   (is_a=True  branch = text)
        video_encoder → image_encoder  (is_a=False branch = image)

    The entire AUG mechanism (add_layer, classfier, boosting loss in
    train script) is identical to the audio-video version.
    """

    def __init__(self, config):
        super(TIGBShareClassifier, self).__init__()

        self.text_encoder  = TextEncoder(config)
        self.image_encoder = ImageEncoder(config)

        self.hidden_dim = 512

        # Embedding projections — one per modality (mirrors AV model)
        self.embedding_a = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )
        self.embedding_v = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
        )

        self.num_class = config['setting']['num_class']
        self.fc_out    = nn.Linear(256, self.num_class)

        # AUG adaptive layers — grown dynamically during training
        self.additional_layers_a = nn.ModuleList()   # text  (weak modality)
        self.additional_layers_v = nn.ModuleList()   # image (strong modality)

        self.relu = nn.ReLU()

    # -------------------------------------------------------------- #
    #  Forward — encodes both modalities                              #
    # -------------------------------------------------------------- #
    def forward(self, input_ids, attention_mask, token_type_ids, image):
        t_feature = self.text_encoder(input_ids, attention_mask, token_type_ids)
        v_feature = self.image_encoder(image)
        return t_feature, v_feature

    # -------------------------------------------------------------- #
    #  AUG: add_layer — VERBATIM from AVGBShareClassifier            #
    # -------------------------------------------------------------- #
    def add_layer(self, is_a=True):
        new_layer = nn.Linear(self.hidden_dim, 256).cuda()
        nn.init.xavier_normal_(new_layer.weight)
        nn.init.constant_(new_layer.bias, 0)
        if is_a:
            self.additional_layers_a.append(new_layer)
        else:
            self.additional_layers_v.append(new_layer)

    # -------------------------------------------------------------- #
    #  AUG: classfier — VERBATIM from AVGBShareClassifier            #
    # -------------------------------------------------------------- #
    def classfier(self, x, is_a=True):
        if is_a:
            result_a  = self.embedding_a(x)
            feature   = self.fc_out(result_a)
            o_fea     = feature
            add_fea   = None
            i         = 0
            layerlen  = len(self.additional_layers_a)
            for layer in self.additional_layers_a:
                addf    = self.relu(layer(x))
                add_fea = self.fc_out(addf)
                feature = feature + add_fea
                i += 1
                if i < layerlen:
                    o_fea = feature
        else:
            result_v  = self.embedding_v(x)
            feature   = self.fc_out(result_v)
            o_fea     = feature
            add_fea   = None
            j         = 0
            layerlen  = len(self.additional_layers_v)
            for layer in self.additional_layers_v:
                addf    = self.relu(layer(x))
                add_fea = self.fc_out(addf)
                feature = feature + add_fea
                j += 1
                if j < layerlen:
                    o_fea = feature
        return feature, o_fea, add_fea