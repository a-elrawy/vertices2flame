import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .base_models import Transformer, LinearEmbedding, PositionalEncoding


class FlameInverter(nn.Module):
    
    def __init__(self, from_pretrained=True, args=None, load_path=None):
        super().__init__()
        if from_pretrained: 
            args= None
            
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args)
        self.args = args

        if from_pretrained:
            # Download and put it in the same directory as this file.
            load_path = os.path.join(os.path.dirname(__file__), 'checkpoints/inverter.pth.tar')
            # Download the file to load_path
            if not os.path.isfile(load_path):
                print("=> Downloading checkpoint '{}'".format(load_path))
                if not os.path.exists(os.path.dirname(load_path)):
                    os.makedirs(os.path.dirname(load_path))
                os.system(f"wget -O {load_path} https://api.wandb.ai/files/yatek/Codetalker/q3xuo3pd/vox/vox_stage3.pth.tar?_gl=1*1ih8w3r*_ga*MTMwMjc1OTk5NC4xNjk0MDg0MzE5*_ga_JH1SJHJQXJ*MTY5NDA5OTA1NC4zLjAuMTY5NDA5OTA1NC42MC4wLjA")
        
        if load_path is not None:
            
            if os.path.isfile(load_path):
                print("=> loading checkpoint '{}'".format(load_path))

                checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage.cpu())
                self.load_state_dict(checkpoint['state_dict'], strict=False)

                print("=> loaded checkpoint '{}'".format(load_path))
            else:
                raise RuntimeError("=> no checkpoint flound at '{}'".format(load_path))




    def encode(self, x, x_a=None):
        h = self.encoder(x) ## x --> z'
        return h


    def decode(self, features):
        pose, exp = self.decoder(features) ## z' --> x
        return pose, exp

    def forward(self, x):
        ###x.shape: [B, L C]
        features = self.encode(x)
        ### quant [B, C, L]
        pose, exp = self.decode(features)

        return pose, exp


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args=None):
    super().__init__()
    self.args = args
    size = self.args.in_dim if args else 15069
    dim = self.args.hidden_size if args else 1024
    neg = self.args.neg if args else 0.2
    affine = self.args.INaffine if args else False
    num_hidden_layers = self.args.num_hidden_layers if args else 6
    num_attention_heads = self.args.num_attention_heads if args else 8
    intermediate_size = self.args.intermediate_size if args else 1536
    self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(neg, True))
    layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(neg, True),
                    nn.InstanceNorm1d(dim, affine=affine)
                )]

    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=dim,
        hidden_size=dim,
        num_hidden_layers=\
                num_hidden_layers,
        num_attention_heads=\
                num_attention_heads,
        intermediate_size=\
                intermediate_size)
    self.encoder_pos_embedding = PositionalEncoding(
        dim)
    self.encoder_linear_embedding = LinearEmbedding(
        dim,
        dim)

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.vertice_mapping(inputs)
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args=None, is_audio=False):
    super().__init__()
    self.args = args
    size = self.args.in_dim if args else 15069
    dim = self.args.hidden_size if args else 1024
    neg = self.args.neg if args else 0.2
    affine = self.args.INaffine if args else False
    num_hidden_layers = self.args.num_hidden_layers if args else 6
    num_attention_heads = self.args.num_attention_heads if args else 8
    intermediate_size = self.args.intermediate_size if args else 1536

    self.decoder_transformer = Transformer(
        in_size=dim,
        hidden_size=dim,
        num_hidden_layers=\
            num_hidden_layers,
        num_attention_heads=\
            num_attention_heads,
        intermediate_size=\
            intermediate_size)
    self.decoder_pos_embedding = PositionalEncoding(
        dim)
    self.decoder_linear_embedding = LinearEmbedding(
        dim,
        dim)

    self.pose_decoder = nn.Linear(dim, 6)
    self.exp_decoder = nn.Linear(dim, 50)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pose_out = self.pose_decoder(decoder_features)
    exp_out = self.exp_decoder(decoder_features)

    return pose_out, exp_out