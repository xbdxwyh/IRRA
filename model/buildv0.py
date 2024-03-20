from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, VisionTransformer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from transformers import BartConfig

from .attn import Config,SelfAttention,OutputLayer
from .bartmodel import BartForLM

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'proj' in args.loss_names:
            self.proj_dec_cfg = BartConfig(
                decoder_layers=1,
                d_model=self.embed_dim
            )
            self.proj_token_num = 16
            self.proj_prefix = nn.parameter(torch.randn((self.proj_token_num,self.embed_dim)))
            self.proj_dec = BartForLM(self.proj_dec_cfg)
            pass

        if 'fusion' in args.loss_names:
            self.dim=64
            self.fusion_config_ff = Config(hidden_size=self.dim,num_attention_heads=1)
            self.fusion_attn_ff = SelfAttention(config=self.fusion_config_ff)
            self.fusion_output_ff = OutputLayer(config=self.fusion_config_ff)
            
            if 'evafusion' in args.loss_names:
                self.fusion_config_all = Config(hidden_size=self.dim+self.embed_dim,num_attention_heads=1+self.embed_dim//64)
                self.fusion_attn_all = SelfAttention(config=self.fusion_config_all)
                self.fusion_output_all = OutputLayer(config=self.fusion_config_all)

                self.img_projection = nn.Linear(self.dim+self.embed_dim, self.embed_dim)
                nn.init.normal_(self.img_projection.weight.data, std=0.001)
                nn.init.constant_(self.img_projection.bias.data, val=0.0)

            if 'id' in args.loss_names:
                self.classifier_fusion = nn.Linear(self.dim, self.num_classes)
                nn.init.normal_(self.classifier_fusion.weight.data, std=0.001)
                nn.init.constant_(self.classifier_fusion.bias.data, val=0.0)

                pass
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if isinstance(self.base_model.visual, VisionTransformer):
            if 'evafusion' in self.current_task:
                x_ff = x[:,:,:self.dim]
                bs,seq_len,_ = x.shape
                attn_mask = torch.LongTensor([[1]]).repeat((bs,seq_len)).to(x.device)
                x_all_out = self.fusion_attn_all(hidden_states=torch.cat([x_ff,x],dim=-1),attention_mask = attn_mask[:, None, None, :])[0]
                x_all_out = self.fusion_output_all(x_all_out)
                # CLS feature of all image information
                x = self.img_projection(x_all_out[:, 0, :]).float()
                pass
            else:
                x = x[:, 0, :].float()
        else:
            x = x.float() # for CLIP ResNet visual model

        return x
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        # print(isinstance(self.base_model.visual, VisionTransformer))
        # print(self.base_model.visual)
        if isinstance(self.base_model.visual, VisionTransformer):
            i_feats = image_feats[:, 0, :].float()
        else:
            i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        
        if 'fusion' in self.current_task:
            # specific information extractor
            image_feats_ff = image_feats[:,:,:self.dim]
            bs,seq_len,_ = image_feats.shape
            attn_mask = torch.LongTensor([[1]]).repeat((bs,seq_len)).to(image_feats.device)
            image_feats_ff_out = self.fusion_attn_ff(hidden_states=image_feats_ff,attention_mask = attn_mask[:, None, None, :])[0]
            #print(image_feats_ff_out.dtype,image_feats_ff.dtype)
            image_feats_ff_out = self.fusion_output_ff(hidden_states = image_feats_ff_out,input_tensor=image_feats_ff)
            # CLS feature of these information
            i_feats_ff = image_feats_ff[:, 0, :].float()
            i_feats_ff_out = image_feats_ff_out[:, 0, :].float()
            t_feats_ff = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()[:,:self.dim]
            if 'evafusion' in self.current_task:
                # cat processed feature and feat
                image_feats_all_out = self.fusion_attn_all(hidden_states=torch.cat([image_feats_ff,image_feats],dim=-1),attention_mask = attn_mask[:, None, None, :])[0]
                image_feats_all_out = self.fusion_output_all(image_feats_all_out)
                # CLS feature of all image information
                i_feats_all_out = self.img_projection(image_feats_all_out[:, 0, :]).float()
            

            # ID loss and itc loss
            if 'itc' in self.current_task:
                ret.update({'itc_loss_ff':(objectives.compute_itc(i_feats_ff, t_feats_ff, logit_scale)+\
                                           objectives.compute_itc(i_feats_ff_out, t_feats_ff, logit_scale))*0.5})
                if 'evafusion' in self.current_task:
                    ret.update({'itc_loss_all': objectives.compute_itc(i_feats_all_out, t_feats, logit_scale)})

            if 'id' in self.current_task:
                image_logits_ff = self.classifier_fusion(i_feats_ff.half()).float()
                text_logits_ff = self.classifier_fusion(t_feats_ff.half()).float()
                image_logits_ff_out = self.classifier_fusion(i_feats_ff_out.half()).float()
                ret.update({'id_loss_ff':
                            (objectives.compute_id(image_logits_ff, text_logits_ff, batch['pids']) + \
                             objectives.compute_id(image_logits_ff_out, text_logits_ff, batch['pids']))*0.5*self.args.id_loss_weight
                    })

                image_pred_ff = torch.argmax(image_logits_ff, dim=1)
                image_pred_ff_out = torch.argmax(image_logits_ff_out, dim=1)
                text_pred_ff = torch.argmax(text_logits_ff, dim=1)

                image_precision_ff = (image_pred_ff == batch['pids']).float().mean()
                image_precision_ff_out = (image_pred_ff_out == batch['pids']).float().mean()
                text_precision_ff = (text_pred_ff == batch['pids']).float().mean()
                ret.update({'img_acc_ff': image_precision_ff})
                ret.update({'img_acc_ff_out': image_precision_ff_out})
                ret.update({'txt_acc_ff': text_precision_ff})

                if 'evafusion' in self.current_task:
                    image_logits_all = self.classifier(i_feats_all_out.half()).float()
                    ret.update({'id_loss_all':
                            objectives.compute_id(image_logits_all, text_logits, batch['pids'])*self.args.id_loss_weight
                    })
                    image_pred_all = torch.argmax(image_logits_all, dim=1)
                    image_precision_all = (image_pred_all == batch['pids']).float().mean()
                    ret.update({'img_acc_all': image_precision_all})
            pass

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    print(model.fusion_output_ff.LayerNorm.weight.dtype)
    return model
