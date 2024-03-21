from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights, VisionTransformer
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import copy

from transformers import BartConfig

from .attn import Config,SelfAttention,OutputLayer
from .bartmodel import BartForLM

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.proj_token_num = args.proj_token_num
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

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

        if 'proj' in args.loss_names:
            embed_dim = self.base_model.visual.transformer.width
            self.cross_attn = nn.MultiheadAttention(embed_dim,
                                                    embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(embed_dim)
            self.ln_pre_i = LayerNorm(embed_dim)
            self.ln_post = LayerNorm(embed_dim)

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

            self.proj_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, self.embed_dim))]))
            
            # # init proj_head head
            # nn.init.normal_(self.proj_head.dense.weight, std=fc_std)
            # nn.init.normal_(self.proj_head.fc.weight, std=proj_std)
            ## concat feature by token
            #### V2 62
            # self.cross_attn_mm = nn.MultiheadAttention(self.embed_dim,
            #                                         self.embed_dim // 64,
            #                                         batch_first=True)
            # self.cross_modal_transformer_mm = Transformer(width=self.embed_dim,
            #                                            layers=args.cmt_depth,
            #                                            heads=self.embed_dim //
            #                                            64)
            # scale = self.cross_modal_transformer_mm.width**-0.5
            
            # self.ln_pre_t_mm = LayerNorm(self.embed_dim)
            # self.ln_pre_i_mm = LayerNorm(self.embed_dim)
            # self.ln_post_mm = LayerNorm(self.embed_dim)

            # proj_std = scale * ((2 * self.cross_modal_transformer_mm.layers)**-0.5)
            # attn_std = scale
            # fc_std = (2 * self.cross_modal_transformer_mm.width)**-0.5
            # for block in self.cross_modal_transformer_mm.resblocks:
            #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # # init cross attn
            # nn.init.normal_(self.cross_attn_mm.in_proj_weight, std=attn_std)
            # nn.init.normal_(self.cross_attn_mm.out_proj.weight, std=proj_std)
            ### V3
            self.cross_attn_mm = copy.deepcopy(self.base_model.visual.transformer.resblocks[-1])

            self.proj_dec_cfg = BartConfig(
                decoder_layers=1,
                d_model=embed_dim
            )
            
            self.proj_prefix = nn.Parameter(torch.randn((self.proj_token_num,embed_dim)))
            self.proj_dec = BartForLM(self.proj_dec_cfg)
            pass


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

# # V2 func
#     def cross_former_mm(self, q, k, v):
#         x = self.cross_attn_mm(
#                 self.ln_pre_i_mm(q),
#                 self.ln_pre_i_mm(k),
#                 self.ln_pre_i_mm(v),
#                 need_weights=False)[0]
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.cross_modal_transformer_mm(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post_mm(x)
#         return x
    def cross_former_mm(self,q,k,v):
        # V3 func
        x = self.cross_attn_mm(q)
        return x
    
    def encode_pair_image(self, image, image_pair):
        bs = image.shape[0]
        # x_self = self.base_model.encode_image(image).half()
        x_self_conv = self.base_model.visual.conv_embedding(image.half())
        # x_self = self.base_model.visual.forward_body(x_self_conv)

        #x_pair = self.base_model.encode_image(image_pair).half()
        x_pair_conv = self.base_model.visual.conv_embedding(image_pair.half())
        y_gt = self.select_topk_token(x_pair_conv[:,1:,:],x_pair_conv[:, :1, :],self.proj_token_num)
        # x_pair = self.base_model.visual.forward_body(x_pair_conv)


        # i_y_feats = x_pair[:, 0, :]
        # sim_tkn = torch.nn.functional.cosine_similarity(i_y_feats.unsqueeze(1),x_pair[:,1:,:],dim=-1)
        # _, idx = torch.sort(sim_tkn,dim=1)
        # y_idx = idx[:,-self.proj_token_num:].reshape(-1)+1
        # x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,self.proj_token_num).reshape(-1)
        # y_gt = x_pair[x_idx,y_idx]

        x_add = torch.cat([y_gt.reshape(bs,self.proj_token_num,-1),x_self_conv],dim=1)
        x_add = self.base_model.visual.forward_body(x_add)
        #i_add = self.proj_head(torch.cat([i_feats.to(i_y_feats.dtype),i_y_feats],dim=-1)).float()
        # x_add = torch.cat([image_feats[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),image_feats[:, 1:, :]],dim=1)
        # x_add = self.cross_former_mm(x_add,x_add,x_add)
        # x_add = x_add @ self.base_model.visual.proj

        # x_add = torch.cat([x_self[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),x_self[:, 1:, :]],dim=1)
        # x_add = self.cross_former_mm(x_add,x_add,x_add)
        #x = self.proj_head(x_add[:, 0, :]).float()
        x = (x_add @ self.base_model.visual.proj)[:, 0, :].float()

        #x = self.proj_head(torch.cat([x_self,x_pair],dim=-1)).float()
        return x 
    
    # def encode_pair_image(self, image, image_pair):
    #     x_self = self.base_model.encode_image(image)[:, 0, :].half()
    #     x_pair = self.base_model.encode_image(image_pair)[:, 0, :].half()
    #     x = self.proj_head(torch.cat([x_self,x_pair],dim=-1)).float()
    #     return x 
    #     pass
        

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        if isinstance(self.base_model.visual, VisionTransformer):
            if 'evafusion' in self.current_task:
                x_ff = x[:,:,:self.dim]
                bs,seq_len,_ = x.shape
                attn_mask = torch.LongTensor([[1]]).repeat((bs,seq_len)).to(x.device)
                x_all_out = self.fusion_attn_all(hidden_states=torch.cat([x_ff,x],dim=-1),attention_mask = attn_mask[:, None, None, :])[0]
                x_all_out = self.fusion_output_all(hidden_states = x_all_out,input_tensor = torch.cat([x_ff,x],dim=-1))
                # CLS feature of all image information
                x = self.img_projection(x_all_out[:, 0, :]).float()
                pass
            else:
                x = x[:, 0, :].float()
        else:
            x = x.float() # for CLIP ResNet visual model

        return x
        # return x.float() # for CLIP ResNet visual model
    
    def select_topk_token(self,feats,cls,k=16,add_cls=False):
        bs = feats.shape[0]

        sim_tkn = torch.nn.functional.cosine_similarity(cls,feats,dim=-1)
        _, idx = torch.sort(sim_tkn,dim=1)
        y_idx = idx[:,-k:].reshape(-1)
        if add_cls:
            y_idx += 1

        x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,k).reshape(-1)
        selected = feats[x_idx,y_idx]
        return selected
    
    def forward_proj(self,image_feats):
        bs = image_feats.shape[0]
        x = self.cross_former(
            self.proj_prefix.unsqueeze(0).repeat(bs,1,1).to(image_feats.dtype),
                image_feats, 
                image_feats
            )
        x_casual = self.proj_dec(
                inputs_embeds = x,
                is_casual=True
            )
        x_attn = self.proj_dec(
            inputs_embeds = x,
            is_casual=False
        )
        x_dec = (x_casual[0]+x_attn[0])*0.5
        return x_dec
        pass

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        images = batch['images']
        bs = images.shape[0]
        caption_ids = batch['caption_ids']

        text_feats = self.base_model.encode_text(caption_ids)
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        image_feats_conv = self.base_model.visual.conv_embedding(images.half())

        # image_feats = self.base_model.visual.forward_body(image_feats_conv)
        #image_feats, text_feats = self.base_model(images, caption_ids)
        # print(isinstance(self.base_model.visual, VisionTransformer))
        # print(self.base_model.visual)
        # if isinstance(self.base_model.visual, VisionTransformer):
        #     i_feats = (image_feats @ self.base_model.visual.proj)[:, 0, :].float()
        # else:
        #     i_feats = image_feats.float() # for CLIP ResNet visual model
        
        if "proj" in self.current_task:
            x_dec = self.forward_proj(image_feats_conv)
            x_dec = x_dec.reshape(bs*self.proj_token_num,-1)

            # Select top k token by cosine similarity
            y_pair = batch['pair_img']
            # y_pair_feats = self.base_model.encode_image(y_pair)
            y_pair_feats_conv = self.base_model.visual.conv_embedding(y_pair.half())
            # y_pair_feats = self.base_model.visual.forward_body(y_pair_feats_conv)

            # i_y_feats = y_pair_feats[:, 0, :]
            # sim_tkn = torch.nn.functional.cosine_similarity(i_y_feats.unsqueeze(1),y_pair_feats[:,1:,:],dim=-1)
            # _, idx = torch.sort(sim_tkn,dim=1)
            # y_idx = idx[:,-self.proj_token_num:].reshape(-1)+1
            # x_idx = torch.arange(bs).unsqueeze(0).reshape(-1,1).repeat(1,self.proj_token_num).reshape(-1)
            # y_gt = y_pair_feats[x_idx,y_idx]
            #####
            y_gt = self.select_topk_token(y_pair_feats_conv[:,1:,:],y_pair_feats_conv[:, :1, :],self.proj_token_num)
            # add L1 and L2 Loss
            
            loss_proj = torch.nn.functional.l1_loss(x_dec,y_gt) + torch.nn.functional.mse_loss(x_dec,y_gt)

            x_add = torch.cat([y_gt.reshape(bs,self.proj_token_num,-1),image_feats_conv],dim=1).half()
            # print(image_feats_conv.shape)
            #print(y_gt.reshape(bs,self.proj_token_num,-1).shape)
            x_add = self.base_model.visual.forward_body(x_add)
            #i_add = self.proj_head(torch.cat([i_feats.to(i_y_feats.dtype),i_y_feats],dim=-1)).float()
            # x_add = torch.cat([image_feats[:, :1, :],y_gt.reshape(bs,self.proj_token_num,-1),image_feats[:, 1:, :]],dim=1)
            # x_add = self.cross_former_mm(x_add,x_add,x_add)
            x_add = x_add @ self.base_model.visual.proj
            #i_add = self.proj_head(x_add[:, 0, :]).float()
            i_feats = x_add[:, 0, :].float()

            # if 'itc' in self.current_task:
            #     ret.update({'itc_add_loss':objectives.compute_itc(i_add, t_feats, logit_scale)})

            ret.update({'loss_proj':loss_proj})
        
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
        
        if 'fusion' in self.current_task or 'evafusion' in self.current_task:
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
                image_feats_all_out = self.fusion_output_all(hidden_states = image_feats_all_out,input_tensor = torch.cat([image_feats_ff,image_feats],dim=-1))
                # CLS feature of all image information
                i_feats_all_out = self.img_projection(image_feats_all_out[:, 0, :]).float()
            

            # ID loss and itc loss
            if 'itc' in self.current_task:
                ret.update({'itc_loss_ff':(objectives.compute_itc(i_feats_ff, t_feats_ff, logit_scale)+\
                                           objectives.compute_itc(i_feats_ff_out, t_feats_ff, logit_scale))*0.5})
                if 'evafusion' in self.current_task:
                    # ret.update({'itc_loss_all': objectives.compute_itc(i_feats_all_out, t_feats, logit_scale)})
                    ret.update({'itc_loss_all': (objectives.compute_itc(i_feats_all_out, t_feats, logit_scale) + \
                                                 objectives.compute_itc(i_feats_all_out, i_feats, logit_scale) )*0.5})

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

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        # if 'mlm' in self.current_task:
        #     mlm_ids = batch['mlm_ids']

        #     mlm_feats = self.base_model.encode_text(mlm_ids)

        #     x = self.cross_former(mlm_feats, image_feats, image_feats)

        #     x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

        #     scores = x.float().reshape(-1, self.args.vocab_size)
        #     mlm_labels = batch['mlm_labels'].reshape(-1)
        #     ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

        #     pred = scores.max(1)[1]
        #     mlm_label_idx = torch.nonzero(mlm_labels)
        #     acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
        #     ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    #print(model.fusion_output_ff.LayerNorm.weight.dtype)
    return model
