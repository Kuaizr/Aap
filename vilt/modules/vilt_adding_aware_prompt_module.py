import torch
import torch.nn as nn
import pytorch_lightning as pl
import vilt.modules.vision_transformer_prompts as vit

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from vilt.modules import heads, objectives, vilt_utils

class GateModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GateModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size*2, 1)
    
    def forward(self, input1, input2):
        embedded1 = self.embedding(input1)
        embedded2 = self.embedding(input2)
        gate_input = torch.cat((embedded1, embedded2), dim=1)
        gate_output = torch.sigmoid(self.gate(gate_input))
        return gate_output

class PoolingModel(nn.Module):
    def __init__(self,window_size):
        super(PoolingModel, self).__init__()
        self.pooling = nn.AvgPool1d(window_size)  # 池化窗口大小为64，可以根据需要进行调整
    
    def forward(self, input):
        output = self.pooling(input.permute(0, 2, 1))  # 将输入的维度顺序调整为（batch_size, 768, 1024）
        output = output.permute(0, 2, 1)  # 将输出的维度顺序调整回（batch_size, 1024, 768）
        return output

class ViLTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.ji = 0
        self.jt = 0
        self.gt = 0
        self.gi = 0
        self.save_hyperparameters()

        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )

        self.text_gate = GateModel(16*768, config["hidden_size"])
        self.text_pool = PoolingModel(64)
        self.image_gate = GateModel(7*768, config["hidden_size"])
        self.image_pool = PoolingModel(31)
        
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        if self.hparams.config["load_path"] == "":
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                pretrained=False, config=self.hparams.config
            )

        self.pooler = heads.Pooler(config["hidden_size"])
        self.pooler.apply(objectives.init_weights)

        if config["loss_names"]["mlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(config["hidden_size"])
            self.itm_score.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_score = heads.MPPHead(bert_config)
            self.mpp_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
            and not self.hparams.config["finetune_first"]
        ):
# 
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            # since the pre-trained max_text_len is 40,
            # we upsample the weight of position embedding to determined max_text_len
            if config["max_text_len"] != 40:
                state_dict['text_embeddings.position_ids'] = torch.Tensor(range(config["max_text_len"])).long().view(1,-1)
                pos_emb = state_dict['text_embeddings.position_embeddings.weight']
                pos_emb = torch.nn.functional.interpolate(pos_emb.view(1,1,40,768), size=(config["max_text_len"],768), mode='bilinear').squeeze()
                state_dict['text_embeddings.position_embeddings.weight'] = pos_emb
            self.load_state_dict(state_dict, strict=False)

        hs = self.hparams.config["hidden_size"]

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.hatememes_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.food101_classifier.apply(objectives.init_weights)               
            
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Sequential(
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, cls_num),
            )
            self.mmimdb_classifier.apply(objectives.init_weights)  
            
        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)            
            print("use pre-finetune model")
  
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1
        from timm.models.layers import trunc_normal_

        no_adding_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        no_adding_prompt[:,0:1,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            no_adding_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.no_adding_prompt = nn.Parameter(no_adding_prompt)

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,1:2,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_text_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)

        adding_gen_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        adding_gen_text_prompt[:,3:4,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            adding_gen_text_prompt[:,prompt_length//2+3:prompt_length//2+4,:].fill_(1)
        self.adding_gen_text_prompt = nn.Parameter(adding_gen_text_prompt)

        adding_jiansuo_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        adding_jiansuo_text_prompt[:,4:5,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            adding_jiansuo_text_prompt[:,prompt_length//2+4:prompt_length//2+5,:].fill_(1)
        self.adding_jiansuo_text_prompt = nn.Parameter(adding_jiansuo_text_prompt)

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,2:3,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_img_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)

        adding_gen_image_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        adding_gen_image_prompt[:,5:6,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            adding_gen_image_prompt[:,prompt_length//2+5:prompt_length//2+6,:].fill_(1)
        self.adding_gen_image_prompt = nn.Parameter(adding_gen_image_prompt)

        adding_jiansuo_image_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        adding_jiansuo_image_prompt[:,6:7,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            adding_jiansuo_image_prompt[:,prompt_length//2+6:prompt_length//2+7,:].fill_(1)
        self.adding_jiansuo_image_prompt = nn.Parameter(adding_jiansuo_image_prompt)


        if not self.learnt_p:
            self.no_adding_prompt.requires_grad=False
            self.missing_img_prompt.requires_grad=False
            self.missing_text_prompt.requires_grad=False
            self.adding_gen_text_prompt.requires_grad=False
            self.adding_jiansuo_text_prompt.requires_grad=False
            self.adding_gen_image_prompt.requires_grad=False
            self.adding_jiansuo_image_prompt.requires_grad=False

        # print(self.complete_prompt)
        # print(self.missing_img_prompt)
        # print(self.missing_text_prompt)

        for param in self.transformer.parameters():
            param.requires_grad=False
        for param in self.text_embeddings.parameters():
            param.requires_grad=False
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        vilt_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
        self.records = {}

    def pad_vector(self,original_vector, target_dim_1):
        # 获取原始向量的维度
        original_dim = list(original_vector.size())

        # 创建目标维度
        target_dim = list(original_dim)
        if target_dim[1] > target_dim_1:
            return original_vector[:, :target_dim_1, :]

        target_dim[1] = target_dim_1

        # 创建一个零张量作为目标向量
        target_tensor = torch.zeros(target_dim)

        # 计算原始向量的平均值
        mean_value = torch.mean(original_vector)

        # 将原始向量的值复制到目标向量的中间位置
        start_index = (target_dim[1] - original_dim[1]) // 2
        end_index = start_index + original_dim[1]
        target_tensor[:, start_index:end_index, :] = original_vector

        # 使用平均值填充目标向量的剩余空白位置
        target_tensor[:, :start_index, :] = mean_value
        target_tensor[:, end_index:, :] = mean_value

        return target_tensor

    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        is_train=None,
    ):
        text_ids = batch[f"text_ids"]
        text_labels = batch[f"text_labels"]
        text_masks = batch[f"text_masks"]
        
        gen_text_ids = batch[f"gen_text_ids"]
        gen_text_masks = batch[f"gen_text_masks"]
        gen_text_embeds = self.text_embeddings(gen_text_ids)
        
        jiansuo_text_ids = batch[f"jiansuo_text_ids"]
        jiansuo_text_masks = batch[f"jiansuo_text_masks"]
        jiansuo_text_embeds = self.text_embeddings(jiansuo_text_ids)
        

        img = batch["image"][0]
        gen_img = batch["gen_image"][0]
        jiansuo_img = batch["jiansuo_image"][0]
        ( gen_image_embeds, gen_image_masks, gen_patch_index, gen_image_labels,) = self.transformer.visual_embed( gen_img, max_image_len=self.hparams.config["max_image_len"], mask_it=mask_image,)
        ( jiansuo_image_embeds, jiansuo_image_masks, jiansuo_patch_index, jiansuo_image_labels,) = self.transformer.visual_embed( jiansuo_img, max_image_len=self.hparams.config["max_image_len"], mask_it=mask_image,)

        gen_text_embeds_ = self.pad_vector(gen_text_embeds, 1024).to(self.device)
        gen_image_embeds_ = self.pad_vector(gen_image_embeds, 217).to(self.device)
        jiansuo_text_embeds_ = self.pad_vector(jiansuo_text_embeds, 1024).to(self.device)
        jiansuo_image_embeds_ = self.pad_vector(jiansuo_image_embeds, 217).to(self.device)
        
        gen_text_embeds_ = self.text_pool(gen_text_embeds_).view(gen_text_embeds.shape[0], -1)
        gen_image_embeds_ = self.image_pool(gen_image_embeds_).view(gen_image_embeds.shape[0], -1)
        jiansuo_text_embeds_ = self.text_pool(jiansuo_text_embeds_).view(jiansuo_text_embeds.shape[0], -1)
        jiansuo_image_embeds_ = self.image_pool(jiansuo_image_embeds_).view(jiansuo_image_embeds.shape[0], -1)

        text_gate = self.text_gate(gen_text_embeds_, jiansuo_text_embeds_)
        image_gate = self.image_gate(gen_image_embeds_, jiansuo_image_embeds_)
        
        # instance wise adding aware prompts
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                prompt = self.no_adding_prompt        
            elif batch["missing_type"][idx] == 1:
                if text_gate[idx] > 0.5:
                    self.gt += 1
                    # prompt = self.adding_gen_text_prompt
                    prompt = torch.cat([self.missing_text_prompt,self.adding_gen_text_prompt],dim = 1)
                    text_ids[idx] = gen_text_ids[idx]
                    text_masks[idx] = gen_text_masks[idx]
                else:
                    self.jt += 1
                    # prompt = self.adding_jiansuo_text_prompt
                    prompt = torch.cat([self.missing_text_prompt[:,:8,:],self.adding_jiansuo_text_prompt[:,:8,:]],dim = 1)
                    text_ids[idx] = jiansuo_text_ids[idx]
                    text_masks[idx] = jiansuo_text_masks[idx]
            elif batch["missing_type"][idx] == 2:
                if image_gate[idx] > 0.5:
                    self.gi += 1
                    # prompt = self.adding_gen_image_prompt
                    prompt = torch.cat([self.missing_img_prompt[:,:8,:],self.adding_gen_image_prompt[:,:8,:]],dim = 1)
                    img[idx] = gen_img[idx]
                else:
                    self.ji += 1
                    # prompt = self.adding_jiansuo_image_prompt
                    prompt = torch.cat([self.missing_img_prompt[:,:8,:],self.adding_jiansuo_image_prompt[:,:8,:]],dim = 1)
                    img[idx] = jiansuo_img[idx]
            
                
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            
            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)
        # import sys
        # sys.exit()
        text_embeds = self.text_embeddings(text_ids)
        ( image_embeds, image_masks, patch_index, image_labels,) = self.transformer.visual_embed( img, max_image_len=self.hparams.config["max_image_len"], mask_it=mask_image,)

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds+ self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx)),
        )
        
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()   
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        x = co_embeds.detach()

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=prompts[:,self.prompt_layers.index(i)], 
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)
            
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))
            
        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        vilt_utils.epoch_wrapup(self)
#         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
#         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
#         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        vilt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        print("jiansuo_image:",self.ji)
        print("gen_image:",self.gi)
        print("jiansuo_text:",self.jt)
        print("gen_text:",self.gt)
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        vilt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return vilt_utils.set_schedule(self)
