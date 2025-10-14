# src/llama_loader.py
import torch
import logging

logger = logging.getLogger(__name__)

def load_llama_weights_into_custom_model(model, llama_state_dict, n_heads, d):
    """
    Load pretrained LLaMA weights into our custom QLoRA+MQA+MoE architecture.
    - model: our custom nn.Module
    - llama_state_dict: state_dict() from HF LLaMA model
    - n_heads: number of attention heads
    - d: per-head dimension
    """

    # 1. Token embedding
    model.token_emb.weight.data.copy_(llama_state_dict["model.embed_tokens.weight"])
    logger.info("Loaded token embeddings ✅")

    # 2. Transformer blocks
    for i, block in enumerate(model.blocks):
        prefix = f"model.layers.{i}.self_attn"

        # ---- Q projection ----
        q_w = llama_state_dict[f"{prefix}.q_proj.weight"]  # [n_embed, n_embed]
        block.attn.q_proj.base.weight.data.copy_(q_w)
        
        # ---- K projection (convert multi-head → MQA) ----
        k_w = llama_state_dict[f"{prefix}.k_proj.weight"]  # [n_embed, n_embed]
        k_w = k_w.view(k_w.shape[0], n_heads, d)           # [n_embed, h, d]
        k_w_mqa = k_w.mean(dim=1)                          # [n_embed, d]
        block.attn.k_proj.base.weight.data.copy_(k_w_mqa)

        # ---- V projection ----
        v_w = llama_state_dict[f"{prefix}.v_proj.weight"]  # [n_embed, n_embed]
        v_w = v_w.view(v_w.shape[0], n_heads, d)
        v_w_mqa = v_w.mean(dim=1)
        block.attn.v_proj.base.weight.data.copy_(v_w_mqa)

        # ---- Output projection ----
        o_w = llama_state_dict[f"{prefix}.o_proj.weight"]
        block.attn.out_proj.base.weight.data.copy_(o_w)

        logger.info(f"Loaded block {i} attention projections ✅")

        # ---- Layer norms ----
        block.ln1.weight.data.copy_(llama_state_dict[f"model.layers.{i}.input_layernorm.weight"])
        block.ln2.weight.data.copy_(llama_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"])

    # 3. Final layer norm
    model.ln_f.weight.data.copy_(llama_state_dict["model.norm.weight"])

    # 4. LM head (tie with embeddings)
    model.lm_head.weight.data.copy_(llama_state_dict["lm_head.weight"])

    logger.info("✅ Finished loading LLaMA weights into custom model.")
    return model
