# LoRA Positioning in the Architecture — Zero to Hero

**Scope**: this doc answers exactly one question, in full depth: **where does
LoRA actually sit inside the model, and how does the segmentation/depth
conditioning signal reach it?** It applies identically to any segmentation
source that plugs into this pipeline's encoder-slot contract — SegFormer here
(see [SEGMENTATION.md](SEGMENTATION.md)), or Grounded-SAM on its own branch of
this repo — both go through the exact same `src/lora.py` / `src/model.py`
code, only the encoder that produces the conditioning map differs.

Every claim below is a direct citation to this repo's own code, re-read and
verified line-by-line while writing this doc (not copy-pasted from a
research pass) — file:line references are given throughout so you can jump
straight to the source and see for yourself.

---

## 0 · Two sentences, if that's all you need

A normal frozen conv layer computes `W(x)`. This repo's structure LoRA
replaces that layer with one that computes `W(x) + lora_scale * B(A(x)
modulated by the segmentation map)` — the original frozen computation is
untouched, and a small trainable side-branch adds a segmentation-aware
correction on top. That side-branch physically **replaces** the original
`nn.Conv2d` inside the live UNet object (`setattr`, `src/model.py:313-317`)
— it isn't a wrapper class, a hook, or a patched `forward()`.

---

## 1 · Vanilla LoRA vs. this repo's conditional LoRA

**Vanilla LoRA** (the technique as usually described, e.g. for fine-tuning a
LLM or a text-to-image model on a style): freeze the original weight matrix
`W`, add a trainable low-rank pair `A` (down-projection) and `B`
(up-projection, zero-initialized so training starts as a no-op), and compute

```
y = W·x + (B·A)·x
```

`B·A` is a fixed, static correction learned during training — once trained,
it always does the same thing to `x`, regardless of any external signal.
Vanilla LoRA is almost always applied to **attention** projections
(`q`/`k`/`v`/`out`).

**This repo's conditional LoRA** (`NewStructLoRAConv`, `src/lora.py:143-197`)
adds one more ingredient: an **external conditioning signal** `c` (here, the
segmentation colour map, already resolution-matched to this exact layer)
that **modulates** the low-rank branch *every forward call*, via a
[FiLM](https://arxiv.org/abs/1709.07871) (Feature-wise Linear Modulation)
step:

```
w_out = W(x)                                  # frozen, unchanged
a_out = A(x)                                   # low-rank down-projection of the SAME input
element_scale = gamma(c) + 1.0                 # per-pixel scale, from the seg map
element_shift = beta(c)                        # per-pixel shift, from the seg map
a_cond = a_out * element_scale + element_shift # <-- FiLM: seg map modulates the bottleneck
b_out  = B(a_cond)                             # low-rank up-projection
return w_out + b_out * lora_scale              # additive residual on top of the frozen output
```

This is a strictly different, strictly more expressive mechanism than
vanilla LoRA: the correction added to `x` isn't fixed at training time — it
changes *per input image*, because it's driven by that image's own
segmentation map. That's what makes this "structure-conditioned" LoRA rather
than a plain style fine-tune.

Source: `src/lora.py:177-197` (`NewStructLoRAConv.forward`, quoted above
nearly verbatim — the only omission is the `lora_scale == 0.0` early-out at
`:180-181`, an optimization that skips the whole branch and returns the
frozen output directly when the LoRA is fully disabled).

---

## 2 · The three LoRA classes, and which config selects which

`src/lora.py` defines three classes. All three share the frozen-`W` +
trainable-`A`/`B` + FiLM-`gamma`/`beta` skeleton from §1, but differ in
**what shape `c` is** and **what layer type they wrap**:

| Class | File:line | Wraps | `c` shape | `gamma`/`beta` are | Selected by |
|---|---|---|---|---|---|
| `SimpleLoraLinear` | `src/lora.py:9-90` | attention `nn.Linear` (`to_k`/`to_v`) | a single vector per image (or per token, if `n_transformations>1`) | `nn.Linear` | `configs/lora/style.yaml`, `configs/lora/struct_attn.yaml` |
| `LoRAConv` | `src/lora.py:94-140` | `nn.Conv2d` | a single vector per image | `nn.Linear`, broadcast over `H,W` via `[..., None, None]` (`:136`) — **one scale/shift per channel, same everywhere spatially** | not used by any config in this repo currently (available, unused) |
| `NewStructLoRAConv` | `src/lora.py:143-197` | `nn.Conv2d` | a **spatial feature map** `[B, c_dim, H, W]`, same resolution as this layer | `nn.Conv2d(c_dim, rank, kernel_size=1)` — **a scale/shift value at every pixel, independently** | `configs/lora/struct.yaml` — **the one used for segmentation and depth structure conditioning** |

The critical difference between `LoRAConv` and `NewStructLoRAConv` is
**spatial resolution of the modulation**: `LoRAConv`'s FiLM is one number
per channel for the whole image (like whispering "make this warmer overall"
into every pixel equally); `NewStructLoRAConv`'s FiLM is a full feature map,
computed by a 1×1 convolution over the segmentation map at the *exact* pixel
grid of this layer (like being able to say something different at every
single pixel — but only by looking at that one pixel, see §7).

`configs/lora/struct.yaml:8-12` — this is the config actually used by both
pipelines:
```yaml
config:
  c_dim: 128
  rank: 128
  adaption_mode: only_res_conv
  lora_cls: NewStructLoRAConv
```
(`configs/lora/struct_attn.yaml` and `configs/lora/style.yaml` exist as
alternate, unused-by-seg/grounded_sam structure/style conditioning schemes —
listed for completeness in §7, not part of the seg/grounded_sam pipelines.)

---

## 3 · Which UNet layers actually get LoRA-wrapped

`ModelBase.add_lora_to_unet` (`src/model.py:138-328`) is the function that
decides this. It iterates **every weight tensor's dotted path** in the real
SD1.5 `UNet2DConditionModel`'s state-dict (`sd = unet.state_dict()`,
`src/model.py:151`) and, for each path, checks it against the currently
selected `adaption_mode` (`config.adaption_mode`, from the YAML in §2):

```python
# src/model.py:175-199 (abbreviated to the two relevant modes)
_continue = True
if adaption_mode == "full_attention" and "attn" in path:            _continue = False
if adaption_mode == "only_self" and "attn1" in path:                 _continue = False
if adaption_mode == "only_cross" and "attn2" in path:                 _continue = False
if adaption_mode == "only_conv" and ("conv1" in path or "conv2" in path): _continue = False
if adaption_mode == "only_first_conv" and "0.conv1" in path:         _continue = False
if adaption_mode == "only_res_conv" and ("0.conv1" in path or "1.conv1" in path): _continue = False
if adaption_mode == "full" and ("attn" in path or "conv" in path):   _continue = False
if adaption_mode == "no_cross" and "attn2" not in path:               _continue = False
# ... (b-lora / sdxl-only modes, not relevant to SD1.5 struct/seg — src/model.py:201-238)
if _continue:
    continue   # this path is NOT adapted, left as the original frozen layer
```

**`struct.yaml` sets `adaption_mode: only_res_conv`**, which matches any
path containing the substring `"0.conv1"` or `"1.conv1"` — i.e. **only the
first convolution (`conv1`) of the first two `ResnetBlock2D` instances in
every down/mid/up resolution stage.** Concretely, this touches paths like
`down_blocks.0.resnets.0.conv1.weight`, `down_blocks.0.resnets.1.conv1.weight`,
`mid_block.resnets.0.conv1.weight`, `up_blocks.3.resnets.1.conv1.weight`,
and so on for every stage. It does **not** touch:
- `conv2` of any resnet (only `conv1` matches the filter),
- any attention layer (`to_k`/`to_v`/`to_q`/`to_out`) — `only_res_conv`
  never matches `"attn"` at all,
- any resnet beyond index 1 in a stage that has more than two.

Also worth knowing: even in modes that *do* target attention (`full_attention`,
`only_self`, etc.), only `to_k` and `to_v` are ever actually wrapped —
`ATTENTION_MODULES = ["to_k", "to_v"]` (`src/model.py:25`) gates it further at
`:264-266`: `to_q` and `to_out.0` are **never** LoRA-adapted, in any mode,
anywhere in this codebase.

Bias tensors are explicitly skipped (`if "bias" in path: continue`,
`src/model.py:243-246`) — a conv's bias is loaded into the LoRA wrapper's
frozen `W` together with the weight (`:299`), not treated as its own
adaptable path.

---

## 4 · How a LoRA module actually gets "inserted" — the `setattr` swap

This is the part that's easy to picture wrong. **There is no monkeypatch, no
forward hook, no wrapper `forward()` that calls the original layer.** For
each matched path, `add_lora_to_unet` does this (`src/model.py:248-317`,
abbreviated):

```python
parent_path = ".".join(path.split(".")[:-2])   # e.g. "down_blocks.0.resnets.0"
target_path = ".".join(path.split(".")[:-1])   # e.g. "down_blocks.0.resnets.0.conv1"
target_name = path.split(".")[-2]              # e.g. "conv1"
parent_module = getattr_recursive(unet, parent_path)
target_module = getattr_recursive(unet, target_path)   # the ORIGINAL nn.Conv2d

lora = NewStructLoRAConv(
    in_channels=target_module.in_channels, out_channels=target_module.out_channels,
    kernel_size=target_module.kernel_size, stride=target_module.stride,
    padding=target_module.padding, data_provider=data_provider, depth=depth,
    **class_config,   # c_dim, rank, lora_scale, ...
)
lora.W.load_state_dict({"weight": w, "bias": b})   # copy the ORIGINAL pretrained weights into lora.W

setattr(parent_module, target_name, lora)   # <-- THE ACTUAL INSERTION
```

`setattr(parent_module, "conv1", lora)` **physically replaces** the
`ResnetBlock2D`'s `self.conv1` attribute — it used to point at a plain
`nn.Conv2d`, now it points at a `NewStructLoRAConv` instance. Diffusers'
own, completely unmodified `ResnetBlock2D.forward()` code still just does
`hidden_states = self.conv1(hidden_states)` — it has no idea anything
changed. Because `self.conv1` is a different object now, that same line of
stock diffusers code transparently runs the LoRA computation from §1 instead
of a plain convolution. This is why the LoRA mechanism needs **zero changes**
to the diffusers UNet source: it exploits the fact that PyTorch attribute
lookup (`self.conv1`) is dynamic.

The full list of instantiated LoRA modules for a given named LoRA task (e.g.
`"struct"`) is collected as they're created: `self.lora_layers[name] = [lora]
+ self.lora_layers.get(name, [])` (`src/model.py:328`). This registry is
reused later by `make_lora_scale_callback` (§6) to mutate every layer's
`.lora_scale` in lockstep at inference time.

---

## 5 · Multi-resolution fan-out: one encoder pass feeds the whole UNet

SD1.5's UNet has 4 resolution "depths" (down_blocks 0-2 + mid_block, channel
widths 320/320/640/1280/1280 → `self.max_depth = len(block_out_channels) - 1
= 3`, `src/model.py:119`). A segmentation map at, say, 512×512 needs to reach
LoRA layers operating on feature maps at 4 *different* spatial resolutions
(e.g. 64×64 down to 8×8 latent-space grids). This is solved once per
resolution, not once per layer:

**Depth assignment** (`src/model.py:254-261`) — computed once, when each
LoRA layer is created:
```python
if "mid_block" in path:      depth = self.max_depth                              # deepest / lowest-res
elif "down_blocks" in path:  depth = int(path.split("down_blocks.")[1][0])        # 0 (shallow) .. max_depth-1
elif "up_blocks" in path:    depth = self.max_depth - int(path.split("up_blocks.")[1][0])  # mirrors back up
```
So `down_blocks.0` → depth 0 (highest resolution), deeper down-blocks → higher
depth, `mid_block` → depth 3 (bottleneck), and `up_blocks` count depth back
*down* toward 0 as the U-Net's decoder upsamples — meaning **a down-path
layer and an up-path layer at the same spatial resolution get the same
depth index**, and therefore the same conditioning map.

**Mapper produces exactly 4 maps, one per depth** (`FixedStructureMapper15`,
`src/mapper_network.py:18-82`): a shared conv trunk (`self.down`, 3 stride-2
halvings) then 3 more stride-2 stages (`block1`/`block2`/`block3`), with a
1×1-conv head after each stage (`out0..out3`, `:64-67`) producing a 4-tuple
`(out0, out1, out2, out3)` of `[B, 128, H_i, W_i]` feature maps at
progressively coarser resolutions (`:69-82`).

**Delivery** (`src/model.py:786-815`, the per-LoRA loop inside `sample_easy`
— see §6 for the full trace): the mapper's 4-tuple output is handed to a
shared `DataProvider` once per forward pass (`dp.set_batch(mapped_cond)`,
`:815`). Every `NewStructLoRAConv` instance for this LoRA task reads from
that *same* `DataProvider` and picks its own slice: `cs =
self.data_provider.get_batch(); c = cs[self.depth]` (`src/lora.py:183-184`).

**Net effect**: one encoder call + one mapper call per generation step
produces the conditioning for the *entire* UNet (every wrapped `conv1`,
across all down/mid/up stages) in one shot — the fan-out to the right
resolution at the right layer happens purely through each layer's
pre-computed `depth` index, not through per-layer recomputation.

---

## 6 · Full end-to-end trace (inference path, `sample_easy`)

Using `ModelBase.sample_easy` (`src/model.py:714-850`) as the canonical
trace — the training path (`forward`, `:473-569`) is structurally identical,
just called with `skip_encode=True` and the real training loss instead of a
`pipe(...)` call.

```
 segmentation colour map `c`  (either a live encoder's output, or a
     [B,3,H,W] in [0,1]        pre-saved map passed straight through when
        |                      skip_encode=True — see SEGMENTATION.md §10.4b)
        v
 ┌──────────────────────┐
 │ cond = c if skip_encode  else encoder(c)         [model.py:798]
 └──────────────────────┘
        |
        v  (optional, off by default)
 ┌──────────────────────┐
 │ box-blur softening: F.avg_pool2d(cond, k)         [model.py:805-807]
 └──────────────────────┘
        |
        v
 ┌──────────────────────┐
 │ mapped_cond = mapper(cond)                        [model.py:809]
 │   -> FixedStructureMapper15: (out0, out1, out2, out3)   [mapper_network.py:69-82]
 └──────────────────────┘
        |
        v
 ┌──────────────────────┐
 │ dp.set_batch(mapped_cond)                         [model.py:815]
 │   -> shared DataProvider, one per LoRA task ("struct")
 └──────────────────────┘
        |
        v
 ┌──────────────────────────────────────────────────────────────────┐
 │ self.pipe(prompt=..., **kwargs)   <-- STOCK, UNMODIFIED diffusers │
 │   internally: UNet2DConditionModel.forward()                     │
 │     -> ResnetBlock2D.forward()  (unchanged diffusers code)       │
 │          hidden_states = self.conv1(hidden_states)               │
 │                            ^^^^^^^^^^                             │
 │                 this IS a NewStructLoRAConv instance now          │
 │                 (setattr-swapped in, see §4)                      │
 └──────────────────────────────────────────────────────────────────┘
        |
        v  (happens independently, once per wrapped conv1, all over the UNet)
 ┌──────────────────────────────────────────────────────────────────┐
 │ NewStructLoRAConv.forward(x)                     [lora.py:177-197]│
 │   w_out = self.W(x)                     <- frozen conv, unchanged │
 │   cs = self.data_provider.get_batch()   <- same 4-tuple as above  │
 │   c = cs[self.depth]                    <- THIS layer's res. map  │
 │   element_scale = self.gamma(c) + 1.0   <- 1x1 conv, per-pixel    │
 │   element_shift = self.beta(c)          <- 1x1 conv, per-pixel    │
 │   a_out  = self.A(x)                                              │
 │   a_cond = a_out * element_scale + element_shift                  │
 │   b_out  = self.B(a_cond)                                         │
 │   return w_out + b_out * self.lora_scale   <- ADDITIVE, on top    │
 └──────────────────────────────────────────────────────────────────┘
```

**Injection semantics, stated precisely**: the segmentation signal never
touches attention (`only_res_conv` excludes it entirely), never touches
`conv2`, and is not blended into `encoder_hidden_states` (the text prompt's
embedding) or into the timestep embedding. It modulates a parallel low-rank
residual branch that is **added onto the output** of specific frozen resnet
`conv1` layers. The modulation is spatially *precise* (a different value at
every pixel, since `beta`/`gamma` are convolutions applied to a
spatially-registered feature map) but informationally *shallow* at any given
pixel — see §8.

---

## 7 · Three coexisting injection mechanisms — don't conflate them

This codebase has **three different ways** of getting an external signal
into the frozen UNet's computation. They coexist and are configured
independently; the segmentation/depth pipelines use only the first:

| Mechanism | Used by | Injection point | How |
|---|---|---|---|
| **Conditional LoRA (FiLM residual)** | `struct.yaml` (`NewStructLoRAConv`) — segmentation, depth | resnet `conv1` output, per wrapped layer | Additive: `w_out + b_out * lora_scale`, §1/§6 above |
| **Cross-attention style LoRA** | `style.yaml` (`SimpleLoraLinear`, `adaption_mode: only_cross`) | attention `to_k`/`to_v` inside `attn2` (cross-attention to text) | Same FiLM-residual math as §1, but on a `nn.Linear`, modulated by a *single pooled vector* (CLIP image embedding via `SimpleMapper`), not a spatial map |
| **ControlNet** (optional, `use_controlnet=True`) | separately, when enabled (`src/model.py:88-103`) | UNet's down-block and mid-block **residual connections**, via a full second `ControlNetModel` (`lllyasviel/sd-controlnet-depth`) | Diffusers' own stock ControlNet mechanism — an entirely separate pretrained network, not a LoRA at all; produces residuals that diffusers adds into the UNet's skip connections. Orthogonal to (can be combined with) the LoRA mechanism |

The segmentation pipeline documented in `SEGMENTATION.md` (and Grounded-SAM,
on its own branch) uses **only** the first row. `use_controlnet` and
`style.yaml` are separate, currently-unused-by-segmentation capabilities that
exist in this shared codebase — mentioned here only so their code isn't
mistaken for part of the segmentation conditioning path.

---

## 8 · Why this design produces "fits the shape but doesn't know how it looks"

This is the direct architectural consequence of everything above, and the
root cause already diagnosed and documented in
[SEGMENTATION.md §10.4a](SEGMENTATION.md#104a-fixing-fits-the-shape-but-doesnt-know-how-it-looks-added-2026-07-17):

`self.beta` and `self.gamma` (`src/lora.py:174-175`) are **1×1 convolutions**
— a 1×1 conv's receptive field is exactly one pixel. So inside any flat,
constant-class region of the segmentation map (e.g. the interior of a car,
which is uniformly "Car" colour everywhere), `beta(c)` and `gamma(c)` produce
the *same* scale/shift value at every pixel in that region — the FiLM
modulation carries **zero information about appearance**, only "this class
is here." Depth maps don't have this problem (a depth map varies
continuously even inside one object, so its FiLM modulation still carries
shape/appearance information pixel-to-pixel); a categorical segmentation map
does. And because `lora_scale` applies at **full strength through every
single denoising step** by default — including the late steps, where a
diffusion model is normally deciding fine texture/appearance, not layout —
the model is given no room to fall back on its own trained prior for what
the object should look like inside a flat region.

The two inference-time mitigations built for this (both documented in
`SEGMENTATION.md §10.4a` and available via `sample_easy`'s
`conditioning_kernel_size` / `lora_scale_start`/`lora_scale_end`/
`lora_scale_decay_start_frac` parameters, `src/model.py:346-410` +
`:805-807` + `:817-843`) work directly on the mechanism traced in this doc:
- **`lora_scale` decay** (`make_lora_scale_callback`) mutates
  `.lora_scale` on every layer in `self.lora_layers[name]` (§4's registry)
  between denoising steps — full strength while layout is being decided,
  fading later so appearance can lean on SD's own prior.
- **`conditioning_kernel_size`** box-blurs `cond` (the colour map, §6)
  *before* the mapper, softening hard silhouette edges — it cannot add
  information to a flat interior (blurring a constant region is still that
  same constant), it only softens the boundary.

Neither mitigation requires retraining — both operate purely on the
mechanism this doc describes: `lora_scale` because it's read fresh every
`forward()` call (§1), the kernel because it operates on `cond` before it
ever reaches the frozen-weight-preserving LoRA math.

---

## Quick reference

| Question | Answer | Where |
|---|---|---|
| Which LoRA class does structure conditioning use? | `NewStructLoRAConv` | `src/lora.py:143-197` |
| Which UNet layers get wrapped? | `conv1` of resnets 0 and 1, every down/mid/up stage (`only_res_conv`) | `src/model.py:192-193` |
| How does the LoRA module get "attached"? | `setattr` replaces the original submodule in the live UNet | `src/model.py:313-317` |
| How does a segmentation map become 4 differently-sized conditioning maps? | `FixedStructureMapper15`'s 4 resolution heads (`out0..out3`) | `src/mapper_network.py:18-82` |
| How does each layer know which of the 4 maps is "its" resolution? | pre-computed `depth` index, `cs[self.depth]` | `src/model.py:254-261`, `src/lora.py:183-184` |
| Where does the conditioning actually get added to the UNet's math? | additively, onto the frozen conv1's own output | `src/lora.py:197` |
| Why can't it "see" inside flat regions? | `beta`/`gamma` are 1×1 convs — one-pixel receptive field | §8 above, `src/lora.py:174-175` |
