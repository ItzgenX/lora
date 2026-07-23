# LoRAdapter → Segmentation · Speaker Guide

**How to use this:** each slide has **Understand it** — a proper, from-scratch explanation so you could teach it yourself — and **Say it** — the actual words to speak. Study the *Understand it* sections until the ideas feel obvious; then the *Say it* scripts will come out naturally instead of memorised. Timings are in brackets; total ≈ 30 minutes.

**The spine of the whole talk (memorise this):** *A giant image generator stays completely frozen. Beside it we bolt a tiny trainable "adapter." Normally that adapter does one fixed thing — but we make its behaviour depend, live, on a segmentation map. So the same frozen model can be steered to any layout we hand it, without ever retraining it.*

**Delivery mindset:** slides 1–14 are you explaining a known method clearly. Slides 15–25 are *your* work — that's where you slow down, make eye contact, and speak with ownership.

---

## Slide 1 — Title *(~45s)*

**Understand it:**
Start by knowing what LoRAdapter *is* and what *you* added. LoRAdapter (paper name: CTRLorALTer, ECCV 2024) is a technique for **controlling** a text-to-image model — steering *where* things go (structure) and *how* they look (style) — cheaply, without retraining the big model. Out of the box it supports **depth** maps (for structure) and **style** references. It does *not* support segmentation. Your project adds exactly that: **segmentation-map conditioning**, with two different ways of producing those maps (SegFormer and Grounded-SAM), and — importantly — without modifying the model's core code.
The equation `y = W(x) + λ·B(FiLM(A(x), seg map))` is the entire method compressed into one line. `W(x)` is the frozen original layer; the rest is the small trainable add-on, and `seg map` is your new driving signal. You don't need to explain it yet — just promise you'll unpack it. Say the word "frozen" with weight; it's the whole efficiency story.

**Say it:**
> "Text-to-image models are incredible, but text alone can't tell them exactly *where* things should go. This paper — LoRAdapter — solves that with a tiny, cheap add-on that steers a frozen model using an extra signal. It already does depth and style. What *we* did is extend it to a new signal — segmentation maps — with two different ways of producing those maps, and without changing the model's core. That one equation up there is the entire idea; by the end of this talk every symbol in it will make sense."

---

## Slide 2 — Roadmap *(~30s)*

**Understand it:**
This just sets expectations and reduces anxiety for anyone who isn't a diffusion expert. Five parts: foundations (background), the core idea, the architecture, *your work*, and results. The single most important thing to plant here is the **colour code**, because it recurs on literally every diagram: **blue = frozen** (pretrained weights we never touch) and **amber = trainable** (the small part we add, and the conditioning signal that flows through it). If the audience internalises that colour code now, they can read every later diagram at a glance.

**Say it:**
> "Here's the plan. First, quick foundations — diffusion, the network, and LoRA — so we share a vocabulary. Then the core idea. Then the architecture. Then the part this talk is really about: *our* segmentation work. And finally, results and honest limitations. One thing to keep in your eye the whole time: **blue means frozen — untouched pretrained weights; amber means trainable — the part we add and the signal that drives it.**"

---

## Slide 3 — Section 01: Foundations *(~10s)*

**Understand it:**
A divider slide — its only job is to signal "we're starting background now." Setting the pace verbally ("three quick slides") tells experts they can relax and beginners that help is coming.

**Say it:**
> "Three quick foundation slides — if you already know diffusion, feel free to coast."

---

## Slide 4 — The problem *(~70s)*

**Understand it:**
The motivation. A text prompt controls *semantics* — the "what" (a car, a street) — but it has no way to specify *spatial arrangement* (which lane, what pose, foreground vs background) or a *precise visual style*. So people want two extra kinds of control that don't fit in words:
- **Structure conditioning** — "put the shapes *here*." The control signal is a **spatial map**: a depth map, an edge map, or — your case — a **segmentation mask** (a per-pixel class label: this pixel is road, that one is car). It governs *where things are and what form they take*.
- **Style conditioning** — "make it look like *this*." The control signal is a compact summary (an embedding) of a **reference image**. It governs *global appearance and texture*.
These two words — structure and style — are the backbone of the entire talk. Everything you present is one or the other, and *your* work is squarely on the structure side. Make sure the audience leaves this slide able to say which is which.

**Say it:**
> "If I prompt 'a car on a rainy street,' I get *a* car *somewhere* — but I can't say *this* exact layout, or make it look like *that* painting. Those are two different kinds of control. **Structure** — where the shapes go — which you drive with a spatial map like a depth map or a segmentation mask. And **style** — the overall look — which you drive with a reference image. LoRAdapter's promise is to do both with one mechanism, cheaply, on a frozen model. Our contribution lives on the structure side: we make *segmentation* a first-class way to control the layout."

---

## Slide 5 — How Stable Diffusion works *(~90s)*

**Understand it:**
This is the "what is a diffusion model" crash course, and you need two ideas from it.
*First, the generation pipeline.* Working directly on 512×512×3 pixels is expensive, so Stable Diffusion first uses a **VAE** (a pre-trained compressor) to shrink the image into a small **latent** — a 64×64×4 grid. All the heavy work happens on that small latent. Generation starts from **pure random noise** in latent space, and the **UNet** — the core neural network — looks at the noisy latent and predicts *what noise to remove*. Do that repeatedly (about 30–50 steps) and the noise gradually resolves into a coherent image. A final VAE step decompresses the finished latent back to pixels. The mental image: sculpting a statue out of a block of static, one careful pass at a time, guided by the text prompt.
*Second, the shape of the UNet.* The UNet is **multi-resolution**. It has an encoder that shrinks the latent through **4 depth stages**, a bottleneck, and a decoder that mirrors back up — that's the "U." Each stage is made of two kinds of layers: **convolutions** (ResNet blocks), which handle *local shape and texture*, and **attention** blocks, which is where the *text prompt* is injected (via cross-attention).
The two facts you must land: (1) the entire backbone — VAE, UNet, text encoder — stays **frozen** in this method, and that's where all the efficiency comes from; (2) convolutions decide *shape* while attention carries *text*, which is precisely why structure conditioning will target the **convolutions** and leave the text path untouched. Also plant the number **4** (resolution depths) and the fact that the conditioning signal will have to reach *all four* — that's a real problem the mapper solves later.

**Say it:**
> "Very quickly, how the generator works. The image is compressed into a small grid called a latent. Then a network called the UNet starts from noise and, over about 30 to 50 steps, repeatedly predicts what noise to remove — sculpting an image out of static. A final step decodes it back to pixels. Two facts matter for us. First, that whole backbone stays *frozen* — we never retrain it, and that's where the efficiency comes from. Second, the UNet works at four different resolutions, and it's built from convolutions, which decide local *shape*, and attention, which carries the *text*. So structure conditioning targets the convolutions and leaves text alone — and whatever signal we inject has to reach all four resolution levels. Hold that thought."

---

## Slide 6 — LoRA in one slide *(~75s)*

**Understand it:**
LoRA (Low-Rank Adaptation) is the cheap-adaptation trick everything is built on. A big pretrained network is full of enormous weight matrices `W`. Retraining them is slow and huge to store. LoRA's insight: **freeze `W` entirely**, and add a small parallel detour of two tiny matrices — `A` **down-projects** the input into a small "rank" (say 128 dimensions), and `B` **up-projects** it back to full size. You train *only* `A` and `B`. The layer's output becomes `W(x) + B(A(x))`. Because `A` and `B` are tiny compared to `W`, training is fast and the extra weights are a rounding error. One important detail: `B` is initialised to **zero**, so at the start the detour outputs nothing — training begins as a perfect no-op and can only improve from there, which makes it very stable.
The catch to emphasise: plain LoRA is **static**. Once trained, `B(A(x))` applies the *same* correction to every input forever. That's fine for "always paint in this style," but useless for "match *this specific* segmentation map, which is different every time." That limitation is the entire setup for the next slide — say the word "static" deliberately.

**Say it:**
> "The building block is LoRA. Imagine a giant pretrained weight matrix — call it W. Retraining it is expensive. LoRA freezes W completely and adds a tiny detour beside it: a matrix A that squeezes the input down to a small size, and B that expands it back. You train only A and B — a rounding error in size compared to W — and W's knowledge is never disturbed. B starts at zero, so at the beginning it does nothing and can only help. But here's the limitation: plain LoRA is *static*. Once trained, it applies the exact same correction to every single image. That's the thing the next idea fixes."

---

## Slide 7 — Section 02: The core idea *(~10s)*

**Understand it:**
Divider. Transition line — everything before was setup; now comes the actual invention.

**Say it:**
> "That limitation — 'static' — is the setup for the actual contribution."

---

## Slide 8 — FiLM: make the LoRA listen *(~80s)*

**Understand it:**
This is the mechanism that turns *static* LoRA into *conditional* LoRA. **FiLM** stands for Feature-wise Linear Modulation, and it's a simple, general idea: let an external signal *scale and shift* a set of features. Concretely, you take the conditioning signal `c` (for us, the segmentation map), and pass it through two small learned functions to produce a **scale, γ (gamma)** and a **shift, β (beta)**. Then you apply them to the middle of the LoRA branch: `γ · A(x) + β`. That's it — a multiply and an add, but the numbers come from your conditioning signal instead of being fixed.
Why this is powerful: the correction the LoRA makes is no longer baked in at training time — it's **recomputed from `c`, live, on every forward pass**. Feed a different map, get a different correction. That is exactly what makes the control **zero-shot**: at generation time you can hand the model a segmentation map it has never seen in training, and it will follow it, with no per-image finetuning. One reassuring detail if asked: γ is set up to start around ×1 (the code adds 1) and β around 0, so — like LoRA's zero-init B — FiLM starts as a gentle no-op and learns from there.

**Say it:**
> "The fix is a small, well-known trick called FiLM — feature-wise linear modulation. Take your conditioning signal — for us, the segmentation map — and from it compute two things: a scale, gamma, and a shift, beta. Then apply them right in the middle of the LoRA branch: multiply by gamma, add beta. Now the correction isn't fixed anymore — it's *computed from the map*, live, every step. That's the whole magic. Because it's computed on the fly, you can hand the model a segmentation map it has never seen and it will follow it — no retraining. That's what 'zero-shot control' means."

---

## Slide 9 — The conditional-LoRA block *(~90s)*

**Understand it:**
This diagram is the whole method in one picture — if the audience gets this, they get the paper. Trace the flow. The input `x` (the features flowing through the UNet at this layer) reaches a split point.
- The **blue (frozen) path** is the original convolution, `W(x)` — completely unchanged. This is why the base model's knowledge is fully preserved.
- The **amber (trainable) path** is the adapter: `A` shrinks the features to the small rank; **FiLM** then reshapes them using γ and β *derived from the segmentation map `c`* (shown feeding in at the bottom); `B` expands back to full size; and the result is scaled by **λ (lora_scale)**.
- Finally, the two paths are **added**: `y = W(x) + λ · B(FiLM(A(x), c))`. It's an *additive residual* — a correction laid on top of the frozen output, not a replacement of it.
Two things to point out explicitly because they're the crux: (1) the segmentation map **never touches `x`** — it only steers the tiny amber branch, so the base computation is untouched; (2) **λ is a live knob** read every step — set it to 0 and the whole block vanishes, leaving exactly the original model. This slide's amber wires are animated in the deck to show the signal "flowing," so let that do some work while you talk.

**Say it:**
> "This is the single most important picture in the talk — the whole method in one diagram. Follow the input, x. It splits two ways. The blue path is the original frozen convolution — completely untouched, so the base model is fully preserved. The amber path is our small trainable branch: A shrinks the signal, then FiLM — driven by the segmentation map at the bottom — reshapes it, then B expands it back, and lambda scales how strong it is. Finally, the two paths are simply *added*. Notice what the segmentation map does and doesn't touch: it never touches x directly — it only steers that tiny amber branch. And lambda is a live knob — set it to zero and the whole block vanishes, leaving the original model. If you take one image away from today, take this one."

---

## Slide 10 — Structure vs style *(~85s)*

**Understand it:**
This is the paper's headline claim — *one block, two jobs* — and understanding it makes everything downstream (including your extension) feel inevitable. The exact same conditional-LoRA block does structure **and** style; the *only* difference is the **shape of the conditioning signal `c`** and, correspondingly, whether γ/β are convolutions or linear layers:
- **Structure:** `c` is a **spatial map** (depth or segmentation), the same resolution as the layer. γ and β are **1×1 convolutions**, so you get a *different* scale/shift at *every pixel*. That per-pixel control is exactly what "where things go" requires.
- **Style:** `c` is a **single pooled vector** (a summary embedding of a reference image). γ and β are **linear layers**, so there's *one* scale/shift for the whole feature map — one global instruction, appropriate for "overall look."
Then the practical part — **tuning** — because you'll get asked "how do you control the strength?" The master dial is **`lora_scale`**, read live every denoising step: 0 = off, 1 = full. You can **decay** it across the steps — strong early to lock the layout in, then weaker late so the model's own prior fills in realistic texture. And for structure specifically there's a **softening kernel** that blurs hard silhouette edges. The big message: because it's all one formulation, adding a *new* structure signal — your segmentation — rides the exact same rails, which is why it stayed small and clean.

**Say it:**
> "Here's the elegant part — the same block handles structure *and* style. The only thing that changes is what you feed as the signal. For structure, the signal is a spatial map, and the modulators are 1×1 convolutions — so you get a different nudge at every pixel, which is exactly what 'where things go' needs. For style, the signal is a single summary vector of a reference image, and the modulators are simple linear layers — one nudge for the whole picture. Same equation, different signal shape. And you tune it with one knob, lora_scale: zero turns it off, one is full strength, and you can even fade it across the denoising steps — hold the layout firmly early, then relax so the model fills in realistic detail. That unification — one block, both jobs — is exactly why the whole thing stays so small, and it's why *we* could add a new structure signal without inventing new machinery."

---

## Slide 11 — Section 03: Inside the architecture *(~10s)*

**Understand it:**
Divider into the engineering. Frame it as "the part people usually skip — and the reason our extension was a clean drop-in."

**Say it:**
> "Now the engineering — the part people usually gloss over, and the reason our extension dropped in so cleanly."

---

## Slide 12 — Only conv1 gets adapted *(~75s)*

**Understand it:**
Two facts: which *class* is used, and which *layers* get touched.
The code has **three** flavours of the block. They share the same skeleton (frozen `W` + trainable `A`/`B` + FiLM γ/β) and differ only in signal shape: `NewStructLoRAConv` (spatial map, per-pixel — **used for structure, including your segmentation**), `SimpleLoraLinear` (pooled vector — used for style), and `LoRAConv` (present but unused). Don't dwell on the table; point at the *pattern* — one idea, three signal shapes.
The placement is controlled by one config switch — `adaption_mode: only_res_conv` — which wraps **only the first convolution (`conv1`)** of the first two ResNet blocks in each resolution stage. It deliberately does **not** touch the second convolution and does **not** touch attention. Why this matters: convolutions carry *local shape*, which is exactly what structure conditioning wants to steer, so wrapping just those gives maximum control per trainable parameter; and leaving attention frozen means the **text prompt still behaves normally**. This small, surgical footprint is *the* reason the method is parameter-cheap.

**Say it:**
> "In the code there are three flavours of this block — they all share the same skeleton and differ only in the signal shape. Structure uses the convolutional one. And there's a single switch that decides *which* layers get the treatment: it wraps only the first convolution of the ResNet blocks — not the second convolution, and crucially not the attention, which is where the text lives. So the footprint is deliberately tiny and surgical: convolutions carry local shape, which is what we want to steer, and leaving attention frozen means your prompt still behaves normally. That's the trick to getting strong control from very few trainable parameters."

---

## Slide 13 — The setattr swap *(~70s)*

**Understand it:**
This explains *how the block is physically inserted*, and it's genuinely elegant — worth a beat. It is **not** a monkeypatch, not a forward hook, not a forked copy of the diffusers library. For each target layer, the code builds a `NewStructLoRAConv`, **copies the original pretrained weights into its frozen `W`** (so nothing is lost), and then does exactly one thing: `setattr(parent, "conv1", lora)` — it reassigns the `conv1` attribute so it now points at our module. The magic is that diffusers' own, *completely unmodified* code still calls `self.conv1(x)` — but because that attribute now points at our object, that same untouched line transparently runs the LoRA math instead of a plain convolution. PyTorch's attribute lookup is dynamic, so one reassignment reroutes the whole computation. The payoff: **zero changes to the underlying library** — it works with the stock pipeline, survives version upgrades, and (say this) is exactly *why our segmentation extension never had to touch this machinery at all*.

**Say it:**
> "How does the block actually get *in* there? This is genuinely clever. It's not a monkeypatch or a hook. For each target layer, the code builds our module, copies the original pretrained weights into its frozen part, and then does exactly one thing — it reassigns the layer: 'this conv1 is now our module.' The diffusers library's own, completely unmodified code still just calls self-dot-conv1, and transparently runs our math instead. So there are *zero* changes to the underlying library — it survives version upgrades, and it's the reason our segmentation work never had to touch this machinery at all."

---

## Slide 14 — One map → the whole UNet *(~80s)*

**Understand it:**
This resolves the "reach all 4 depths" puzzle from Slide 5. The wrapped `conv1` layers operate at four *different* spatial resolutions, but the segmentation map is a single image. The solution runs *once per resolution*, not once per layer:
- A small **mapper** network takes the one conditioning map and, in a **single pass**, produces exactly **4 feature maps** — one at each UNet depth (e.g. 64², 32², 16², 8²).
- Each wrapped LoRA layer was stamped with its **depth index** when it was created, and simply grabs *its* matching map from a shared holder (the `DataProvider`).
- Because the UNet is symmetric, the layers going *down* and the layers coming back *up* at the same resolution have the same depth, so they **share** the same conditioning map — for free.
The efficiency point to make explicit: one encoder pass + one mapper pass conditions the **entire** UNet — every wrapped convolution, across all stages — per denoising step. There's no expensive per-layer recomputation. This is also the last piece of the *original* architecture; after this, everything is your extension.

**Say it:**
> "Remember the UNet has four resolution levels, and the conditioning has to reach all of them. That's what this mapper solves. It takes the one segmentation map and, in a single pass, produces four versions of it — one at each resolution. Every wrapped layer knows its own depth and just grabs the matching one. And because the network is symmetric, the layers going down and the layers coming back up at the same resolution share the same map. So a single pass through the mapper conditions the *entire* UNet, every step — not one expensive recomputation per layer. That's the efficiency story, visualised."

---

## Slide 15 — Section 04: Our work *(~12s)*

**Understand it:**
The pivotal divider. Everything before was shared vocabulary; this is *your* contribution. Deliver it with a slight change of energy — this is the part you own.

**Say it:**
> "Everything so far was the shared language. This next part is what *we* built — and it's more than just a port."

---

## Slide 16 — What we changed vs the original repo *(~90s)*

**Understand it:**
This is the single most important slide for "what did *you* do," so know it cold. In the **original repo**, the key training line is `cs = [imgs]` — the conditioning handed to the adapter is literally the *same raw photo* you're trying to reconstruct. Combined with `skip_encode=False`, this means a **live encoder (MiDaS)** runs *inside the training loop every single step*, computing a fresh depth map from that photo on the fly. So the stock dataset only ever needs `(image, caption)` — the depth is derived live, nothing is pre-saved.
Your extension makes **two deliberate departures**: (1) instead of computing a map live, you feed a **pre-saved segmentation map** and set **`skip_encode=True`**, so the live-encoder branch is never taken — the saved map is used as the conditioning directly; (2) a new dataset that reads `(image, seg_map, prompt)` triples. Everything *downstream* of that — the FiLM math, the UNet wrapping, the mapper fan-out — is **byte-identical, unmodified stock code** (verified: `src/lora.py`, `src/mapper_network.py`, `train.py` never change). The one-sentence version to land: **we changed how the conditioning map is *produced*, not how it's *used*.** That's why it was clean, and why a fair depth-vs-segmentation comparison is even possible.

**Say it:**
> "This is the slide that answers 'what did you actually do.' In the original repo, the key line hands the raw photo straight to a live depth model that computes a depth map on the fly, every single training step — so the dataset only needs images and captions. We made two deliberate changes. One: instead of computing a map live, we feed a *pre-saved* segmentation map and flip a flag, skip_encode, so that live encoder never runs. Two: a dataset that reads image, seg-map, and prompt together. And here's the punchline — look at the two rows. The entire back half — the mapper, the FiLM LoRA, the injection into the UNet — is *byte-identical, unmodified* original code. We changed how the conditioning map is *produced*, not how it's *used*. That discipline is exactly why the extension was clean, and why a depth-versus-segmentation comparison stays fair."

---

## Slide 17 — The approach: match the contract *(~80s)*

**Understand it:**
This explains *why* the extension slotted in without core changes. Every encoder in this system obeys a strict **interface contract**: it must accept an input tensor of shape `[B, 3, H, W]` with values in `[-1, 1]` (a batch of RGB images), and return `[B, 3, size, size]` with values in `[0, 1]` (a colour conditioning map). The original depth encoder (MiDaS) obeys this; the mapper downstream expects exactly this. You built your **segmentation encoder to satisfy the identical contract** — same input shape and range, same output shape and range — so it drops straight into the same slot and the mapper genuinely cannot tell it's segmentation rather than depth. That's the whole reason `model.py` and the mapper needed zero edits.
The second half is a **discipline** that protects the science: you kept **four strictly separated stages** — (A) compute depth, (B) train depth, (C) compute segmentation, (D) train segmentation — that never share code paths. This is what makes a later depth-vs-segmentation comparison *valid*: the pipelines are identical except for the conditioning signal, so any difference in output is attributable to the signal, not to some incidental code difference.

**Say it:**
> "*Why* was it so clean? Because the original depth encoder follows a strict contract: a specific input shape and range in, a specific output shape and range out. We built our segmentation encoder to satisfy that *exact same* contract — same shapes, same ranges — so it slots straight in and the rest of the pipeline literally can't tell it's segmentation instead of depth. We also kept a hard discipline: four separated stages — compute depth, train depth, compute segmentation, train segmentation — that never share code paths. That's what lets us later put depth and segmentation side by side and trust the comparison, because the only thing that differs is the signal."

---

## Slide 18 — Colour, not raw class-IDs *(~90s)*

**Understand it:**
This is the design decision that proves you *understood* the architecture rather than just wiring things together. A segmentation map is **categorical** — each pixel holds a discrete class *number* (road = 0, car = 14, and so on). The tempting shortcut is to feed those numbers straight in as a grayscale ramp (`id / N`). That's a trap, and here's the precise reason: the mapper is a convolutional network originally built for **depth**, which is a *continuous* signal where nearby values mean "nearby distances." Feeding class numbers as a ramp imposes a **false ordering** — it tells the network that class 5 ("pole") sits *between* class 4 ("fence") and class 6 ("traffic light"), as if they were on a spectrum. They're not; they're unrelated categories. The network would learn a lie.
Your fix: **colourise** each class into a **distinct, well-separated RGB colour**, so categories become a signal the conv can actually read — the segmentation analogue of depth's smooth gradient, and exactly what ControlNet's segmentation variant does. Two supporting correctness details worth mentioning: you always resize with **nearest-neighbour** (averaging class numbers would invent a class that doesn't exist at boundaries), and you use a **fixed palette** so the same class is always the same colour — which is what guarantees **train/inference parity**. A neat engineering touch: you save the raw IDs and colourise *at load time*, so changing the palette never requires re-running the segmenter.

**Say it:**
> "Here's a design decision that really matters. Segmentation gives you a class *number* per pixel — road is 0, car is 14, and so on. The tempting shortcut is to feed those numbers in as a grayscale ramp. But that's a trap: it invents a false ordering. 'Pole' isn't *between* 'fence' and 'traffic light' just because its number is — but a grayscale ramp says they're neighbours, and the mapper, which was built for depth's smooth gradient, would believe it. So instead we colourise every class into a distinct, well-separated colour — which turns categories into a signal the network can actually read, and it's exactly what ControlNet's segmentation model does too. Two supporting details: we only ever resize with nearest-neighbour, because averaging class numbers would invent classes that don't exist; and we keep a fixed palette, so the same class is always the same colour in training and inference. That decision is really the crux of making segmentation work on machinery built for depth."

---

## Slide 19 — Making it survive real data *(~90s)*

**Understand it:**
This is where most of the real engineering lived — the gap between the paper's clean setup and messy real simulator data. Four concrete items:
1. **16-bit format.** The real CARLA masks are saved as 16-bit PNGs (`I;16` mode), not the 8-bit you'd assume. If you convert them naively, different versions of the imaging library (Pillow) handle it differently and can silently corrupt the labels. Your fix: read the **raw pixel values** directly, which is version-proof.
2. **The alignment bug (fixed, commit 5db3624).** The masks are non-square (1280×800). The old code **stretched** the mask to a square while the paired RGB was **letterboxed** (padded to square). The result: they disagreed about *where* content was, by up to **96 pixels — about 19%** of the frame height at the top and bottom. That means the model was told "road here" while the actual road in the target was somewhere else — a silent, damaging mismatch between conditioning and supervision. The fix: letterbox the mask with the *same* geometry as the image.
3. **resize_mode toggle.** A single switch (letterbox vs centre-crop) that drives *both* the image and its mask through the *same* geometry, so they can never drift apart again.
4. **The parity rule.** The maps used at training and the maps used at inference must come from the same model, same preprocessing, same value range — or the two data distributions diverge and quality quietly collapses. This is the single most important rule for anyone reproducing the work.
The framing to land: none of this is in the paper. This is the unglamorous, load-bearing work that makes a research pipeline actually hold up.

**Say it:**
> "This is the unglamorous part, and honestly where most of the real work was — making it survive actual data. Four things. First, the real masks are 16-bit images, not the 8-bit you'd assume, so we read the raw pixels directly, which keeps the labels correct across any library version. Second — and this is a genuine bug we caught and fixed — the non-square masks were being *stretched* to square while the photo was *letterboxed*, which misaligned them by up to nineteen percent at the top and bottom. That means the model was being told 'the road is here' while the actual road was somewhere else — a silent, damaging mismatch. We fixed it so the mask uses the exact same geometry as its image. Third, a toggle for how we square things. And fourth, a strict parity rule: the maps at training and inference must come from the same model, same preprocessing, same value range — or the whole thing quietly degrades. None of this is in the paper; it's what it takes to make it real."

---

## Slide 20 — Two map sources *(~90s)*

**Understand it:**
Because both encoders satisfy the same contract (Slide 17), the *front end* is swappable — and you built two, with genuinely different trade-offs:
- **SegFormer.** A real neural segmentation network (SegFormer-b5, ~82M-param backbone) run **offline** on photos to produce 19-class Cityscapes maps. Crucially, its encoder has a **live path** — it can segment a brand-new image — which means it can even **re-segment the generated image** and score how faithfully the layout was reproduced. That controllability score is called **mIoU**.
- **Grounded-SAM.** Here the maps come straight from **CARLA's own segmentation camera** — 29 classes, and *no model has to run* to produce them; they exist at capture time. The trade-off: its encoder is a **training-only placeholder** (a "Tier-1 slot filler") with no live path, so the mIoU self-scoring is automatically **skipped** rather than faked.
The point to hammer is *not* the feature table — it's the **design**: one clean contract absorbs a real-world segmenter *and* a simulator's native labels, with no change to the generative core. The bottom row of the table says it all — the core is identical for both. And the honest one-liner on the trade-off: SegFormer is general and self-scoring but capped at 19 classes; Grounded-SAM is richer (29 classes) with free labels but no live inference.

**Say it:**
> "Because both obey the same contract, we can swap the entire front end — and we built two. SegFormer: a real segmentation network we run offline on photos. It's capped at nineteen classes, but it has a live path, which means it can even re-segment the *generated* image and score how faithfully the layout was followed. Grounded-SAM: here the maps come straight out of the CARLA simulator's segmentation camera — twenty-nine classes, and no model has to run to produce them, they're free at capture time. The trade-off is that its encoder is a training-only placeholder with no live path, so that self-scoring metric is simply skipped rather than faked. But look at the bottom row — the generative core is *identical* for both. The real point isn't the feature table; it's that one clean contract absorbs both a real-world segmenter and a simulator's native labels without touching the core."

---

## Slide 21 — The diagnostics harness *(~80s)*

**Understand it:**
This is a maturity signal — you didn't just train and hope, you built the *instruments* to measure quality objectively. Four tools:
- **Checkpoint monitoring grids.** Every time a checkpoint is saved, it renders a fixed set of validation scenes as **four panels**: the original photo, the segmentation map, the generation *with* the prompt, and the generation *without* the prompt. Using *fixed* scenes means checkpoint-3000 and checkpoint-6000 are directly comparable. The **empty-prompt panel** is the clever bit — it isolates *pure structure adherence*, so you can tell whether the model followed the *map* or whether the *text prompt* did the heavy lifting.
- **PSNR + SSIM** per checkpoint against the real image — numeric quality *trends* you can watch across a run.
- **mIoU controllability** (SegFormer only) — re-segment the output and measure "did it follow the structure I asked for?"
- **Dataset coverage scans** — before blaming the model for rendering a class badly, you can quantitatively check whether that class was even well-represented in the training data.
Plus TensorBoard curves and a per-run parameter snapshot. The message: generation quality is subjective, and we made it *measurable and reproducible*.

**Say it:**
> "Generation quality is subjective, so we built instruments to make it measurable. Every time a checkpoint is saved, it renders a set of fixed scenes as four panels: the original photo, the segmentation map, the generation *with* the prompt, and the generation *without* the prompt. That last panel is important — it isolates pure structure adherence, so you can tell whether the model followed the *map* or whether the *prompt* did the work. On top of that we log image-quality trends per checkpoint, a controllability score where the encoder supports it, and dataset coverage scans — so when a class renders badly, we can first ask 'was it even in the training data?' rather than blaming the model. In short, we made it reproducible and measurable, not a vibe check."

---

## Slide 22 — Section 05: Results & limitations *(~10s)*

**Understand it:**
Divider. The important mindset here is honesty — you're about to separate the paper's proven results from your own, still-in-progress results, and not overclaim.

**Say it:**
> "Last stretch — results, and I'll be honest about what's proven and what isn't."

---

## Slide 23 — Two claims, kept separate *(~85s)*

**Understand it:**
The key discipline on this slide is *not blurring two different things*. The left card is the **paper's** results, and they are for **depth and style**: fewest trainable parameters of any compared method, and state-of-the-art structure control on the standard metrics (cycle-consistency / MSE-d, FID, LPIPS). Those are *their* numbers for *their* signals — cite them as such.
The right card is **your segmentation** work, with honest status tags:
- **✓ works** — the core mechanism *transfers*: a real CARLA-conditioned sample follows the input layout (roads, buildings, vehicle placement all track the map). This is your solid, standable result *today*.
- **✓ fixed** — the alignment bug, confirmed by actually running the code.
- **⚠ open** — two known artifacts (the next two slides).
- **◷ pending** — full-scale quality *numbers* await the real training run on the cluster; the measurement harness is built and waiting, but you will *not* quote a metric you haven't earned at scale.
The honesty is the point, and it's actually a strength: the mechanism is proven to transfer, the pipeline is correct, the instruments are ready. If you have real generated images, this is the slide to point at the two image slots.

**Say it:**
> "I want to keep two things separate and be honest about both. On the left, the *paper's* results — for depth and style: it uses the fewest trainable parameters of any method compared, and it's state-of-the-art at structure control. Those are their numbers, for their signals. On the right is *our* segmentation work, and here's exactly where we stand. What's verified: the core mechanism *transfers* — a real CARLA-conditioned sample follows the input layout, roads and buildings and vehicles all land where the map says. We also found and fixed that alignment bug, confirmed by actually running it. Two artifacts are still open — I'll explain both next. And full-scale quality numbers are *pending* the real training run on the cluster. The measurement harness is built and waiting; I'm deliberately not going to quote a metric we haven't earned at scale. That honesty is the point."

---

## Slide 24 — Limitation: fits the shape, not the look *(~85s)*

**Understand it:**
The first artifact — and a great example of *architecture explaining behaviour*, diagnosed from the design rather than guessed. Recall from Slide 10 that for structure, γ and β are **1×1 convolutions** — a 1×1 conv has a receptive field of exactly **one pixel**. Now consider a large flat region of a single class — the whole interior of a car, which is one uniform colour in the segmentation map. At every pixel in that region, the 1×1 conv sees the *same* colour, so it produces the *same* scale/shift everywhere. The signal there carries only *"this class is here"* — **zero information about appearance**. That's why generations can nail the *silhouette* of an object but fill it with a generic, texture-less blob.
Why it's **specific to segmentation**: a depth map varies *continuously* even inside one object (near edge vs far edge), so its per-pixel signal still carries shape/appearance cues; a categorical segmentation map is flat inside a class, so it doesn't. Then the two **fixes, both without retraining**: (1) **decay `lora_scale`** in the late denoising steps, so once the layout is locked the model's own prior is free to invent realistic appearance; (2) a **softening kernel** on the hard silhouette edges. The takeaway line: understanding *where* the block sits told us *why* this happens and pointed straight at cheap fixes.

**Say it:**
> "Here's a limitation we diagnosed straight from the architecture — and it's specific to segmentation. Those modulators, gamma and beta, are one-by-one convolutions — they see exactly one pixel. So inside a big flat region of a single class — say the whole interior of a car, all one colour — every pixel gets the identical nudge. The signal literally only says 'this class is here' and carries *nothing* about what it should look like. Depth doesn't have this problem, because depth varies smoothly even inside an object. The nice thing is, understanding *where* the block sits tells us exactly *why* this happens — and both fixes are cheap and need no retraining: fade the conditioning strength in the late denoising steps so the model's own imagination fills in appearance, and soften the hard silhouette edges. Architecture explaining behaviour, and pointing straight at the fix."

---

## Slide 25 — Why the padding shows up *(~90s)*

**Understand it:**
The second artifact — and the direct answer to the natural question "if we padded it, shouldn't that band just be ignored?" When a frame isn't square, we **letterbox** it — add flat bands top and bottom to make it 512×512. Intuition says those bands are inert filler. They are not, and here's exactly why, in code:
1. **Padding is real pixels, not a mask.** The letterboxed 512×512 image *is* the training target — the code VAE-encodes the *whole* image, band included, into the latent the model is trained to reconstruct. So the model is literally supervised to produce that band.
2. **The seg map's band is a real conditioning colour.** The mask's pad region is filled with the "Unlabeled" class, which colourises to a specific colour and, through FiLM, *actively steers* the adapter in that region — it's not a null signal.
3. **Every training pair has the same band in the same place**, so the model very reliably *learns* the mapping "band here → flat pad colour" and dutifully reproduces it at generation time.
Why it isn't "static noise": we never add noise — we add a **definite flat colour** — and diffusion has **no per-pixel 'ignore this region' mask** in its loss; the pad is supervised exactly like real content. The real fix is to **not pad at all** — train at the native aspect ratio (e.g. 512×320) so there's no band to learn. That's a design change you've scoped but not yet implemented. The one-line root cause: *letterbox padding is treated as real, supervised content on both the input map and the target, so the model faithfully reproduces it.*

**Say it:**
> "The second artifact — and this answers a question I got: why does the padding actually get *rendered*? When a frame isn't square, we pad it with flat bands top and bottom. You'd think that band is just filler and should stay inert. It doesn't, for three reasons — and they're all in the code. First, that padding is *real pixels* — the padded image *is* what the model is trained to reproduce, band included. Second, the segmentation map's band is a real colour — 'Unlabeled' — which actively steers the block in that region. And third, every training image has the same band in the same place, so the model very reliably *learns* to paint it. It's not random noise, because we never add noise — we add a definite flat colour, and diffusion has no notion of 'ignore this area.' The real fix is to stop padding altogether — train at the native aspect ratio so there's no band to learn. That's a design change we've scoped but not yet made."

---

## Slide 26 — Key takeaways *(~55s)*

**Understand it:**
The five one-line memories you want the audience to leave with. Two are general (freeze the giant + train a whisper; conditional not static), and three are *yours* (added a modality with zero core changes; the hard part was real data; architecture explains behaviour). Deliver these fast, one breath each — it's a recap, not new material. If you're short on time elsewhere, protect #3 and #4, because those are your story.

**Say it:**
> "Five things to walk out with. One: freeze the giant, train a whisper — the backbone never moves. Two: the block is *conditional*, not static — a signal reshapes it live. Three, and this is ours: we added a whole new modality — segmentation — with *zero* changes to the model core, by respecting the existing contract. Four: the real work was making it survive real data — colour instead of raw IDs, the 16-bit format, that nineteen-percent alignment bug, the parity rule. And five: understanding the architecture is what let us explain the artifacts *and* find cheap fixes. That's the talk."

---

## Slide 27 — Thank you / Questions *(~30s + Q&A)*

**Understand it:**
Close by returning to the equation — now with "seg map" in it — so the last thing they see is the whole talk in one line. Then open for questions. Keep the cheat-sheet below in your head.

**Say it:**
> "So — one frozen model, one tiny conditional block, and we taught it a new way to see structure, for the cost of a rounding error in parameters. Thank you — I'm happy to take questions."

**Q&A cheat-sheet (have these ready):**
- *Why colour, not class IDs?* → categories have no natural order; a grayscale ramp fakes one; colour keeps classes separable (Slide 18).
- *SegFormer vs Grounded-SAM — which is better?* → trade-offs: SegFormer is general and self-scores (mIoU) but 19 fixed classes; Grounded-SAM has 29 classes and free sim labels but no live inference (Slide 20).
- *Did you change the model?* → no — the mapper, LoRA, and UNet injection are unmodified; we changed only how the map is produced (Slide 16).
- *Where are the numbers?* → structure adherence is verified qualitatively; full metrics await the real cluster run; the harness is built (Slide 23).
- *Why does the pad band appear?* → padding is supervised, real content on both input and target; fix is native aspect ratio (Slide 25).
- *How do you keep depth vs seg comparison fair?* → identical pipeline except the signal; four separated stages, no shared code (Slide 17).
- *What's mIoU?* → intersection-over-union between the generated image's re-segmentation and the input map — a direct "did it follow the layout?" score (Slide 21).
