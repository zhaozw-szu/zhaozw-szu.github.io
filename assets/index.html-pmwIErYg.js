import{_ as n,c as a,b as e,o as i}from"./app-DjcZiNBf.js";const p="/assets/image-20240528220811746-xRQNuFiC.png",l="/assets/image-20240528221000632-C917-34Q.png",d={};function c(t,s){return i(),a("div",null,[...s[0]||(s[0]=[e('<p>Segment Anything Model for Medical Images?</p><p>发表于MICCAI 2024</p><p>Testing pipeline of SAM</p><p><img src="'+p+`" alt="image-20240528220811746"></p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>train with box</span></span>
<span class="line"><span>接下来是的代码来自</span></span>
<span class="line"><span>https://github.com/yuhoo0302/Segment-Anything-Model-for-Medical-Images</span></span>
<span class="line"><span>任务是医学图片的分割</span></span>
<span class="line"><span></span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>优化器和损失函数设计：</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span># Set up the optimizer, hyperparameter tuning will improve performance here</span></span>
<span class="line"><span>optimizer = torch.optim.AdamW(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)</span></span>
<span class="line"><span>seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction=&#39;mean&#39;)</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>为一个训练过程中的代码：</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>outputs = []</span></span>
<span class="line"><span># do not compute gradients for image encoder and prompt encoder</span></span>
<span class="line"><span>with torch.no_grad():</span></span>
<span class="line"><span>    none_grad_features = {&quot;sparse&quot;: {}, &quot;dense&quot;: {}}</span></span>
<span class="line"><span>    for idx, image_record in enumerate(batched_input):</span></span>
<span class="line"><span>        sparse_embeddings, dense_embeddings = model.prompt_encoder(</span></span>
<span class="line"><span>                    points=None,</span></span>
<span class="line"><span>                    boxes=image_record[&quot;box&quot;].to(device),</span></span>
<span class="line"><span>                    masks=None,</span></span>
<span class="line"><span>                )</span></span>
<span class="line"><span>        none_grad_features[&quot;sparse&quot;][idx] = sparse_embeddings</span></span>
<span class="line"><span>        none_grad_features[&quot;dense&quot;][idx] = dense_embeddings </span></span>
<span class="line"><span></span></span>
<span class="line"><span>batched_loss = 0</span></span>
<span class="line"><span>for id, im_record in enumerate(batched_input):</span></span>
<span class="line"><span>    # low_res_masks.shape == (B, M, 256, 256) M is set to 1</span></span>
<span class="line"><span>    low_res_masks, iou_predictions = model.mask_decoder(</span></span>
<span class="line"><span>        image_embeddings=im_record[&quot;img_embed&quot;].unsqueeze(0).to(device), # (1, 256, 64, 64) !!1 = batch size</span></span>
<span class="line"><span>        image_pe=model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64) !!1 = batch size</span></span>
<span class="line"><span>        sparse_prompt_embeddings=none_grad_features[&quot;sparse&quot;][id], # (B, 2, 256) !!B = target num instead of batch size</span></span>
<span class="line"><span>        dense_prompt_embeddings=none_grad_features[&quot;dense&quot;][id], # (B, 256, 64, 64) !!B = target num instead of batch size</span></span>
<span class="line"><span>        multimask_output=False,</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    # upscale + eliminate padding + restore to ori size</span></span>
<span class="line"><span>    masks = model.postprocess_masks(</span></span>
<span class="line"><span>        low_res_masks,</span></span>
<span class="line"><span>        input_size=tuple(im_record[&quot;size_before_pad&quot;]),</span></span>
<span class="line"><span>        original_size=tuple(im_record[&quot;image_ori_size&quot;]),</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    outputs.append({</span></span>
<span class="line"><span>        &quot;masks&quot;: masks,</span></span>
<span class="line"><span>        &quot;iou_predictions&quot;: iou_predictions,</span></span>
<span class="line"><span>        &quot;low_res_logits&quot;: low_res_masks,</span></span>
<span class="line"><span>        &quot;gt2D&quot;: im_record[&quot;gt2D&quot;].to(device)</span></span>
<span class="line"><span>    })</span></span>
<span class="line"><span>    # first ele: 1, B, ori_H, ori_W</span></span>
<span class="line"><span>    # second ele: 1, B, ori_H, ori_W</span></span>
<span class="line"><span>    # considering the multi-object situation</span></span>
<span class="line"><span>    batched_loss += criterion(masks.squeeze(1).unsqueeze(0), im_record[&quot;gt2D&quot;].to(device).unsqueeze(0)) </span></span>
<span class="line"><span>loss = batched_loss / len(batched_input)</span></span>
<span class="line"><span>optimizer.zero_grad()</span></span>
<span class="line"><span>loss.backward()</span></span>
<span class="line"><span>optimizer.step()</span></span>
<span class="line"><span>epoch_loss += loss.item()</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Segment Anything in High Quality</p><p>发表于NeurIPS 2023</p><p><img src="`+l+`" alt="image-20240528221000632"></p><p>图3:HQ-SAM将HQ输出令牌和全局局部特征融合引入SAM，用于高质量掩模预测。为了保持SAM的零样本能力，轻量级HQ-Output-Token重用SAM的掩码解码器，并生成新的MLP层，用于执行具有融合HQ-Features的点向产品。在训练过程中，当我们固定预先训练的SAM的模型参数时，HQ-SAM中只有少数可学习的参数是可训练的。为了清晰起见，此处省略了提示编码器。误差校正简单地用作推理期间SAM的输出令牌和HQ输出令牌的预测logits之间的直接元素和。</p><p>损失函数设置</p><p>We supervise mask prediction of the new HQ-Output token with a combination of both BCE Loss and Dice Loss.</p><p>我们用BCE损失和Dice 损失组合的联合损失监督新HQ输出token的掩码预测。</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>train with box、point、noise_mask</span></span>
<span class="line"><span>接下来是的代码来自</span></span>
<span class="line"><span>https://github.com/SysCV/SAM-HQ</span></span>
<span class="line"><span>任务是SAM的高质量掩模预测问题</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>from utils.loss_mask import loss_masks</span></span>
<span class="line"><span></span></span>
<span class="line"><span>net = MaskDecoderHQ(args.model_type)</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>if input_type == &#39;box&#39;:</span></span>
<span class="line"><span>	dict_input[&#39;boxes&#39;] = labels_box[b_i:b_i+1]</span></span>
<span class="line"><span>elif input_type == &#39;point&#39;:</span></span>
<span class="line"><span>	point_coords = labels_points[b_i:b_i+1]</span></span>
<span class="line"><span>    dict_input[&#39;point_coords&#39;] = point_coords</span></span>
<span class="line"><span>    dict_input[&#39;point_labels&#39;] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]</span></span>
<span class="line"><span>elif input_type == &#39;noise_mask&#39;:</span></span>
<span class="line"><span>    dict_input[&#39;mask_inputs&#39;] = labels_noisemask[b_i:b_i+1]</span></span>
<span class="line"><span>else:</span></span>
<span class="line"><span>    raise NotImplementedError</span></span>
<span class="line"><span>dict_input[&#39;original_size&#39;] = imgs[b_i].shape[:2]</span></span>
<span class="line"><span>batched_input.append(dict_input)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>with torch.no_grad():</span></span>
<span class="line"><span>	batched_output, interm_embeddings = sam(batched_input, multimask_output=False)</span></span>
<span class="line"><span>	</span></span>
<span class="line"><span>batch_len = len(batched_output)</span></span>
<span class="line"><span>encoder_embedding = torch.cat([batched_output[i_l][&#39;encoder_embedding&#39;] for i_l in range(batch_len)], dim=0)</span></span>
<span class="line"><span>image_pe = [batched_output[i_l][&#39;image_pe&#39;] for i_l in range(batch_len)]</span></span>
<span class="line"><span>sparse_embeddings = [batched_output[i_l][&#39;sparse_embeddings&#39;] for i_l in range(batch_len)]</span></span>
<span class="line"><span>dense_embeddings = [batched_output[i_l][&#39;dense_embeddings&#39;] for i_l in range(batch_len)]</span></span>
<span class="line"><span>masks_hq = net(</span></span>
<span class="line"><span>    image_embeddings=encoder_embedding,</span></span>
<span class="line"><span>    image_pe=image_pe,</span></span>
<span class="line"><span>    sparse_prompt_embeddings=sparse_embeddings,</span></span>
<span class="line"><span>    dense_prompt_embeddings=dense_embeddings,</span></span>
<span class="line"><span>    multimask_output=False,</span></span>
<span class="line"><span>    hq_token_only=True,</span></span>
<span class="line"><span>    interm_embeddings=interm_embeddings,</span></span>
<span class="line"><span>)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))</span></span>
<span class="line"><span>loss = loss_mask + loss_dice</span></span>
<span class="line"><span></span></span>
<span class="line"><span>loss_dict = {&quot;loss_mask&quot;: loss_mask, &quot;loss_dice&quot;:loss_dice}</span></span>
<span class="line"><span></span></span>
<span class="line"><span># reduce losses over all GPUs for logging purposes</span></span>
<span class="line"><span>loss_dict_reduced = misc.reduce_dict(loss_dict)</span></span>
<span class="line"><span>losses_reduced_scaled = sum(loss_dict_reduced.values())</span></span>
<span class="line"><span>loss_value = losses_reduced_scaled.item()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>optimizer.zero_grad()</span></span>
<span class="line"><span>loss.backward()</span></span>
<span class="line"><span>optimizer.step()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>metric_logger.update(training_loss=loss_value, **loss_dict_reduced)</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>loss_masks：</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0):</span></span>
<span class="line"><span>    &quot;&quot;&quot;Compute the losses related to the masks: the focal loss and the dice loss.</span></span>
<span class="line"><span>    targets dicts must contain the key &quot;masks&quot; containing a tensor of dim [nb_target_boxes, h, w]</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    # No need to upsample predictions as we are using normalized coordinates :)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    with torch.no_grad():</span></span>
<span class="line"><span>        # sample point_coords</span></span>
<span class="line"><span>        point_coords = get_uncertain_point_coords_with_randomness(</span></span>
<span class="line"><span>            src_masks,</span></span>
<span class="line"><span>            lambda logits: calculate_uncertainty(logits),</span></span>
<span class="line"><span>            112 * 112,</span></span>
<span class="line"><span>            oversample_ratio,</span></span>
<span class="line"><span>            0.75,</span></span>
<span class="line"><span>        )</span></span>
<span class="line"><span>        # get gt labels</span></span>
<span class="line"><span>        point_labels = point_sample(</span></span>
<span class="line"><span>            target_masks,</span></span>
<span class="line"><span>            point_coords,</span></span>
<span class="line"><span>            align_corners=False,</span></span>
<span class="line"><span>        ).squeeze(1)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    point_logits = point_sample(</span></span>
<span class="line"><span>        src_masks,</span></span>
<span class="line"><span>        point_coords,</span></span>
<span class="line"><span>        align_corners=False,</span></span>
<span class="line"><span>    ).squeeze(1)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    loss_mask = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks)</span></span>
<span class="line"><span>    loss_dice = dice_loss_jit(point_logits, point_labels, num_masks)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    del src_masks</span></span>
<span class="line"><span>    del target_masks</span></span>
<span class="line"><span>    return loss_mask, loss_dice</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span>
<span class="line"><span>get_uncertain_point_coords_with_randomness：</span></span>
<span class="line"><span> &quot;&quot;&quot;</span></span>
<span class="line"><span>    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties</span></span>
<span class="line"><span>        are calculated for each point using &#39;uncertainty_func&#39; function that takes point&#39;s logit</span></span>
<span class="line"><span>        prediction as input.</span></span>
<span class="line"><span>    See PointRend paper for details.</span></span>
<span class="line"><span>    Args:</span></span>
<span class="line"><span>        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for</span></span>
<span class="line"><span>            class-specific or class-agnostic prediction.</span></span>
<span class="line"><span>        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that</span></span>
<span class="line"><span>            contains logit predictions for P points and returns their uncertainties as a Tensor of</span></span>
<span class="line"><span>            shape (N, 1, P).</span></span>
<span class="line"><span>        num_points (int): The number of points P to sample.</span></span>
<span class="line"><span>        oversample_ratio (int): Oversampling parameter.</span></span>
<span class="line"><span>        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.</span></span>
<span class="line"><span>    Returns:</span></span>
<span class="line"><span>        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P</span></span>
<span class="line"><span>            sampled points.</span></span>
<span class="line"><span> &quot;&quot;&quot;</span></span>
<span class="line"><span>point_sample：</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    A wrapper around :function:\`torch.nn.functional.grid_sample\` to support 3D point_coords tensors.</span></span>
<span class="line"><span>    Unlike :function:\`torch.nn.functional.grid_sample\` it assumes \`point_coords\` to lie inside</span></span>
<span class="line"><span>    [0, 1] x [0, 1] square.</span></span>
<span class="line"><span>    Args:</span></span>
<span class="line"><span>        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.</span></span>
<span class="line"><span>        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains</span></span>
<span class="line"><span>        [0, 1] x [0, 1] normalized point coordinates.</span></span>
<span class="line"><span>    Returns:</span></span>
<span class="line"><span>        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains</span></span>
<span class="line"><span>            features for points in \`point_coords\`. The features are obtained via bilinear</span></span>
<span class="line"><span>            interplation from \`input\` the same way as :function:\`torch.nn.functional.grid_sample\`.</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>sigmoid_ce_loss_jit = torch.jit.script(</span></span>
<span class="line"><span>    sigmoid_ce_loss</span></span>
<span class="line"><span>)  # type: torch.jit.ScriptModule</span></span>
<span class="line"><span>def sigmoid_ce_loss(</span></span>
<span class="line"><span>        inputs: torch.Tensor,</span></span>
<span class="line"><span>        targets: torch.Tensor,</span></span>
<span class="line"><span>        num_masks: float,</span></span>
<span class="line"><span>    ):</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    Args:</span></span>
<span class="line"><span>        inputs: A float tensor of arbitrary shape.</span></span>
<span class="line"><span>                The predictions for each example.</span></span>
<span class="line"><span>        targets: A float tensor with the same shape as inputs. Stores the binary</span></span>
<span class="line"><span>                 classification label for each element in inputs</span></span>
<span class="line"><span>                (0 for the negative class and 1 for the positive class).</span></span>
<span class="line"><span>    Returns:</span></span>
<span class="line"><span>        Loss tensor</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=&quot;none&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    return loss.mean(1).sum() / num_masks</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>dice_loss_jit = torch.jit.script(</span></span>
<span class="line"><span>    dice_loss</span></span>
<span class="line"><span>)  # type: torch.jit.ScriptModule</span></span>
<span class="line"><span>def dice_loss(</span></span>
<span class="line"><span>        inputs: torch.Tensor,</span></span>
<span class="line"><span>        targets: torch.Tensor,</span></span>
<span class="line"><span>        num_masks: float,</span></span>
<span class="line"><span>    ):</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    Compute the DICE loss, similar to generalized IOU for masks</span></span>
<span class="line"><span>    Args:</span></span>
<span class="line"><span>        inputs: A float tensor of arbitrary shape.</span></span>
<span class="line"><span>                The predictions for each example.</span></span>
<span class="line"><span>        targets: A float tensor with the same shape as inputs. Stores the binary</span></span>
<span class="line"><span>                 classification label for each element in inputs</span></span>
<span class="line"><span>                (0 for the negative class and 1 for the positive class).</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    inputs = inputs.sigmoid()</span></span>
<span class="line"><span>    inputs = inputs.flatten(1)</span></span>
<span class="line"><span>    numerator = 2 * (inputs * targets).sum(-1)</span></span>
<span class="line"><span>    denominator = inputs.sum(-1) + targets.sum(-1)</span></span>
<span class="line"><span>    loss = 1 - (numerator + 1) / (denominator + 1)</span></span>
<span class="line"><span>    return loss.sum() / num_masks</span></span>
<span class="line"><span>------------------------------------------------------------------------------------------------------------</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,13)])])}const o=n(d,[["render",c]]),u=JSON.parse('{"path":"/papers/rd7augeu/","title":"SAM1 | papers","lang":"zh-CN","frontmatter":{"title":"SAM1","date":"2024-05-28T21:03:57.000Z","tags":null,"createTime":"2024/05/28 21:03:57","permalink":"/papers/rd7augeu/","description":"Segment Anything Model for Medical Images? 发表于MICCAI 2024 Testing pipeline of SAM image-20240528220811746 Segment Anything in High Quality 发表于NeurIPS 2023 image-2024052822100063...","head":[["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"SAM1\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2024-05-28T21:03:57.000Z\\",\\"dateModified\\":\\"2026-03-06T04:01:25.000Z\\",\\"author\\":[]}"],["meta",{"property":"og:url","content":"https://zhaozw-szu.github.io/papers/rd7augeu/"}],["meta",{"property":"og:site_name","content":"KnowledgeTree"}],["meta",{"property":"og:title","content":"SAM1"}],["meta",{"property":"og:description","content":"Segment Anything Model for Medical Images? 发表于MICCAI 2024 Testing pipeline of SAM image-20240528220811746 Segment Anything in High Quality 发表于NeurIPS 2023 image-2024052822100063..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2026-03-06T04:01:25.000Z"}],["meta",{"property":"article:published_time","content":"2024-05-28T21:03:57.000Z"}],["meta",{"property":"article:modified_time","content":"2026-03-06T04:01:25.000Z"}]]},"readingTime":{"minutes":4.15,"words":1245},"git":{"createdTime":1772769685000,"updatedTime":1772769685000,"contributors":[{"name":"zhaozw","username":"zhaozw","email":"2300432033@email.szu.edu.cn","commits":1,"avatar":"https://avatars.githubusercontent.com/zhaozw?v=4","url":"https://github.com/zhaozw"}]},"autoDesc":true,"filePathRelative":"papers/SAM1.md","headers":[]}');export{o as comp,u as data};
