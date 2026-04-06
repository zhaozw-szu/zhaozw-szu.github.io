import{_ as n,c as a,b as e,o as i}from"./app-DhIhdf2h.js";const l="/assets/02_F1_Comparison-CDnBiB62.png",p="/assets/01_Loss_Comparison-B2L5ftSl.png",d={};function c(r,s){return i(),a("div",null,[...s[0]||(s[0]=[e(`<p>​ 经过之前的训练我们的方法已经稳定得到结果，在这一节，我们对模型进行改进，为解码器加入跳跃连接，让编码器的每次分辨率信息与解码器相互联通，接下来将详细介绍，也就是老师提供的<code>3-skipConnect.py</code></p><p>​ 我们现有的方法在前向传播的过程如下：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>    def forward(self, x):</span></span>
<span class="line"><span>        x = self.conv1(x)</span></span>
<span class="line"><span>        x = self.bn1(x)</span></span>
<span class="line"><span>        x = self.relu(x)</span></span>
<span class="line"><span>        x = self.maxpool(x)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        x = self.layer1(x)</span></span>
<span class="line"><span>        x = self.layer2(x)</span></span>
<span class="line"><span>        x = self.layer3(x)</span></span>
<span class="line"><span>        x = self.layer4(x)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        x = self.decoder(x)</span></span>
<span class="line"><span>        return x</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>​ 我们可以发现，对于解码器而言，其获取到的信息只有编码器提取的高维特征，并没有其他尺度的特征，针对此，我们使用跳跃连接的方法，首先我们先定义解码器：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>self.up4 = UpBlock(512, 256, 256)  # x4 + x3</span></span>
<span class="line"><span>self.up3 = UpBlock(256, 128, 128)  # x + x2</span></span>
<span class="line"><span>self.up2 = UpBlock(128, 64, 64)    # x + x1</span></span>
<span class="line"><span>self.up1 = UpBlock(64, 64, 64)     # x + x0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 1/2 -&gt; 1</span></span>
<span class="line"><span>self.final_conv = nn.Sequential(</span></span>
<span class="line"><span>    DoubleConv(32, 32),</span></span>
<span class="line"><span>    nn.Conv2d(32, 1, kernel_size=1)</span></span>
<span class="line"><span>)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>​ 在随后的前向传播中，我们连接相同尺度的信息：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>def forward(self, x):</span></span>
<span class="line"><span>    # 原图尺寸</span></span>
<span class="line"><span>    input_h, input_w = x.shape[-2:]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    # ===== Encoder =====</span></span>
<span class="line"><span>    x0 = self.conv1(x)     # [B, 64, H/2, W/2]</span></span>
<span class="line"><span>    x0 = self.bn1(x0)</span></span>
<span class="line"><span>    x0 = self.relu(x0)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    x = self.maxpool(x0)   # [B, 64, H/4, W/4]</span></span>
<span class="line"><span>    x1 = self.layer1(x)    # [B, 64, H/4, W/4]</span></span>
<span class="line"><span>    x2 = self.layer2(x1)   # [B,128, H/8, W/8]</span></span>
<span class="line"><span>    x3 = self.layer3(x2)   # [B,256, H/16,W/16]</span></span>
<span class="line"><span>    x4 = self.layer4(x3)   # [B,512, H/32,W/32]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    # ===== Decoder =====</span></span>
<span class="line"><span>    x = self.up4(x4, x3)   # -&gt; 256, 1/16</span></span>
<span class="line"><span>    x = self.up3(x, x2)    # -&gt; 128, 1/8</span></span>
<span class="line"><span>    x = self.up2(x, x1)    # -&gt; 64, 1/4</span></span>
<span class="line"><span>    x = self.up1(x, x0)    # -&gt; 64, 1/2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    x = self.final_up(x)   # -&gt; 32, 1/1</span></span>
<span class="line"><span>    x = self.final_conv(x) # -&gt; 1, 1/1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    if x.shape[-2:] != (input_h, input_w):</span></span>
<span class="line"><span>        x = torch.nn.functional.interpolate(</span></span>
<span class="line"><span>            x, size=(input_h, input_w), mode=&#39;bilinear&#39;, align_corners=False</span></span>
<span class="line"><span>        )</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    return x</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>​ 这样在解码器解码的过程中，可以通过跳跃连接获取相同尺度下的编码器信息。这也就形成了经典网络Unet的基本架构。</p><p>其主要结果如下：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>Epoch 18/20</span></span>
<span class="line"><span>  Train Loss: 0.3359 | Val Loss: 0.3749</span></span>
<span class="line"><span>  Train F1:   0.8514 | Val F1:   0.7924</span></span>
<span class="line"><span>  Train P/R:  0.8028/0.9063</span></span>
<span class="line"><span>  Val   P/R:  0.7614/0.8260</span></span>
<span class="line"><span>  提示：轻度过拟合</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  学习率保持: 0.00005000</span></span>
<span class="line"><span>Epoch 19/20</span></span>
<span class="line"><span>  Train Loss: 0.3239 | Val Loss: 0.3420</span></span>
<span class="line"><span>  Train F1:   0.8769 | Val F1:   0.8186</span></span>
<span class="line"><span>  Train P/R:  0.8736/0.8802</span></span>
<span class="line"><span>  Val   P/R:  0.8327/0.8050</span></span>
<span class="line"><span>  提示：轻度过拟合</span></span>
<span class="line"><span>  已保存最佳模型：best_model_resnet18skipconnect_简单数据集.pth</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  学习率保持: 0.00005000</span></span>
<span class="line"><span>Epoch 20/20</span></span>
<span class="line"><span>  Train Loss: 0.3166 | Val Loss: 0.4180</span></span>
<span class="line"><span>  Train F1:   0.8271 | Val F1:   0.7741</span></span>
<span class="line"><span>  Train P/R:  0.7379/0.9410</span></span>
<span class="line"><span>  Val   P/R:  0.7006/0.8648</span></span>
<span class="line"><span>  提示：轻度过拟合</span></span>
<span class="line"><span></span></span>
<span class="line"><span>  学习率保持: 0.00005000</span></span>
<span class="line"><span>============================================================</span></span>
<span class="line"><span>训练完成</span></span>
<span class="line"><span>============================================================</span></span>
<span class="line"><span>模型：resnet18skipconnect</span></span>
<span class="line"><span>最佳验证 F1：0.8186</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>​ 其与2-augmentation.py比较的结果如图：</p><p><img src="`+l+'" alt="02_F1_Comparison"></p><p><img src="'+p+'" alt="01_Loss_Comparison"></p>',13)])])}const v=n(d,[["render",c]]),u=JSON.parse('{"path":"/posts/gv89ovj1/","title":"4-残差连接 | posts","lang":"zh-CN","frontmatter":{"title":"4-残差连接","createTime":"2026/03/27 16:52:49","permalink":"/posts/gv89ovj1/","description":"​ 经过之前的训练我们的方法已经稳定得到结果，在这一节，我们对模型进行改进，为解码器加入跳跃连接，让编码器的每次分辨率信息与解码器相互联通，接下来将详细介绍，也就是老师提供的3-skipConnect.py ​ 我们现有的方法在前向传播的过程如下： ​ 我们可以发现，对于解码器而言，其获取到的信息只有编码器提取的高维特征，并没有其他尺度的特征，针对此，...","head":[["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"4-残差连接\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2026-04-06T08:51:54.000Z\\",\\"author\\":[]}"],["meta",{"property":"og:url","content":"https://zhaozw-szu.github.io/posts/gv89ovj1/"}],["meta",{"property":"og:site_name","content":"KnowledgeTree"}],["meta",{"property":"og:title","content":"4-残差连接"}],["meta",{"property":"og:description","content":"​ 经过之前的训练我们的方法已经稳定得到结果，在这一节，我们对模型进行改进，为解码器加入跳跃连接，让编码器的每次分辨率信息与解码器相互联通，接下来将详细介绍，也就是老师提供的3-skipConnect.py ​ 我们现有的方法在前向传播的过程如下： ​ 我们可以发现，对于解码器而言，其获取到的信息只有编码器提取的高维特征，并没有其他尺度的特征，针对此，..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2026-04-06T08:51:54.000Z"}],["meta",{"property":"article:modified_time","content":"2026-04-06T08:51:54.000Z"}]]},"readingTime":{"minutes":2.02,"words":606},"git":{"createdTime":1775465514000,"updatedTime":1775465514000,"contributors":[{"name":"zhaozw","username":"zhaozw","email":"2300432033@email.szu.edu.cn","commits":1,"avatar":"https://avatars.githubusercontent.com/zhaozw?v=4","url":"https://github.com/zhaozw"}]},"autoDesc":true,"filePathRelative":"posts/4-残差连接.md","headers":[],"categoryList":[]}');export{v as comp,u as data};
