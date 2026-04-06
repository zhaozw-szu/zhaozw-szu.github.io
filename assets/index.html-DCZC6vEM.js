import{_ as n,c as a,b as i,o as e}from"./app-DjcZiNBf.js";const l="/assets/image-20260326113335887-DidimmPk.png",p="/assets/image-20260326115022497-CytVxPk7.png",d="/assets/image-20260326115314094-CYKgZ2zY.png",c="/assets/image-20260326120014613-BKb8lvOu.png",r="/assets/cnn-Q2ST43Uc.png",v="/assets/Tp_D_NND_L_B_arc00072_nat00097_00292_%E5%AF%B9%E6%AF%94-B18QnMg-.png",t="/assets/Tp_D_NRN_M_B_art00014_art00011_11835_%E5%AF%B9%E6%AF%94-6E34SJq5.png",m="/assets/Tp_D_NNN_M_O_arc00041_nat00097_00261_%E5%AF%B9%E6%AF%94-Ufiq_n5f.png",u="/assets/Tp_S_CRN_M_N_pla00037_pla00037_10981_%E5%AF%B9%E6%AF%94-Dzdp5pMv.png",o="/assets/Tp_S_NNN_S_N_pla00016_pla00016_00555_%E5%AF%B9%E6%AF%94-wNmJCP7B.png",_="/assets/Tp_D_NRN_M_N_nat10143_nat00095_12035_%E5%AF%B9%E6%AF%94-Cr79wQf3.png",b="/assets/Tp_D_NRN_M_N_art00012_art00014_11815_%E5%AF%B9%E6%AF%94-BaNcu5az.png",g="/assets/Tp_D_NNN_S_N_nat00003_cha00096_00623_%E5%AF%B9%E6%AF%94-B7rHedri.png",h={};function f(k,s){return e(),a("div",null,[...s[0]||(s[0]=[i('<p>我们这这次的任务是完成一个图像篡改定位的任务，我们使用的框架是pytorch，编译器是pycharm，在上一节，我们教了如何安装pytorch等环境，这一节我们将学习如何导入环境和跑通baseline</p><h2 id="_1-导入项目文件" tabindex="-1"><a class="header-anchor" href="#_1-导入项目文件"><span>1.导入项目文件</span></a></h2><p>（1）打开Pycharm软件，打开项目文件夹，选择你下载本地的项目文件：<br><img src="'+l+'" alt="image-20260326113335887"></p><p>（2）打开Pycharm软件，打开项目文件夹，选择你下载到本地的项目文件：</p><p><img src="'+p+'" alt="image-20260326115022497"></p><p>其中数据集是我们的数据，images文件夹中是篡改图像，masks文件夹中是对应的mask：</p><p><img src="'+d+'" alt="image-20260326115314094"></p><p>（3）选择已导入的env环境</p><p><img src="'+c+`" alt="image-20260326120014613"></p><p>pip install scikit-learn</p><h2 id="_2-跑通baseline" tabindex="-1"><a class="header-anchor" href="#_2-跑通baseline"><span>2.跑通baseline</span></a></h2><p>0-baseline.py 为老师提供的代码，其模型是一个简单的CNN，具体的构成如下：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>class SimpleCNN(nn.Module):</span></span>
<span class="line"><span>    def __init__(self):</span></span>
<span class="line"><span>        super(SimpleCNN, self).__init__()</span></span>
<span class="line"><span>        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)</span></span>
<span class="line"><span>        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)</span></span>
<span class="line"><span>        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)</span></span>
<span class="line"><span>        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)</span></span>
<span class="line"><span>        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    def forward(self, x):</span></span>
<span class="line"><span>        x = F.relu(self.conv1_1(x))</span></span>
<span class="line"><span>        x = F.relu(self.conv1_2(x))</span></span>
<span class="line"><span>        x = self.pool1(x)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        x = F.relu(self.conv2_1(x))</span></span>
<span class="line"><span>        x = F.relu(self.conv2_2(x))</span></span>
<span class="line"><span>        x = self.pool2(x)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        x = F.interpolate(x, size=(256, 256), mode=&#39;bilinear&#39;, align_corners=False)</span></span>
<span class="line"><span>        logits = self.conv_out(x)</span></span>
<span class="line"><span>        return logits</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>具体而言，其是通过卷积、池化、上采样完成任务的，其尺度-通道数变化如下：</p><p><img src="`+r+`" alt="cnn"></p><p>接下来我将具体的解释所有的代码：</p><p>首先是数据加载部分：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>class TamperDataset(Dataset):</span></span>
<span class="line"><span>    &quot;&quot;&quot;图像篡改检测数据集类</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    Args:</span></span>
<span class="line"><span>        image_dir: 图像目录路径</span></span>
<span class="line"><span>        mask_dir: Mask 目录路径</span></span>
<span class="line"><span>        img_size: 目标图像尺寸</span></span>
<span class="line"><span>        choice: &#39;train&#39; 或 &#39;val&#39;，决定是否应用数据增强</span></span>
<span class="line"><span>    &quot;&quot;&quot;</span></span>
<span class="line"><span>    def __init__(self, image_dir, mask_dir, img_size=256, choice=&#39;train&#39;):</span></span>
<span class="line"><span>        self.image_dir = Path(image_dir)</span></span>
<span class="line"><span>        self.mask_dir = Path(mask_dir)</span></span>
<span class="line"><span>        self.img_size = img_size</span></span>
<span class="line"><span>        self.choice = choice</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        self.image_files = sorted(list(self.image_dir.iterdir()))</span></span>
<span class="line"><span>        print(f&quot;找到 {len(self.image_files)} 张图像&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        if self.choice == &#39;train&#39;:</span></span>
<span class="line"><span>            self.albu = A.Compose([</span></span>
<span class="line"><span>                A.RandomScale(scale_limit=(-0.5, 0.0), p=0.75),</span></span>
<span class="line"><span>                A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, p=1.0),</span></span>
<span class="line"><span></span></span>
<span class="line"><span>                A.OneOf([</span></span>
<span class="line"><span>                    A.HorizontalFlip(p=1.0),</span></span>
<span class="line"><span>                    A.VerticalFlip(p=1.0),</span></span>
<span class="line"><span>                    A.RandomRotate90(p=1.0),</span></span>
<span class="line"><span>                    A.Transpose(p=1.0),</span></span>
<span class="line"><span>                ], p=0.75),</span></span>
<span class="line"><span>                A.ImageCompression(quality_range=(50, 95), compression_type=&quot;jpeg&quot;, p=0.75),</span></span>
<span class="line"><span>                A.OneOf([</span></span>
<span class="line"><span>                    A.OneOf([</span></span>
<span class="line"><span>                        A.Blur(blur_limit=3, p=1.0),</span></span>
<span class="line"><span>                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),</span></span>
<span class="line"><span>                        A.MedianBlur(blur_limit=3, p=1.0),</span></span>
<span class="line"><span>                        A.MotionBlur(blur_limit=5, p=1.0),</span></span>
<span class="line"><span>                    ], p=1.0),</span></span>
<span class="line"><span></span></span>
<span class="line"><span>                    A.OneOf([</span></span>
<span class="line"><span>                        A.Downscale(scale_range=(0.5, 0.9), p=1.0),</span></span>
<span class="line"><span>                        A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=1.0),</span></span>
<span class="line"><span>                        A.ISONoise(p=1.0),</span></span>
<span class="line"><span>                        A.RandomBrightnessContrast(p=1.0),</span></span>
<span class="line"><span>                        A.RandomGamma(p=1.0),</span></span>
<span class="line"><span>                        A.Sharpen(p=1.0),</span></span>
<span class="line"><span>                    ], p=1.0),</span></span>
<span class="line"><span>                ], p=0.25),</span></span>
<span class="line"><span>                A.Resize(self.img_size, self.img_size)</span></span>
<span class="line"><span>            ])</span></span>
<span class="line"><span>        else:</span></span>
<span class="line"><span>            self.albu = A.Compose([</span></span>
<span class="line"><span>                A.Resize(self.img_size, self.img_size)</span></span>
<span class="line"><span>            ])</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    def __len__(self):</span></span>
<span class="line"><span>        return len(self.image_files)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    def __getitem__(self, idx):</span></span>
<span class="line"><span>        img_path = self.image_files[idx]</span></span>
<span class="line"><span>        mask_path = self.mask_dir / (img_path.stem + &quot;_gt.png&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        if not mask_path.exists():</span></span>
<span class="line"><span>            raise FileNotFoundError(f&quot;未找到对应 mask: {mask_path}&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        image = np.array(Image.open(img_path).convert(&quot;RGB&quot;))</span></span>
<span class="line"><span>        mask = np.array(Image.open(mask_path).convert(&quot;L&quot;))</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        # train: 75% 概率增强；val: 只做 resize</span></span>
<span class="line"><span>        if self.choice == &#39;train&#39; and random.random() &lt; 0.75:</span></span>
<span class="line"><span>            aug = self.albu(image=image, mask=mask)</span></span>
<span class="line"><span>            image = aug[&#39;image&#39;]</span></span>
<span class="line"><span>            mask = aug[&#39;mask&#39;]</span></span>
<span class="line"><span>        else:</span></span>
<span class="line"><span>            aug = A.Compose([</span></span>
<span class="line"><span>                A.Resize(self.img_size, self.img_size)</span></span>
<span class="line"><span>            ])(image=image, mask=mask)</span></span>
<span class="line"><span>            image = aug[&#39;image&#39;]</span></span>
<span class="line"><span>            mask = aug[&#39;mask&#39;]</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        image = image.astype(np.float32) / 255.0</span></span>
<span class="line"><span>        mask = mask.astype(np.float32)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        mask = (mask &gt; 127).astype(np.float32)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        image = torch.from_numpy(image).permute(2, 0, 1)</span></span>
<span class="line"><span>        mask = torch.from_numpy(mask).unsqueeze(0)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        return image, mask</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>接下的方法是载入训练集和测试集，并构建dataset和DataLoader：</p><p>加载数据并划分训练集和验证集</p><ul><li>train_loader 是用于训练的 加载器</li><li>val_val_loader 是用于评估测试集的 加载器</li><li>val_train_loader 是用于评估训练集的 加载器</li></ul><p>train_loader和val_train_loader的区别在于：</p><ul><li>前者用于训练模型，后者用于评估训练集上的性能，以检测过拟合情况</li><li>前者进行了数据增强，后者没有进行数据增强</li></ul><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>def load_and_split_data(batch_size=8, img_size=256):</span></span>
<span class="line"><span>    # 训练集（用于训练）</span></span>
<span class="line"><span>    train_dataset = TamperDataset(</span></span>
<span class="line"><span>        image_dir=&quot;数据集/训练集/images&quot;,</span></span>
<span class="line"><span>        mask_dir=&quot;数据集/训练集/masks&quot;,</span></span>
<span class="line"><span>        img_size=img_size,</span></span>
<span class="line"><span>        choice=&#39;train&#39;</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    # 评估测试集</span></span>
<span class="line"><span>    val_val_dataset = TamperDataset(</span></span>
<span class="line"><span>        image_dir=&quot;数据集/测试集/images&quot;,</span></span>
<span class="line"><span>        mask_dir=&quot;数据集/测试集/masks&quot;,</span></span>
<span class="line"><span>        img_size=img_size,</span></span>
<span class="line"><span>        choice=&#39;val&#39;</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    # 评估训练集</span></span>
<span class="line"><span>    val_train_dataset = TamperDataset(</span></span>
<span class="line"><span>        image_dir=&quot;数据集/训练集/images&quot;,</span></span>
<span class="line"><span>        mask_dir=&quot;数据集/训练集/masks&quot;,</span></span>
<span class="line"><span>        img_size=img_size,</span></span>
<span class="line"><span>        choice=&#39;val&#39;</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    print(f&quot;\\n✓ 训练集：{len(train_dataset)} 张图像&quot;)</span></span>
<span class="line"><span>    print(f&quot;✓ 验证集：{len(val_val_dataset)} 张图像&quot;)</span></span>
<span class="line"><span>    print(f&quot;✓ 评估训练集：{len(val_train_dataset)} 张图像&quot;)</span></span>
<span class="line"><span>    print(f&quot;✓ 实际训练比例：{len(train_dataset) / (len(train_dataset) + len(val_val_dataset)):.2%}&quot;)</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    # 创建数据加载器</span></span>
<span class="line"><span>    train_loader = DataLoader(</span></span>
<span class="line"><span>        train_dataset,</span></span>
<span class="line"><span>        batch_size=batch_size,</span></span>
<span class="line"><span>        shuffle=True,</span></span>
<span class="line"><span>        num_workers=0,</span></span>
<span class="line"><span>        pin_memory=torch.cuda.is_available()</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    val_val_loader = DataLoader(</span></span>
<span class="line"><span>        val_val_dataset,</span></span>
<span class="line"><span>        batch_size=batch_size,</span></span>
<span class="line"><span>        shuffle=False,</span></span>
<span class="line"><span>        num_workers=0,</span></span>
<span class="line"><span>        pin_memory=torch.cuda.is_available()</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    val_train_loader = DataLoader(</span></span>
<span class="line"><span>        val_train_dataset,</span></span>
<span class="line"><span>        batch_size=batch_size,</span></span>
<span class="line"><span>        shuffle=False,</span></span>
<span class="line"><span>        num_workers=0,</span></span>
<span class="line"><span>        pin_memory=torch.cuda.is_available()</span></span>
<span class="line"><span>    )</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    return train_loader, val_val_loader, val_train_loader</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>训练一个epoch的代码如下；</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>def train_one_epoch(model, train_loader, criterion, optimizer, device):</span></span>
<span class="line"><span>    model.train()</span></span>
<span class="line"><span>    total_loss = 0.0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    for images, masks in train_loader:</span></span>
<span class="line"><span>        images = images.to(device)</span></span>
<span class="line"><span>        masks = masks.to(device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        optimizer.zero_grad()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        logits = model(images)</span></span>
<span class="line"><span>        loss = criterion(logits, masks)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        loss.backward()</span></span>
<span class="line"><span>        optimizer.step()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        total_loss += loss.item()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    return total_loss / len(train_loader)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>接下来是训练流程：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>    # ========= 配置 =========</span></span>
<span class="line"><span>    # 训练轮次</span></span>
<span class="line"><span>    EPOCHS = 20</span></span>
<span class="line"><span>    # 训练批次大小</span></span>
<span class="line"><span>    BATCH_SIZE = 8</span></span>
<span class="line"><span>    # 学习率</span></span>
<span class="line"><span>    LEARNING_RATE = 1e-4</span></span>
<span class="line"><span>    # 输入图像大小</span></span>
<span class="line"><span>    IMG_SIZE = 256</span></span>
<span class="line"><span>    # 模型和数据集名称（用于保存模型和日志）</span></span>
<span class="line"><span>    MODEL_NAME = &quot;SimpleCNN&quot;</span></span>
<span class="line"><span>    DATA_NAME = f&quot;{datetime.datetime.now().strftime(&#39;%H_%M_%S&#39;)}&quot;</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    # 设置模型使用的设备</span></span>
<span class="line"><span>    device = torch.device(&quot;cuda&quot; if torch.cuda.is_available() else &quot;cpu&quot;)</span></span>
<span class="line"><span>    print(f&quot;使用设备：{device}\\n&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>然后是定义训练的损失函数和优化函数：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>    criterion = nn.BCEWithLogitsLoss()</span></span>
<span class="line"><span>    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div></div></div><p>其训练流程如下：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span></span></span>
<span class="line"><span>    for epoch in range(EPOCHS):</span></span>
<span class="line"><span>        # 训练一个 epoch，并返回训练损失</span></span>
<span class="line"><span>        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        # 评估训练集的性能</span></span>
<span class="line"><span>        train_eval_loss, train_prec, train_rec, train_f1 = evaluate(</span></span>
<span class="line"><span>            model, val_train_loader, criterion, device</span></span>
<span class="line"><span>        )</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        # 评估验证集的性能</span></span>
<span class="line"><span>        val_loss, val_prec, val_rec, val_f1 = evaluate(</span></span>
<span class="line"><span>            model, val_val_loader, criterion, device</span></span>
<span class="line"><span>        )</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        print(f&quot;Epoch {epoch+1}/{EPOCHS}&quot;)</span></span>
<span class="line"><span>        print(f&quot;  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}&quot;)</span></span>
<span class="line"><span>        print(f&quot;  Train F1:   {train_f1:.4f} | Val F1:   {val_f1:.4f}&quot;)</span></span>
<span class="line"><span>        print(f&quot;  Train P/R:  {train_prec:.4f}/{train_rec:.4f}&quot;)</span></span>
<span class="line"><span>        print(f&quot;  Val   P/R:  {val_prec:.4f}/{val_rec:.4f}&quot;)</span></span>
<span class="line"><span>        log += f&quot;Epoch {epoch+1}/{EPOCHS}\\n&quot;</span></span>
<span class="line"><span>        log += f&quot;  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\\n&quot;   </span></span>
<span class="line"><span>        log += f&quot;  Train F1:   {train_f1:.4f} | Val F1:   {val_f1:.4f}\\n&quot;</span></span>
<span class="line"><span>        log += f&quot;  Train P/R:  {train_prec:.4f}/{train_rec:.4f}\\n&quot;</span></span>
<span class="line"><span>        log += f&quot;  Val   P/R:  {val_prec:.4f}/{val_rec:.4f}\\n&quot;</span></span>
<span class="line"><span>        if val_f1 &gt; best_val_f1:</span></span>
<span class="line"><span>            best_val_f1 = val_f1</span></span>
<span class="line"><span>            # 保存最佳模型</span></span>
<span class="line"><span>            torch.save(</span></span>
<span class="line"><span>                {</span></span>
<span class="line"><span>                    &quot;epoch&quot;: epoch + 1,</span></span>
<span class="line"><span>                    &quot;model_name&quot;: MODEL_NAME,</span></span>
<span class="line"><span>                    &quot;model_state_dict&quot;: model.state_dict(),</span></span>
<span class="line"><span>                    &quot;optimizer_state_dict&quot;: optimizer.state_dict(),</span></span>
<span class="line"><span>                    &quot;val_f1&quot;: val_f1,</span></span>
<span class="line"><span>                },</span></span>
<span class="line"><span>                f&quot;权重/best_model_{MODEL_NAME}.pth&quot;</span></span>
<span class="line"><span>            )</span></span>
<span class="line"><span>            print(f&quot;  已保存最佳模型：权重/best_model_{MODEL_NAME}.pth&quot;)</span></span>
<span class="line"><span></span></span>
<span class="line"><span>        print()</span></span>
<span class="line"><span></span></span>
<span class="line"><span>    print(&quot;=&quot; * 60)</span></span>
<span class="line"><span>    print(&quot;训练完成&quot;)</span></span>
<span class="line"><span>    print(&quot;=&quot; * 60)</span></span>
<span class="line"><span>    print(f&quot;模型：{MODEL_NAME}&quot;)</span></span>
<span class="line"><span>    print(f&quot;最佳验证 F1：{best_val_f1:.4f}&quot;)</span></span>
<span class="line"><span>    # 添加总结信息到日志</span></span>
<span class="line"><span>    log += &quot;=&quot; * 60 + &quot;\\n&quot;</span></span>
<span class="line"><span>    log += &quot;训练完成\\n&quot;</span></span>
<span class="line"><span>    log += &quot;=&quot; * 60 + &quot;\\n&quot;</span></span>
<span class="line"><span>    log += f&quot;模型：{MODEL_NAME}\\n&quot;</span></span>
<span class="line"><span>    log += f&quot;最佳验证 F1: {best_val_f1:.4f}\\n&quot;</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    # 训练结束后，一次性保存到文件</span></span>
<span class="line"><span>    with open(f&quot;日志/train_log_{MODEL_NAME}_{DATA_NAME}.txt&quot;, &#39;w&#39;, encoding=&#39;utf-8&#39;) as f:</span></span>
<span class="line"><span>        f.write(log)</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    print(f&quot;\\n日志已保存到：日志/train_log_{MODEL_NAME}_{DATA_NAME}.txt&quot;)</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>其结果最后如下：</p><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#393a34;--shiki-dark:#dbd7caee;--shiki-light-bg:#ffffff;--shiki-dark-bg:#121212;"><pre class="shiki shiki-themes vitesse-light vitesse-dark vp-code"><code class="language-"><span class="line"><span>...</span></span>
<span class="line"><span>Epoch 8/10</span></span>
<span class="line"><span>  Train Loss: 0.4205 | Val Loss: 0.4359</span></span>
<span class="line"><span>  Train F1:   0.5127 | Val F1:   0.4719</span></span>
<span class="line"><span>  Train P/R:  0.7463/0.3904</span></span>
<span class="line"><span>  Val   P/R:  0.7862/0.3371</span></span>
<span class="line"><span>  提示：训练/验证较接近</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Epoch 9/10</span></span>
<span class="line"><span>  Train Loss: 0.4036 | Val Loss: 0.4273</span></span>
<span class="line"><span>  Train F1:   0.5755 | Val F1:   0.5711</span></span>
<span class="line"><span>  Train P/R:  0.7148/0.4817</span></span>
<span class="line"><span>  Val   P/R:  0.7652/0.4556</span></span>
<span class="line"><span>  提示：训练/验证较接近</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Epoch 10/10</span></span>
<span class="line"><span>  Train Loss: 0.4077 | Val Loss: 0.4140</span></span>
<span class="line"><span>  Train F1:   0.5932 | Val F1:   0.6232</span></span>
<span class="line"><span>  Train P/R:  0.6063/0.5806</span></span>
<span class="line"><span>  Val   P/R:  0.6744/0.5792</span></span>
<span class="line"><span>  提示：训练/验证较接近</span></span>
<span class="line"><span></span></span>
<span class="line"><span>============================================================</span></span>
<span class="line"><span>训练完成</span></span>
<span class="line"><span>============================================================</span></span>
<span class="line"><span>模型：baseline</span></span>
<span class="line"><span>最佳验证 F1：0.6233</span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>可视化结果：</p><p>simpleCNN:</p><p>好的样本：</p><p><img src="`+v+'" alt="Tp_D_NND_L_B_arc00072_nat00097_00292_对比"></p><p><img src="'+t+'" alt="Tp_D_NRN_M_B_art00014_art00011_11835_对比"></p><p>中等的样本：</p><p><img src="'+m+'" alt="Tp_D_NNN_M_O_arc00041_nat00097_00261_对比"></p><p><img src="'+u+'" alt="Tp_S_CRN_M_N_pla00037_pla00037_10981_对比"></p><p><img src="'+o+'" alt="Tp_S_NNN_S_N_pla00016_pla00016_00555_对比"></p><p>差的样本：</p><p><img src="'+_+'" alt="Tp_D_NRN_M_N_nat10143_nat00095_12035_对比"></p><p><img src="'+b+'" alt="Tp_D_NRN_M_N_art00012_art00014_11815_对比"></p><p><img src="'+g+'" alt="Tp_D_NNN_S_N_nat00003_cha00096_00623_对比"></p>',47)])])}const N=n(h,[["render",f]]),z=JSON.parse('{"path":"/posts/ahuviwlz/","title":"1-项目的导入以及跑通baseline | posts","lang":"zh-CN","frontmatter":{"title":"1-项目的导入以及跑通baseline","createTime":"2026/03/26 11:42:09","permalink":"/posts/ahuviwlz/","description":"我们这这次的任务是完成一个图像篡改定位的任务，我们使用的框架是pytorch，编译器是pycharm，在上一节，我们教了如何安装pytorch等环境，这一节我们将学习如何导入环境和跑通baseline 1.导入项目文件 （1）打开Pycharm软件，打开项目文件夹，选择你下载本地的项目文件： image-20260326113335887 （2）打开P...","head":[["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"1-项目的导入以及跑通baseline\\",\\"image\\":[\\"\\"],\\"dateModified\\":\\"2026-04-06T08:51:54.000Z\\",\\"author\\":[]}"],["meta",{"property":"og:url","content":"https://zhaozw-szu.github.io/posts/ahuviwlz/"}],["meta",{"property":"og:site_name","content":"KnowledgeTree"}],["meta",{"property":"og:title","content":"1-项目的导入以及跑通baseline"}],["meta",{"property":"og:description","content":"我们这这次的任务是完成一个图像篡改定位的任务，我们使用的框架是pytorch，编译器是pycharm，在上一节，我们教了如何安装pytorch等环境，这一节我们将学习如何导入环境和跑通baseline 1.导入项目文件 （1）打开Pycharm软件，打开项目文件夹，选择你下载本地的项目文件： image-20260326113335887 （2）打开P..."}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2026-04-06T08:51:54.000Z"}],["meta",{"property":"article:modified_time","content":"2026-04-06T08:51:54.000Z"}]]},"readingTime":{"minutes":6.09,"words":1828},"git":{"createdTime":1775465514000,"updatedTime":1775465514000,"contributors":[{"name":"zhaozw","username":"zhaozw","email":"2300432033@email.szu.edu.cn","commits":1,"avatar":"https://avatars.githubusercontent.com/zhaozw?v=4","url":"https://github.com/zhaozw"}]},"autoDesc":true,"filePathRelative":"posts/1-项目环境的导入以及跑通baseline.md","headers":[],"categoryList":[]}');export{N as comp,z as data};
