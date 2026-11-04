# edit_review（temperature 0.2）

## system

你是视频剪辑师，负责从原始素材里整理精彩瞬间。只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。

## prompt

你是这条成片的剪辑总监。程序已经按大纲选出一版粗剪，请审片并定出最终剪辑。

用户的剪辑要求：「把两集合成一条深圳旅行故事，以冲浪为高潮」
成片时长：60 秒，最终各镜头秒数合计在 59–66 秒之间。
片名：深圳，去冲浪
大纲（按播放顺序）：
- 雨中的深圳（约 12 秒）：书架隧道里的冲浪板开场，雨天抵达深圳
- 一路向海（约 12 秒）：夜里骑摩托出发，一路来到海边悬崖
- 下水之前（约 12 秒）：冲浪店前交谈，给冲浪板涂蜡
- 冲浪（约 13 秒）：海上划水、第一视角冲浪，黄昏抱板走进海浪
- 海边黄昏（约 11 秒）：黄昏的海边沙滩，后备箱里露营收尾

这批素材属于「旅行 vlog」，这类成片常用的结构是：开场最抓人的一个画面 → 出发与在路上 → 按地点依次展开，每地一两个
最好的镜头 → 最高潮的地点或体验 → 收尾（回望、夜景或离开）。
设计时可以参考，但以用户的要求为准。

现在的粗剪，按播放顺序：
1. [雨中的深圳] 用 1.6s ← c1 | 视频1 0:34–0:36 1.6s | s10 | 一名女子手持冲浪板站在螺旋书架隧道中，背景灯光璀璨。 | — | 暖色环形灯带 | 4.6 | 3.7 | 全景 | 对白：0.0–1.6s「This week is getting a little bit crazy I went to one of the richest city in China but get locked in by the military with an X Chinese pro circle tryin…
2. [雨中的深圳] 用 2.4s ← c2 | 视频1 0:47–0:49 2.4s | s18 | 女子坐在酒店门口台阶上，身旁放着黑色行李箱，抬头望向天空或远方。 | — | 自然光，光线充足但柔和 | 3.1 | 3.1 | 中景 | 对白：0.0–2.4s「This week is getting a little bit crazy I went to one of the richest city in China but get locked in by the military with an X Chinese pro circle tryin…
3. [雨中的深圳] 用 6.0s ← c4 | 视频1 1:32–1:41 8.8s | s35 | 女子在出租车内讲述深圳旅游经历，窗外雨景模糊 | — | 车内自然光偏暗 | 2.7 | 2.6 | 近景 | 对白：0.0–8.8s「I am on my way to Shenzhen, I got to the station way too late I missed my train and all the new ticket was sold out for today so the only way for me to…
4. [一路向海] 用 5.1s ← c11 | 视频1 4:24–4:29 5.1s | s60 | 第一人称视角夜间摩托车骑行，车灯照亮前方弯道与路边植被。 | — | 车灯照明，环境较暗 | 3.3 | 2.7 | 特写 | 对白：0.0–5.1s「O我了 don't understand how Gen likeday.」
5. [一路向海] 用 3.9s ← c12 | 视频1 4:29–4:33 3.9s | s61 | 夜间骑行中，戴着头盔的人正在快速行驶，画面带有动态模糊效果。 | — | 路灯照明，局部过曝 | 3.3 | 2.3 | 近景 | 对白：0.0–0.8s「O我了 don't understand how Gen likeday.」
6. [一路向海] 用 3.5s ← c28 | 视频1 10:42–10:46 3.5s | s129 | 镜头俯瞰海边悬崖上的绿色植被与大海，画面稳定且构图开阔。 | — | 自然光，阴天 | 3.9 | 2.1 | 全景 | 对白：0.0–0.7s「就如果你家就已经是在这种地方了，对你还这样有点浪费钱。」 0.7–2.2s「因为你家真的好漂亮。」 2.2–3.5s「是啊，嗯，看到弯嗯前段时间呆的地方，那边有个口，是不是你有没有看到里面的波纹是从这边往里面推进去的，是这样这个方向过去的，今天的浪是有点西南西南。」
7. [下水之前] 用 3.0s ← c38 | 视频1 14:25–14:28 3.0s | s167 | 女子在冲浪店前俯身说话，背景有彩色冲浪板 | — | 自然光充足 | 3.3 | 4.2 | 近景 | 对白：0.0–3.0s「Its a and度个咁但系可以坐个表嗰度嗰标系度系喺嘅啊嘛呢度咧个前滩喺。」
8. [下水之前] 用 2.5s ← c40 | 视频1 15:03–15:05 2.5s | s177 | 一名女子在户外冲浪板店前涂抹冲浪蜡，背景可见多块彩色冲浪板和帐篷。 | — | 自然光，阴天或散射光 | 3.2 | 2.8 | 中景 | 对白：0.0–2.5s「偷嘅感觉全部都系嘅，唔嚟嚟全部咗咗过嚟。」
9. [冲浪] 用 5.7s ← c95 | 视频2 11:29–11:35 5.7s | s186 | 女子在海边凉亭对着镜头自拍说话，背景可见沙滩、远山和海景。 | — | 自然光，阴天 | 3.6 | 4.5 | 近景 | 对白：0.3–5.7s「this from yesterday today I additions think I need go home My body is destroyed by jellyfishs I am very allergic now going to show you my whole body or…
10. [冲浪] 用 6.0s ← c41 | 视频1 15:42–15:50 8.0s | s186 | 女子在海上冲浪板前划水，随后镜头被浪花遮挡进入水下画面 | — | 阴天自然光 | 2.8 | 3.0 | 近景 | 对白：0.0–8.0s「First time you surf in this location it's called Xicchung it's beach bottomom if you've been following this is the second beach bottom we go to and mor…
11. [冲浪] 用 2.7s ← c70 | 视频2 2:04–2:07 2.7s | s46 | 一名女子在傍晚时分手持冲浪板于海浪中行走 | — | 自然光，偏暗 | 3.0 | 2.5 | 远景
12. [冲浪] 用 2.4s ← c71 | 视频2 2:17–2:20 2.4s | s50 | 人物在沙滩上调整冲浪板尾部的黄色鳍片 | — | 自然光，光线充足但略偏暗 | 3.0 | 3.9 | 近景
13. [海边黄昏] 用 1.9s ← c46 | 视频1 17:35–17:37 1.9s | s201 | 一名女子在沙滩更衣区整理泳衣，背景可见冲浪板与折叠躺椅。 | — | 自然光充足 | 3.1 | 4.3 | 近景 | 对白：0.0–1.9s「going to teach you how to change in public without a changing room girls with long hair always always put your hair up first whitet hair tend to stuck …
14. [海边黄昏] 用 6.0s ← c48 | 视频1 18:17–18:24 6.4s | s212 | 女子在户外帐篷旁整理冲浪板，讨论毛巾问题 | — | 自然光，阴天或树荫下 | 2.9 | 3.7 | 中景 | 对白：0.0–6.2s「I do found that thinner tower, it's easier, I used to use my bath towel and honestly that's kind of hard because it's so thick.」
15. [海边黄昏] 用 6.0s ← c49 | 视频1 18:24–18:30 6.4s | s212 | 女子在户外帐篷旁整理衣物，背景是一排彩色冲浪板。 | — | 自然光，柔和偏暗 | 2.9 | 3.0 | 中景 | 对白：0.2–6.3s「If you have a thinner one, you can wear your pants over the towel and once you put on your pants.」

还没用上的候选镜头，按所属部分和检索相关度排列：
【雨中的深圳】
c65 | 视频2 0:00–0:07 7.5s | s0 | 深圳西涌书店内，游客在环形书架隧道中穿行、拍照打卡。 | 深圳西涌书店 | 暖色调室内灯光，柔和均匀 | 4.7 | 2.8 | 全景
c87 | 视频2 8:24–8:26 1.9s | s152 | 两人背对镜头拍摄一个由发光书架组成的螺旋隧道 | — | 暖色室内光，书架自带灯带 | 4.1 | 2.0 | 中景 | 对白：0.0–1.9s「I do get a little bit nervous because all the attention is on me so here's me trying to get the shot as quickly as possible without disturbing everyone…
c66 | 视频2 0:27–0:29 1.6s | s14 | 游客在环形书架隧道中挑选书籍 | — | 暖色灯光 | 4.0 | 2.8 | 全景 | 对白：0.0–1.6s「This week is getting a little bit crazy I went to one of the richest city in China but getlock is an ex Chinese pro surfer trying to show me the best v…
c52 | 视频1 19:38–19:46 7.4s | s233 | 女子在室内对着镜头讲述去冲浪时停车免票的经历。 | — | 室内顶灯，光线均匀 | 3.1 | 4.3 | 近景 | 对白：0.0–7.4s「kind of like the fact that he can just kind of scan his face as an identity like people know him like when we drive in to serve, he doesn't need to pay…
c99 | 视频2 12:44–12:49 5.2s | s201 | 一名女子在书店内戴上耳机浏览书籍和墙上的海报。 | — | 室内暖光 | 3.1 | 2.2 | 中景 | 对白：1.5–2.4s「Yeah.」
c54 | 视频1 20:36–20:44 8.4s | s240 | 一名女子在车内驾驶并对着镜头解说关于海滩管理的英文内容。 | — | 自然光，车厢内略暗但有侧 | 2.9 | 3.1 | 中景 | 对白：0.0–0.1s「经营有钱的话，那你就有钱分。」 0.1–3.5s「然后因为这个沙滩上是他们村的集体产业嘛。」 4.2–8.4s「不用他给门票的话，就是给自己分红嘛。」
c72 | 视频2 2:52–2:55 3.2s | s60 | 两名男子在铺有人造草坪的休息区坐着，前景出现一块冲浪板。 | — | 自然光，光线充足 | 2.7 | 3.5 | 中景 | 对白：0.0–3.2s「几个都没但你看我这一超多 was making fun of me that it never happened to them only to me so I was joking around that maybe just because I'm from the other city why …
c50 | 视频1 19:00–19:08 7.5s | s220 | 两名女性在户外露台交谈后，其中一人驾驶汽车行驶在路上。 | — | 自然光充足 | 3.1 | 2.2 | 中景 | 对白：0.0–6.1s「viky Viy你好你好你好你要拍那个视频的喽，给我先拍吧。」 6.1–7.5s「因为怕明天下大雨，所白秒加速4.4，我们两个怎么都一样，不怕输，我的我。」
c6 | 视频1 2:26–2:35 8.9s | s48 | 女子在酒店房间内对着镜头说话，随后伸手拿取桌上的物品。 | — | 室内暖光照明，画面明亮 | 2.6 | 3.3 | 近景 | 对白：0.2–8.9s「Southern part of China is super humid and right now is super hot in here so let's try to find a seat south of China is like kind of close to tropical w…
c44 | 视频1 16:47–16:54 7.1s | s195 | 海边沙滩上，一名男子坐在冲浪板上与拍摄者进行英语对话互动。 | — | 自然光，多云天气 | 3.2 | 3.4 | 中景
c56 | 视频1 21:03–21:13 9.8s | s245 | 一名女子在车内驾驶，向车外的人询问车辆是否能通过狭窄路段。 | — | 自然光，车内较暗 | 2.8 | 3.2 | 中景 | 对白：0.0–5.7s「Right now she's trying to make a commercial for this a new model of electric car from China, she wanted to put this car where there is so then you can …
c63 | 视频1 23:55–24:03 7.6s | s293 | 女子在室内对着镜头讲述周末经历及朋友留下的物品 | — | 室内顶灯照明，光线充足 | 2.9 | 2.8 | 近景 | 对白：0.0–7.6s「So basically he left the town already, he's gonna to compete I think during the weekend but he's going to drive up with his scooter so he left his boar…
【一路向海】
c8 | 视频1 3:06–3:12 6.7s | s53 | 夜晚街头，两人从摩托车旁走过，随后骑手独自坐在车上，背景是停放的车辆和围栏。 | — | 路灯照明，整体偏暗但有补 | 3.1 | 3.1 | 中景 | 对白：2.4–6.7s「我就得跳不上去，我就得凑不上去。」
c43 | 视频1 16:11–16:16 4.5s | s189 | 第一人称视角拍摄海边冲浪场景，远处有沙滩和山峦，画面中可见冲浪板与海浪。 | — | 阴天，光线柔和 | 3.1 | 3.4 | 全景 | 对白：0.0–4.5s「definitelyite feel spoiled because I got a surf scent at home, it's like a point break, but here this is what we got.」
c13 | 视频1 5:08–5:11 3.0s | s67 | 夜晚街头停放着蓝白相间的摩托车，旁有人物互动并留下头盔。 | — | 路灯照明，车灯反射 | 3.1 | 3.6 | 中景 | 对白：0.1–2.2s「O, one套的.」
c22 | 视频1 8:18–8:21 2.8s | s105 | 一名男子坐在摩托车上，在路边向同伴询问目的地，画面包含街道、车辆和树木。 | — | 自然光，阴天 | 2.8 | 2.5 | 中景 | 对白：0.0–1.4s「我们现在去哪呀？」 1.4–2.8s「找他吧，会回家说的有点点兴奋点好不好？」
c42 | 视频1 16:00–16:06 5.6s | s188 | 第一人称视角记录女子在海浪中冲浪的过程，背景为海滩与山峦。 | — | 阴天，光线柔和 | 3.2 | 3.3 | 特写 | 对白：0.0–5.6s「Seem like challenges have lots of beach bottom bottom with a lawn board often is a challenge for me because I'm not used to it, however I do think that…
c45 | 视频1 16:54–17:00 5.5s | s196 | 女子在浅水区指导男子练习冲浪，背景为海滩与远山 | — | 自然光，多云 | 3.2 | 3.0 | 中景
c92 | 视频2 10:06–10:09 2.7s | s175 | 身穿红色比基尼的女子在沙滩遮阳棚下操作架在三脚架上的长焦相机 | — | 自然光，光线柔和偏暗 | 3.4 | 4.0 | 中景 | 对白：0.0–2.7s「所以说我们就是在这里set up the tripod，然后他把相机调好所有的模式给我摄影速成班，然后我将从这里去朝前拍摄它，你OK吗？」
c62 | 视频1 23:21–23:23 2.5s | s283 | 身穿粉色比基尼的女子在阳台用三脚架相机拍摄海景 | — | 自然光，阴天 | 3.6 | 4.5 | 中景 | 对白：0.0–2.5s「Hello hello sorry I have to stop you here the films are 20 minutes now and I should probably end here the spell is coming tomorrow so if you want to se…
c57 | 视频1 21:40–21:42 2.3s | s253 | 黄昏时分，一名女子躺在沙滩上打开的汽车后备箱里摆拍，背景是海面和远山。 | — | 自然光，黄昏色温偏冷 | 4.1 | 2.3 | 全景 | 对白：0.0–2.3s「'm just see what I can in five this up many mosto anymore.」
c24 | 视频1 9:15–9:18 3.3s | s113 | 人物在车窗边眺望窗外连绵的绿色山景，画面稳定且构图平衡。 | — | 自然光，柔和均匀 | 4.0 | 2.2 | 近景 | 对白：0.0–1.5s「带个三角过来又一个那边又一个，这里又一个，然后这里面又一个这个是最好看的地方吗？」 1.5–3.3s「还是要继续往上走。」
c69 | 视频2 1:31–1:37 6.1s | s40 | 一名女子在海浪上冲浪，周围有其他人在水中等待或游泳。 | — | 自然光，光线充足 | 3.1 | 2.0 | 中景
c17 | 视频1 6:29–6:38 8.2s | s85 | 一名年轻女子在酒店房间内对着镜头说话，背景可见床铺和装饰画。 | — | 室内暖光，光线柔和但略显 | 2.9 | 4.0 | 近景 | 对白：0.0–8.0s「Back on motorcycle again my Kney it's 11 and I can't hold anymore I will give you the update about the boy I just met tonight.」
【下水之前】
c47 | 视频1 17:39–17:45 6.1s | s203 | 女子在户外遮阳棚下整理湿发，背景是彩色冲浪板与绿树。 | — | 自然光，阴天 | 3.3 | 2.4 | 中景 | 对白：0.0–6.1s「going to teach you how to change in public without a changing room girls with long hair always always put your hair up first whitet hair tend to stuck …
c51 | 视频1 19:26–19:29 2.5s | s231 | 一名年轻男子在路边商店前对着镜头说话，背景有车辆和店铺招牌。 | — | 自然光偏暗，黄昏色温 | 2.8 | 4.3 | 近景 | 对白：0.0–0.5s「昨天我刚来的时候，他激我的时候，我就说我能下车吗？」 0.5–2.5s「等你等我肯定等我，他刷完才能进去。」
c90 | 视频2 9:42–9:45 3.1s | s169 | 两名年轻人在酒店前台办理入住手续，其中一人正在整理衣物。 | — | 室内顶灯照明，光线均匀 | 2.9 | 2.4 | 中景 | 对白：0.0–3.1s「I met Sha in the second episode of the Hainan tro she was the first girl to pure shrimp for me within less than 10 minutes of meeting and after that I …
c67 | 视频2 0:59–1:01 2.2s | s29 | 一名女子在印有冲浪图案的集装箱旁整理衣物准备出发 | — | 自然光，光线充足 | 3.0 | 3.0 | 中景 | 对白：0.0–2.2s「This sounds like a propaganda right now, but trust me it is not I got to serve some waves but my whole body got attacked by jlyfish can my whole st my …
c83 | 视频2 7:01–7:04 2.3s | s135 | 一名男子在户外天桥上为一名手持冲浪板的女子拍照，背景是城市高楼与绿树。 | — | 自然光，阴天或散射光 | 3.2 | 2.3 | 中景 | 对白：0.0–2.3s「聚焦就要重新跑 I was lucky enough that my cousin actually live in the city and then he has an employee that have time today so he's gonna to help shooting but…
c59 | 视频1 22:29–22:36 6.7s | s269 | 女子在厨房处理并品尝一种紫色贝类海鲜，背景有洗衣液和窗户。 | — | 室内灯光，偏暗 | 3.1 | 4.3 | 近景 | 对白：0.0–5.6s「哇，动了，明天去抓的哇，没菜吃，我去西班牙，我看他们说有这个吃，我都没找到，真的吗？」 5.6–6.7s「我吃西哦，可以了，，谢谢阿姨。」
c19 | 视频1 7:11–7:15 4.6s | s92 | 一名女子在室内吧台区域制作咖啡，背景可见楼梯和装饰墙贴。 | — | 室内自然光与灯光混合，较 | 2.8 | 2.8 | 中景 | 对白：0.0–4.6s「So I just add hot water in the morning and it turned into a hot Americano This one's not really a hotel but at least it's very clean according to socia…
c64 | 视频1 24:04–24:12 7.5s | s295 | 女子在室内对着镜头说话，背景有抓娃娃机和货架。 | — | 室内顶灯照明，光线均匀 | 2.8 | 2.5 | 中景 | 对白：0.0–7.5s「So basically he left the town already, he's gonna to compete I think during the weekend but he's going to drive up with his scooter so he left his boar…
c60 | 视频1 22:42–22:45 3.2s | s272 | 一名男子在贴有“福”字的室内厨房准备食物，背景有人走动。 | — | 室内自然光，略显昏暗 | 2.3 | 3.3 | 中景 | 对白：0.3–2.8s「好啊，我还饭。」
c18 | 视频1 6:55–6:59 4.1s | s88 | 女子在咖啡店吧台后准备咖啡，画面稳定清晰。 | — | 自然光充足，室内光线明亮 | 3.0 | 2.4 | 中景 | 对白：0.0–4.1s「Hotel won't start serving coffee till  Aam and I usually wake up at  seven a little tip and tricks for you if you are coffee at it as well, I asked the…
c20 | 视频1 7:33–7:41 7.4s | s96 | 第一人称视角展示酒店房卡与价格信息，背景为酒店走廊地毯 | — | 室内灯光 | 2.4 | 4.5 | 特写 | 对白：0.0–7.4s「It's worth it for me to spend a bit more money it costs $55 per night during the weekday and $100 during the weekend It's quite expensive for Chinese S…
c75 | 视频2 3:57–4:01 3.8s | s82 | 两名男子在一家海鲜餐厅内对着镜头开怀大笑交谈。 | — | 室内自然光，光线充足 | 3.3 | 2.7 | 中景 | 对白：0.0–3.8s「And they might invite me to eat lunch so lucky me咁你哋点识噶喺度冲蓝色嘅你哋唔识啊哦坐一齐食饭嘅，一个人上哇。」
【冲浪】
c26 | 视频1 9:49–9:53 4.3s | s120 | 一位年轻女性在户外对着镜头说话，背景有金属塔架和远处的山与湖。 | — | 阴天自然光，偏暗 | 3.1 | 3.5 | 近景 | 对白：0.0–3.9s「Usually supposed to hike up instead of drive up due to this tenderly military zone.」
c81 | 视频2 6:10–6:15 4.9s | s119 | 车内中控屏显示车辆充电动画，手指轻触屏幕操作充电功能。 | — | 自然光，车内较暗 | 2.7 | 2.1 | 近景 | 对白：0.0–0.8s「这什么意思？」 0.8–4.9s「就绿色的就是已经充好电的电池换进来。」
c68 | 视频2 1:06–1:10 3.8s | s32 | 一位年轻女性在沙滩营地对着镜头俏皮眨眼并指眼互动 | — | 自然光充足 | 3.8 | 4.7 | 近景 | 对白：0.0–3.8s「This sounds like a propaganda right now, but trust me it is not I got to serve some waves but my whole body got attacked by jlyfish can my whole st my …
c98 | 视频2 12:38–12:42 3.9s | s199 | 车内一名女子面带微笑，与同伴一同乘车旅行。 | — | 自然光，车厢内光线柔和 | 3.4 | 3.9 | 中景 | 对白：0.0–3.5s「真的从来没有在国内冲过蓝天白云的那，没有过女神活体。」
c7 | 视频1 2:35–2:41 6.1s | s49 | 一名女子在酒店房间内调整空调，画面稳定清晰。 | — | 室内暖光 | 2.6 | 3.7 | 中景 | 对白：0.0–6.1s「Southern part of China is super humid and right now is super hot in here so let's try to find a seat south of China is like kind of close to tropical w…
c5 | 视频1 1:56–2:01 5.5s | s40 | 女子在酒店走廊内对着镜头说话并抬头看上方 | — | 室内顶灯照明，光线充足但 | 2.7 | 2.6 | 近景 | 对白：0.5–1.6s「3 or5.」 2.4–3.3s「3 or five.」 4.1–5.5s「Oh, I think that's me.」
c30 | 视频1 11:20–11:25 4.9s | s135 | 一名男子站在白色建筑前，背景为山林，字幕显示等待时间。 | — | 自然光，柔和 | 2.8 | 3.6 | 中景 | 对白：0.0–1.4s「不可以的吧，就等10分钟左右吧。」 1.4–3.7s「哦，那还好对，我以为是等一天呢？」 3.7–4.9s「哇，怎么可能等一天把车放这里，我们走回去算要等一天的话。」
c21 | 视频1 7:55–7:59 3.8s | s100 | 女子在室内对着镜头微笑并挥手告别 | — | 自然光 | 3.5 | 4.1 | 近景 | 对白：0.0–3.8s「I don't have a surfboard, I don't have wax and I have no way to transport myself to the beach I mean I don't even know the spot well but somehow lucky …
c78 | 视频2 4:58–5:00 2.1s | s102 | 一名背着黑色背包的女子坐在酒店门口台阶上休息，抬头望向天空。 | — | 自然光，柔和均匀 | 3.4 | 3.3 | 中景 | 对白：0.0–2.1s「The location we're supposed to shoot is about three hours drive from here due to traffic and it costs a lot to get there because of how expensive the c…
c55 | 视频1 20:44–20:49 5.0s | s240 | 一名年轻女子在车内驾驶汽车，窗外是绿色的树林景色。 | — | 自然光，车外光线充足 | 3.0 | 3.6 | 中景 | 对白：0.0–4.9s「哦，那他那他都自自己收这个钱的，还干嘛要给呢？」 4.9–5.0s「我觉得因为别人坑是很大。」
c82 | 视频2 6:21–6:24 3.2s | s124 | 车内特写，一名佩戴珍珠项链的女子正对着镜头说话并看向侧方。 | — | 自然光，车窗透入光线 | 3.3 | 4.6 | 近景 | 对白：0.0–3.2s「It's ready let's go so you go to like a station and then they kind of swap the battery for you automatically kind of this is the first time we've ever …
c74 | 视频2 3:25–3:28 2.9s | s71 | 一名穿红色比基尼的女子坐在绿色地垫上的折叠椅上休息，身旁放着白色背包和衣物。 | — | 自然光充足 | 2.1 | 3.8 | 中景 | 对白：0.0–2.9s「Wow this is actually super cool Wee to part two of serving in one of the richest city in China If you miss in part one, go back to last week Wednesday …
【海边黄昏】
c61 | 视频1 22:54–23:04 9.4s | s279 | 三名年轻人在木质结构的简易屋内围坐进食互动 | — | 室内顶灯照明，偏冷色调 | 3.0 | 2.9 | 中景 | 对白：0.0–3.9s「你问我吧，为了剧情，那我下午再去。」 7.0–9.4s「你这次不会的，我准备好了。」
c97 | 视频2 12:15–12:19 4.1s | s193 | 一名女子在酒店房间内整理行李，从站立到蹲下将衣物放入行李箱。 | — | 室内自然光，光线柔和但稍 | 2.8 | 2.0 | 中景 | 对白：0.0–4.1s「If you have been following since the first episode, thank you so much for being on this adventure it makes me feel less lonely and let me know where I …
c25 | 视频1 9:31–9:34 2.9s | s116 | 女子站在废弃停机坪上讲述此处曾是直升机降落点 | — | 阴天自然光 | 3.5 | 4.3 | 近景 | 对白：0.4–2.2s「以前是个直升飞机啊，真的吗？」 2.2–2.9s「真的哇，好漂亮啊，真的的O。」
c76 | 视频2 4:05–4:09 3.7s | s85 | 女子在户外简易棚下用餐，手持食物面带笑容交谈。 | — | 自然光充足 | 3.6 | 4.2 | 近景 | 对白：0.0–3.7s「有两三有两三啊好了公司都是发到哪里的啊youtube上面在话他还可以通过播放量啊对有收嘛。」
c53 | 视频1 20:07–20:13 5.6s | s238 | 女子在室内对着镜头讲述行程费用与独自出行的计划 | — | 室内自然光，光线充足 | 3.1 | 3.9 | 近景 | 对白：0.0–2.2s「So that's 35 yuan per day for year Pass is 200 yuan.」 2.3–5.6s「So just to know that if I go there by myself, I do need to pay因为以前就是.」
c73 | 视频2 3:04–3:08 4.8s | s63 | 一名年轻女子在户外停车场边走边说话，背景有车辆和蓝色楼梯 | — | 自然光，柔和 | 2.9 | 3.9 | 近景 | 对白：0.0–4.8s「几个都没但你看我这一超多 was making fun of me that it never happened to them only to me so I was joking around that maybe just because I'm from the other city why …
c84 | 视频2 7:20–7:30 9.5s | s138 | 一名年轻女子在公园长椅上涂抹防晒霜，随后抬头微笑看向镜头。 | — | 自然光，光线充足 | 3.3 | 4.5 | 近景 | 对白：0.0–9.5s「We are almost done right now we have one more spot to go, we are going to a library trying to get a last shot but unfortunately we have a support so th…
c32 | 视频1 12:09–12:11 2.2s | s142 | 两名年轻人在绿树环绕的山路上边走边聊，镜头跟随拍摄。 | — | 自然光，阴天 | 3.2 | 3.6 | 近景 | 对白：0.0–1.1s「我们几时能出去啊，不知道，看上面这个心情，真的吗？」 1.1–2.2s「我们可以在那个在那个摄摄像头那里说吗？」
c15 | 视频1 5:35–5:42 7.1s | s76 | 一男一女在餐厅内对坐交谈，桌上摆放着饮料和食物。 | — | 室内灯光，亮度正常 | 2.7 | 2.7 | 中景 | 对白：0.0–7.1s「口水 excusecus me咦要我推车冲翻我两蚊做真正咁窝噶，系咩两千加一百咁百。」
c79 | 视频2 5:32–5:37 5.5s | s111 | 一名女子在车内对着镜头说话，背景可见城市夜景和绿植。 | — | 车内暖光，窗外城市灯光 | 3.1 | 4.5 | 近景 | 对白：0.1–5.5s「Instead of charging it, they change the whole battery for, so it only takes three to five minutes, so right now we're going to recharge the car basical…
c91 | 视频2 9:45–9:52 6.7s | s170 | 两名女性在窗边桌前休息交谈，窗外可见街道与绿植。 | — | 自然光，室内略暗 | 2.7 | 2.8 | 中景 | 对白：0.0–6.7s「I met Sha in the second episode of the Hainan tro she was the first girl to pure shrimp for me within less than 10 minutes of meeting and after that I …
c14 | 视频1 5:33–5:35 1.9s | s75 | 一男一女在餐厅内低头看手机，桌上放着饮料和食物。 | — | 室内灯光充足 | 2.8 | 3.3 | 中景 | 对白：0.7–1.9s「口水 excusecus me咦要我推车冲翻我两蚊做真正咁窝噶，系咩两千加一百咁百。」

镜头每行：编号 | 视频与原片时间、时长 | 连续镜头 | 画面描述 | 地标 | 光线 | 精彩度（1–5）| 画质分（1–5）| 景别；有人说话的镜头另列对白
“连续镜头”相同（如都是 s12）的几段出自原片同一个不间断的镜头。

审片时：
- 先看整体：开场第一个镜头要能抓人、交代地点或主角；结尾镜头要有收束感；
  中间按大纲推进；
- 衔接：相邻两个镜头要在景别、角度或场景上有明显变化，不要连着几个几乎
  一样的画面；同一连续镜头的两段不要前后挨着放，那会像跳切；换场景时用
  远景、空镜或一个动作的开始和结束来过渡；同一场景里景别可以由远到近推进，
  运动方向前后一致；
- 节奏：每个镜头的秒数按内容定，动作、人物和快切段落短，风景、建筑和
  需要看清的画面长；不要每段都一样长；
- 画面描述是依据，不要用描述里没有的内容做判断；画质分低或有晃动的镜头
  只在内容不可替代时使用；电影和夜景本来就暗，暗不算毛病；
- 不要用片头片尾字幕、黑场或纯文字画面；
- 有对白的镜头按台词取舍：要用就让秒数盖住完整的句子（对白前的数字是句子在
  镜头里的起止秒数），不要停在半句话上；程序也会把切到的句子补全；
- 可以替换、删除、增加或调整顺序，只能用上面列出的编号，每个编号最多用一次，
  秒数不超过这个镜头的时长。

输出 JSON：
- notes：一两句话说明你对粗剪改了什么、为什么；
- on_beat：剪辑要求让切点卡在音乐节拍上时写 true（程序会把每个镜头的起点
  挪到节拍上、秒数取整拍），否则写 false；
- clips：最终的镜头，按播放顺序，每项 segment 是编号（如 c12），part 是所属部分的
  名称（大纲里的名称之一），seconds 是这个镜头在成片里放多少秒，why 是一句话理由。
只输出 JSON。

## schema

```json
{
 "type": "object",
 "properties": {
  "notes": {
   "type": "string",
   "minLength": 1,
   "maxLength": 200
  },
  "on_beat": {
   "type": "boolean"
  },
  "clips": {
   "type": "array",
   "minItems": 1,
   "items": {
    "type": "object",
    "properties": {
     "segment": {
      "type": "string",
      "pattern": "^c[0-9]+$"
     },
     "part": {
      "type": "string",
      "minLength": 1
     },
     "seconds": {
      "type": "number",
      "minimum": 1.5
     },
     "why": {
      "type": "string",
      "minLength": 1,
      "maxLength": 40
     }
    },
    "required": [
     "segment",
     "part",
     "seconds",
     "why"
    ]
   }
  }
 },
 "required": [
  "notes",
  "on_beat",
  "clips"
 ]
}
```
