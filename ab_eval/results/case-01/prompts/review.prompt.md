# edit_review（temperature 0.2）

## system

你是视频剪辑师，负责从原始素材里整理精彩瞬间。只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。

## prompt

你是这条成片的剪辑总监。程序已经按大纲选出一版粗剪，请审片并定出最终剪辑。

用户的剪辑要求：「剪成一条悬疑感的电影预告片，保留关键对白，不剧透结局」
成片时长：60 秒，最终各镜头秒数合计在 59–66 秒之间。
片名：子午线
大纲（按播放顺序）：
- 警局里的谜团（约 16 秒）：1947 年的警局，侦探面对一堆线索和争吵
- 追查（约 15 秒）：公路上的电话和雷雨中的海边悬崖，谜团越来越深
- 黑暗中的对峙（约 15 秒）：黑暗里手电、人偶和枪口之间的快切
- 悬念（约 14 秒）：黑暗中出现的白衣女子，留下疑问

这批素材属于「剧情短片」，这类成片常用的结构是：冷开场：一个抓人的画面或一句对白 → 交代世界和主角（时代、地点、人物
身份）→ 谜团或冲突出现 → 节奏加快，几个最紧张的镜头快切 → 收在一个
悬念画面或一句话上，不交代结局。不必按剧情时间顺序；对白挑能交代前提、
制造悬念的句子，保留完整的句子；镜头 1.5–4 秒，对白镜头按句子长短。
设计时可以参考，但以用户的要求为准。

现在的粗剪，按播放顺序：
1. [警局里的谜团] 用 2.8s ← c23 | 1:39–1:45 5.2s | s21 | 一名身穿西装马甲的男子在昏暗房间中说话，背景墙上贴有 LAPD 标志。 | — | 左侧台灯侧光，面部受光较 | 3.5 | 4.1 | 近景 | 对白：0.7–5.1s「A witness a witness, and why didn't you say that in the first place?」
2. [警局里的谜团] 用 2.2s ← c27 | 1:59–2:01 2.2s | s25 | 一名身穿西装的男子在昏暗的室内环境中神情紧张地说话。 | — | 侧光，面部有明显阴影 | 3.9 | 4.5 | 近景 | 对白：0.0–0.2s「I know I don't want to believe him.」 0.8–2.2s「What did he say, he said he was out there walking his dog along the bluffs above El managerador, right?」
3. [警局里的谜团] 用 2.8s ← c15 | 1:04–1:12 7.2s | s14 | 一名中年男子在昏暗房间内低头沉思，神情凝重且充满忧虑。 | — | 侧逆光，窗边光线形成明暗 | 4.0 | 4.6 | 近景 | 对白：0.0–3.5s「I mean I checked them all out, there's no criminal records, no association any criminal organizations, one of the guys was arrested for drunk driving a…
4. [警局里的谜团] 用 2.3s ← c20 | 1:27–1:29 2.3s | s19 | 一名身穿西装马甲的男子在昏暗室内说话，背景墙上倒置写着 LAPD DETECTIVE SQUAD | — | 侧光，面部明暗对比明显 | 3.7 | 4.6 | 近景 | 对白：0.0–1.9s「When you a comedian if that's the case you sure don't want to go out there Yeah look I know what happened all right, they go out to the rock, they take…
5. [警局里的谜团] 用 2.8s ← c10 | 0:36–0:45 9.1s | s10 | 一名身穿西装的男子在昏暗的室内环境中神情凝重地说话。 | — | 低照度侧光 | 2.9 | 4.5 | 近景 | 对白：0.0–2.9s「These guys just disappeared Yeah, on that stretch of road right above El Matador, you know what, with the big rock, that's right.」 4.8–8.9s「Mic Cohen u…
6. [警局里的谜团] 用 2.8s ← c12 | 0:48–0:55 6.4s | s12 | 特写镜头展示钉在软木板上的三张黑白人物照片及周围线索文件。 | — | 室内暖光，局部阴影 | 2.8 | 2.1 | 特写 | 对白：0.0–1.7s「Myom heads, nine at this time, one of these guys was a school teacher.」 2.2–6.0s「Next guy he sold insurance, last guy he was retired.」
7. [追查] 用 2.8s ← c59 | 5:11–5:19 8.3s | s46 | 一名戴帽男子站在敞开的老式汽车旁打电话，神情紧张地交谈。 | — | 阴天漫射光，略带灰暗感 | 2.9 | 3.6 | 中景 | 对白：0.4–2.3s「Yeah, dispatch, this is Captain Fo.」 3.2–7.4s「I'm out here at the El Matador location, there's no sign a De Sullivan.」
8. [追查] 用 2.8s ← c60 | 5:19–5:28 8.4s | s46 | 一名戴帽男子在车内打电话，神情由平静转为惊恐后挂断电话望向远方。 | — | 阴天漫射光 | 4.0 | 3.9 | 中景 | 对白：0.0–1.7s「Now, it's this car.」 2.5–6.0s「Listen, I' going to take a look around before you send anybody out here.」
9. [追查] 用 2.8s ← c65 | 5:57–6:01 3.9s | s49 | 一名戴帽男子在闪电交加的暴风雨夜仰望天空，神情凝重。 | — | 自然光（闪电）与暗调环境 | 4.5 | 3.0 | 近景
10. [追查] 用 2.8s ← c67 | 6:03–6:13 10.0s | s51 | 一名戴帽男子在雷雨夜神情凝重地注视前方，随后镜头切换至翻滚乌云与闪电的特写。 | — | 闪电照明，环境昏暗 | 4.6 | 2.1 | 近景
11. [追查] 用 2.8s ← c69 | 6:21–6:28 7.3s | s52 | 一名男子在暴风雨前的海滩上独自行走，天空乌云密布并伴有闪电。 | — | 阴天自然光，伴有闪电闪光 | 4.2 | 2.9 | 中景
12. [追查] 用 2.8s ← c71 | 6:35–6:41 5.7s | s53 | 乌云密布的天空下，一道闪电劈向岩石峭壁顶端。 | — | 闪电强光与乌云阴影的强烈 | 4.7 | 4.6 | 远景
13. [黑暗中的对峙] 用 2.8s ← c86 | 8:02–8:09 7.3s | s64 | 一名戴帽男子在黑暗中持枪对峙，手持手电筒照亮枪口，表情紧张。 | — | 正面补光，背景全黑 | 4.7 | 3.1 | 中景
14. [黑暗中的对峙] 用 1.9s ← c81 | 7:39–7:41 1.9s | s59 | 一个穿着西装打着红领结的人偶在黑暗背景中凝视镜头 | — | 正面补光，面部清晰 | 4.4 | 4.4 | 特写
15. [黑暗中的对峙] 用 2.8s ← c82 | 7:41–7:47 5.8s | s60 | 一名戴帽风衣男子在黑暗中持枪对峙，神情紧张。 | — | 正面强光照亮人物，背景全 | 4.6 | 2.8 | 中景
16. [黑暗中的对峙] 用 2.8s ← c89 | 8:24–8:31 7.5s | s66 | 一名戴帽子的男子在黑暗背景下神情凝重地站立，面部光影变化明显。 | — | 正面侧光，形成强烈明暗 | 4.3 | 3.8 | 中景
17. [黑暗中的对峙] 用 2.8s ← c91 | 8:39–8:47 8.6s | s68 | 一名戴帽男子在黑暗中突然举起手枪射击 | — | 单侧强逆光 | 4.5 | 2.9 | 中景
18. [悬念] 用 2.8s ← c92 | 8:47–8:50 3.1s | s69 | 一名戴帽穿风衣的男子在黑暗中手持强光手电进行扫描或照明动作。 | — | 单点强光源（手电筒） | 4.2 | 2.5 | 中景 | 对白：1.9–3.1s「LAPPD, put your hands in the here.」
19. [悬念] 用 2.8s ← c94 | 8:52–8:55 3.0s | s71 | 一名戴帽风衣男子在黑暗中持枪瞄准，枪口发出强光。 | — | 枪口强光逆光 | 4.8 | 2.2 | 中景 | 对白：0.9–2.2s「Your hands up now.」
20. [悬念] 用 2.8s ← c99 | 9:23–9:27 4.1s | s75 | 一名身穿西装的男子在黑暗中凝视前方，面部被一道强光从背后照亮。 | — | 逆光（轮廓光） | 4.5 | 4.0 | 中景
21. [悬念] 用 2.8s ← c100 | 9:27–9:30 3.3s | s76 | 画面被垂直分割为左右两部分：左侧是一位身穿白色连衣裙的女性，右侧是一位戴着帽子的男性，两人均处于黑暗背景中。 | — | 侧光，面部受光明显，背景 | 4.1 | 3.6 | 近景
22. [悬念] 用 2.8s ← c101 | 9:30–9:38 8.0s | s77 | 一名身穿浅色吊带裙的女子在纯黑背景下凝视镜头，面部表情平静且略带忧郁。 | — | 正面柔光，面部轮廓清晰 | 3.7 | 2.9 | 中景

还没用上的候选镜头，按所属部分和检索相关度排列：
【警局里的谜团】
c35 | 2:45–2:48 2.7s | s31 | 一名身穿西装的男子在昏暗的室内凝视前方，背景墙上贴着倒置的 LAPD 警徽标志。 | — | 侧光，面部受光不均 | 4.1 | 4.2 | 近景 | 对白：0.0–0.6s「He says they got really, really dark, really fast, and bombs start going off in the air, thunder and lightning, weather like he'd never seen before, co…
c98 | 9:13–9:23 9.8s | s74 | 一名戴帽侦探在黑暗中手持手电筒，神情惊恐地向前移动并说话。 | — | 单点光源（手电筒） | 4.3 | 2.4 | 中景
c40 | 3:11–3:18 6.9s | s35 | 一名男子在昏暗的警局办公室内，面对镜头露出神秘微笑。 | — | 单侧台灯照明，形成强烈明 | 4.1 | 4.7 | 中景 | 对白：0.0–0.5s「He looks back at the weather because it's getting worse and worse and decides to hightail it out of there.」 3.4–5.9s「And that was his story, right?」
c18 | 1:20–1:26 5.8s | s17 | 一名身着西装马甲的男子坐在昏暗办公室内，面对镜头说话并做手势。 | — | 台灯局部照明，整体偏暗 | 2.9 | 2.5 | 中景 | 对白：0.0–5.8s「When you a comedian if that's the case you sure don't want to go out there Yeah look I know what happened all right, they go out to the rock, they take…
c13 | 0:55–0:58 3.6s | s13 | 一名身穿西装马甲的男子坐在昏暗的办公桌后，神情严肃地注视前方。 | — | 台灯侧光，面部阴影明显 | 3.6 | 3.6 | 中景 | 对白：0.7–1.7s「Any connection?」
c37 | 2:55–3:00 4.5s | s33 | 一名男子在昏暗办公室内与另一名男子对峙，阳光透过百叶窗形成强烈光影。 | — | 侧逆光，百叶窗阴影 | 3.9 | 2.4 | 中景 | 对白：0.0–2.8s「She's in a white dress, alabaster's skin, his words, and he says she's staring right up at him.」
c25 | 1:48–1:50 2.0s | s23 | 两名身着复古制服的男子在昏暗办公室内神情凝重地交谈。 | — | 室内自然光与台灯混合，明 | 3.8 | 3.4 | 近景 | 对白：0.7–1.8s「But now you believe them.」
c38 | 3:00–3:05 5.8s | s33 | 一名男子在昏暗办公室内与另一名男子对峙，光线从百叶窗射入形成强烈明暗对比。 | — | 侧逆光，百叶窗投射条纹光 | 4.2 | 2.4 | 中景 | 对白：0.0–2.8s「He sees her, she's beautiful.」 3.1–5.8s「But like she's from some other times, some other place.」
c32 | 2:21–2:25 4.4s | s29 | 一名男子坐在办公桌后与另一名男子对峙，阳光从百叶窗射入形成强烈光影。 | — | 侧逆光，百叶窗条纹光 | 4.1 | 2.6 | 中景 | 对白：0.4–4.4s「And he's thinking to himself how the hell do you get up there, you got to be a goddamn mountain climber, but more importantly he says the guy's acting …
c16 | 1:12–1:16 4.0s | s15 | 一名男子坐在昏暗办公室的桌前，灯光下表情从沉思转为微笑。 | — | 台灯侧光 | 3.1 | 2.6 | 中景 | 对白：1.5–3.4s「Maybe you don't want to go out there, bus.」
c30 | 2:13–2:17 4.0s | s27 | 一名身穿马甲的男子在昏暗的办公室内抬头看向上方，神情从闭目转为惊讶。 | — | 单侧顶光 | 4.0 | 4.0 | 中景 | 对白：0.0–4.0s「And he looks down at the beach at that huge rock, you know, the one that's separated from the mainland right where the ocean meets ashore, his words lo…
c24 | 1:45–1:48 3.5s | s22 | 两名男子在昏暗办公室内对峙，一人站立吸烟，另一人坐于桌前。 | — | 侧逆光，窗户射入强光形成 | 4.0 | 2.7 | 中景 | 对白：0.2–3.4s「Gus, I thought the guy was nuts, what he said.」
【追查】
c75 | 7:02–7:10 7.5s | s56 | 一名男子在洞穴出口处走向海边，背景天空出现闪电与强光。 | — | 逆光、闪电闪光 | 4.4 | 2.1 | 远景
c64 | 5:50–5:57 7.1s | s48 | 一名男子站在悬崖边，背景乌云密布且伴有闪电。 | — | 自然光（阴天/雷雨） | 4.4 | 4.3 | 远景
c61 | 5:28–5:36 8.4s | s46 | 一名男子在阴云密布的天空下从车内探身，手持电话通话。 | — | 阴天漫射光，整体偏暗 | 3.6 | 3.2 | 中景 | 对白：2.9–3.7s「I on minute。」
c52 | 4:22–4:25 3.0s | s43 | 车内后视镜中，一名女子面无表情地凝视前方，背景为模糊的雨景光斑。 | — | 冷色调环境光，车外有散景 | 4.1 | 3.4 | 特写
c53 | 4:25–4:31 6.0s | s44 | 一名身穿西装的男子坐在车内，眼神惊恐地凝视前方。 | — | 冷色调侧光，面部明暗对比 | 4.5 | 2.9 | 特写
c58 | 5:03–5:11 8.4s | s46 | 一名戴帽男子在车内接电话，神情严肃地与人通话。 | — | 阴天自然光，整体偏暗 | 3.1 | 3.9 | 中景
c68 | 6:13–6:21 7.3s | s52 | 一名黑衣人独自走在暴风雨将至的海滩上，远处天空出现闪电。 | — | 阴沉天光，闪电闪光 | 4.1 | 3.8 | 远景
c41 | 3:18–3:24 6.6s | s36 | 一辆黑色老式轿车行驶在蜿蜒的山间公路上，镜头逐渐推近至车头特写。 | — | 自然光，光线充足但画面略 | 3.3 | 3.8 | 远景 | 对白：0.7–1.8s「So now.」 2.6–3.5s「Now.」 3.9–6.0s「I need you to go up there and find that woman.」
c66 | 6:01–6:03 2.8s | s50 | 远景俯瞰海边悬崖下的沙滩与海浪拍打岩石的景象。 | — | 自然光，逆光/侧光 | 2.3 | 3.2 | 远景
c62 | 5:36–5:43 6.8s | s47 | 一名戴帽穿风衣的男子背对镜头，独自走在阴天的海边悬崖小径上。 | — | 阴天的漫射光，整体偏暗 | 3.5 | 3.5 | 中景
c72 | 6:41–6:49 8.2s | s54 | 一名戴帽男子站在暴风雨前的海滩上仰望天空，神情惊恐。 | — | 阴云密布、逆光 | 4.4 | 3.5 | 中景
c47 | 4:01–4:10 8.6s | s39 | 一名西装男子坐在昏暗的车内，神情紧张地环顾四周并做出手势。 | — | 昏暗冷色调，车窗外透入微 | 3.7 | 4.0 | 中景 | 对白：0.7–8.6s「When he looks back at the rock, the guy's gone, but there's a woman there in a long white dress with alabaster skin, and she stared right up at him, no…
【黑暗中的对峙】
c80 | 7:33–7:39 5.6s | s58 | 一名戴帽男子在黑暗中手持手电筒和枪械对峙 | — | 手电筒光与轮廓光 | 4.7 | 2.4 | 中景
c96 | 8:58–9:06 7.4s | s73 | 一名戴帽男子在黑暗中持枪瞄准，枪口微光闪烁，神情紧张。 | — | 单一强光源（枪口微光） | 4.6 | 3.3 | 中景
c83 | 7:47–7:49 2.5s | s61 | 黑色背景下，一个穿着西装打着红领结的人偶面部特写，双眼睁大直视镜头。 | — | 正面补光，面部清晰 | 4.5 | 4.2 | 近景
c22 | 1:35–1:39 4.5s | s20 | 一名男子在昏暗的办公室内神情紧张地说话，光线从百叶窗射入形成光束。 | — | 侧逆光（百叶窗射入） | 3.9 | 3.4 | 中景 | 对白：1.1–4.3s「Somebody was out there the day the last guy disappeared.」
c21 | 1:29–1:35 5.8s | s20 | 一名男子在昏暗办公室内神情紧张地说话，光线从百叶窗射入。 | — | 侧逆光，百叶窗投射的光束 | 3.0 | 3.5 | 中景 | 对白：0.5–1.5s「Not this time.」 2.7–5.8s「Well, for one thing, there's no bodies.」
c34 | 2:35–2:45 10.1s | s30 | 一名男子在昏暗的办公室内情绪激动地激烈争吵或独白。 | — | 侧逆光，窗边强光与室内暗 | 3.8 | 3.3 | 中景 | 对白：0.0–10.1s「He says they got really, really dark, really fast, and bombs start going off in the air, thunder and lightning, weather like he'd never seen before, c…
c33 | 2:25–2:35 9.9s | s30 | 一名男子在昏暗办公室内神情紧张地说话，背景有挂钟和百叶窗。 | — | 侧逆光、百叶窗光影 | 3.2 | 2.3 | 中景 | 对白：0.0–9.5s「And he's thinking to himself how the hell do you get up there, you got to be a goddamn mountain climber, but more importantly he says the guy's acting …
c28 | 2:01–2:07 6.1s | s26 | 一名男子在昏暗房间内神情紧张地说话，光线从百叶窗射入。 | — | 侧逆光，百叶窗透光 | 3.7 | 4.6 | 近景 | 对白：0.0–3.3s「What did he say, he said he was out there walking his dog along the bluffs above El managerador, right?」 3.9–6.1s「And he looks down at the beach at tha…
c17 | 1:16–1:20 4.3s | s16 | 一名男子在昏暗房间内吸烟，烟雾缭绕中神情紧张地凝视前方。 | — | 侧逆光，明暗对比强烈 | 4.3 | 4.4 | 近景 | 对白：0.7–4.3s「When you a comedian if that's the case you sure don't want to go out there Yeah look I know what happened all right, they go out to the rock, they take…
c31 | 2:17–2:21 3.9s | s28 | 一名男子在昏暗的房间内情绪激动地说话，背景有挂钟和百叶窗。 | — | 侧逆光，百叶窗形成光影 | 3.9 | 4.7 | 近景 | 对白：0.0–3.6s「And he looks down at the beach at that huge rock, you know, the one that's separated from the mainland right where the ocean meets ashore, his words lo…
c11 | 0:45–0:48 3.4s | s11 | 一名神情凝重的男子在昏暗的办公室内低头沉思，随后抬头望向窗外。 | — | 侧逆光，窗边强光与室内暗 | 3.8 | 3.1 | 中景 | 对白：0.3–3.4s「Myom heads, nine at this time, one of these guys was a school teacher.」
c26 | 1:50–1:59 8.4s | s24 | 一名男子在昏暗办公室内低头沉思，随后抬头凝视镜头方向。 | — | 侧逆光 | 3.2 | 2.5 | 中景 | 对白：0.7–4.9s「I had him checked out too, guys gotd Eagle Scout so now.」 5.3–6.0s「Now I don't know what.」 7.0–8.4s「I know I don't want to believe him.」
【悬念】
c29 | 2:07–2:13 6.1s | s26 | 一名男子在昏暗房间内神情紧张地说话，背景有挂钟和百叶窗。 | — | 侧逆光，百叶窗形成条纹状 | 3.2 | 4.4 | 近景 | 对白：0.0–6.1s「And he looks down at the beach at that huge rock, you know, the one that's separated from the mainland right where the ocean meets ashore, his words lo…
c39 | 3:05–3:11 5.2s | s34 | 一名男子在昏暗的办公室内说话，神情严肃。 | — | 侧逆光，窗边强光 | 3.0 | 3.2 | 中景 | 对白：0.0–0.4s「But like she's from some other times, some other place.」 1.0–5.2s「He looks back at the weather because it's getting worse and worse and decides to high…
c14 | 0:58–1:04 6.3s | s14 | 一名男子在昏暗房间内神情紧张地说话，背景有挂钟和百叶窗。 | — | 侧逆光、阴影重 | 3.9 | 4.5 | 近景 | 对白：0.1–6.3s「I mean I checked them all out, there's no criminal records, no association any criminal organizations, one of the guys was arrested for drunk driving a…
c36 | 2:48–2:55 7.3s | s32 | 一名男子在昏暗房间内表情惊恐地说话，面部特写捕捉其情绪变化。 | — | 侧逆光，明暗对比强烈 | 4.4 | 4.8 | 近景 | 对白：0.0–1.2s「When he looks back at the rock, guy's gone.」 2.0–3.1s「There's a woman there.」 3.5–7.3s「She's in a white dress, alabaster's skin, his words, and he says…
c110 | 10:53–11:02 9.4s | s78 | 纯黑背景上的白色演职人员名单滚动展示 | — | 均匀白光 | 1.0 | 2.5 | 特写
c107 | 10:25–10:34 9.4s | s78 | 黑底白字制作人员名单画面，显示导演名及出品方信息。 | — | 均匀白光 | 1.0 | 2.3 | 特写
c74 | 6:55–7:02 7.5s | s56 | 一名男子独自站在海边洞穴出口处，背景是阴沉的天空和波涛汹涌的海面。 | — | 自然光，逆光 | 3.5 | 2.5 | 远景
c106 | 10:16–10:25 9.4s | s78 | 黑底白字的制作人员名单滚动展示，包含特效监督、选角及演员名单。 | — | 均匀照明 | 1.0 | 2.5 | 特写
c70 | 6:28–6:35 7.3s | s52 | 一名穿风衣戴帽的男子独自走在阴云密布的海滩上，抬头仰望天空。 | — | 阴天漫射光，背景有逆光 | 3.7 | 3.6 | 中景
c43 | 3:34–3:45 10.5s | s37 | 一名穿西装的男子在车内驾驶，神情严肃地注视前方。 | — | 自然光，车内较暗 | 3.7 | 3.8 | 中景 | 对白：0.0–2.6s「And just somebody was out there the day the last guy disappeared, a witness, Yeah, a witness.」 3.0–10.5s「He said that he was up there walking his dog a…
c8 | 0:23–0:29 6.6s | s8 | 画面中央叠加白色文字'Meridian'，背景为复古街道全景，行人车辆穿行于大型建筑前。 | — | 自然光，整体偏暗 | 3.2 | 2.1 | 全景 | 对白：4.2–6.6s「These guys just disappeared Yeah, on that stretch of road right above El Matador, you know what, with the big rock, that's right.」
c76 | 7:10–7:17 7.5s | s56 | 从洞穴内部向外拍摄的海滩全景，海浪拍打着岩石和沙滩。 | — | 自然光，逆光 | 2.3 | 2.7 | 远景

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
