# edit_review（temperature 0.2）

## system

你是视频剪辑师，负责从原始素材里整理精彩瞬间。只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。

## prompt

你是这条成片的剪辑总监。程序已经按大纲选出一版粗剪，请审片并定出最终剪辑。

用户的剪辑要求：「把两段合成一条西安一日短片：白天古迹、夜晚大唐不夜城、回民街小吃」
成片时长：60 秒，最终各镜头秒数合计在 59–66 秒之间。
片名：西安一日
大纲（按播放顺序）：
- 白天的古城（约 20 秒）：广仁寺、古城墙和博物馆里的陶马
- 大唐不夜城（约 20 秒）：入夜后灯火辉煌的大唐不夜城
- 回民街小吃（约 20 秒）：炭火烤肉、糯米饼和灌汤包，在小吃里收尾

这批素材属于「旅行 vlog」，这类成片常用的结构是：开场最抓人的一个画面 → 出发与在路上 → 按地点依次展开，每地一两个
最好的镜头 → 最高潮的地点或体验 → 收尾（回望、夜景或离开）。
设计时可以参考，但以用户的要求为准。

现在的粗剪，按播放顺序：
1. [白天的古城] 用 3.2s ← c15 | 视频1 4:26–4:29 3.2s | s50 | 镜头在西安古城墙下移动，两侧可见传统城楼与现代高楼 | 西安城墙 | 阴天自然光，亮度适中 | 3.9 | 2.7 | 远景 | 对白：0.0–3.2s「It's one of the most iconic landmark in the entire city, an ancient bell.」
2. [白天的古城] 用 5.4s ← c11 | 视频1 3:08–3:15 6.6s | s32 | 女子在寺庙前吃冰淇淋并调整墨镜，背景有彩色经幡和游客。 | — | 自然光，明亮 | 3.4 | 4.2 | 近景 | 对白：0.0–1.2s「So they name her the green terror It is one of them.」 2.1–2.5s「もし？」 3.0–6.6s「Famous and religious temple here in this province, and I highly recommend …
3. [白天的古城] 用 5.2s ← c16 | 视频1 4:39–4:45 5.2s | s55 | 从城墙高处俯瞰西安古城墙与远处城市天际线的风景画面 | 西安城墙 | 阴天漫射光，亮度正常 | 3.7 | 2.2 | 全景 | 对白：0.4–4.5s「Oh my god, I'm actually inside, this is so impressive.」
4. [白天的古城] 用 3.3s ← c26 | 视频1 7:25–7:29 3.3s | s77 | 博物馆内展示的四匹陶马雕塑特写，灯光聚焦于展品。 | — | 展厅射灯照明 | 3.6 | 3.0 | 中景 | 对白：0.0–3.3s「Beautiful Jde pieces and intrigucate Bronx vessels that show how people lived.」
5. [白天的古城] 用 5.4s ← c29 | 视频1 8:16–8:21 5.5s | s85 | 博物馆内展示古代陶器与建筑沙盘模型 | — | 室内均匀照明 | 3.1 | 3.0 | 中景 | 对白：0.0–1.7s「That time in the world.」 2.8–5.5s「This was one of the burial sites at that time they started using these cup made from J there are two paid submission …
6. [大唐不夜城] 用 2.8s ← c35 | 视频1 10:02–10:05 2.8s | s100 | 夜幕下的大唐不夜城街头，游客在灯光璀璨的传统建筑与巨型鼓前驻足观赏。 | 大唐不夜城 | 暖色调霓虹灯光与彩色装饰 | 4.1 | 2.8 | 中景 | 对白：0.5–2.8s「At the sleepleless night, the strike, if you can see, there is a fountain here every day at 7 pm.」
7. [大唐不夜城] 用 5.4s ← c40 | 视频1 11:19–11:28 8.7s | s110 | 夜晚灯火通明的商业步行街，游客在挂满粉色灯笼的街道上漫步前行。 | — | 街道灯光与店铺霓虹灯混合 | 3.8 | 3.5 | 中景 | 对白：0.0–8.7s「Both sides of district so if you don't want to snack on street food definitely go to a restaurant and I have been having a lot of snacks already so tha…
8. [大唐不夜城] 用 1.7s ← c86 | 视频2 8:22–8:23 1.7s | s145 | 特写镜头展示酱汁淋在裹粉虾仁与彩椒上，色泽诱人 | — | 柔和暖光，突出食物质感 | 4.9 | 4.3 | 特写 | 对白：0.1–1.7s「Some versions are topped with garlic sauce.」
9. [大唐不夜城] 用 4.6s ← c31 | 视频1 8:50–8:55 4.6s | s91 | 画面从白天的城市广场切换到夜晚灯火通明的街道，展示了不同的旅行场景。 | — | 自然光（白天）、人工照明 | 3.3 | 2.7 | 中景 | 对白：0.0–4.5s「And on the right side is where you have to get the tickets if you already register because it will be on your passport or your ID This is the drum Towe…
10. [大唐不夜城] 用 5.4s ← c33 | 视频1 9:34–9:40 6.4s | s98 | 镜头从街边灌汤包子和羊肉专卖店的招牌特写，推入店内拍摄厨师制作食物的场景。 | — | 店铺灯光与自然光混合，偏 | 3.0 | 3.8 | 中景 | 对白：0.1–5.1s「Now I'm just going to get some dumplings and take it back to the hotel to eat, I am just too exhausted.」
11. [回民街小吃] 用 2.2s ← c66 | 视频2 2:37–2:39 2.2s | s48 | 炭火烤架上滋滋作响的肉串冒着热气，油光发亮。 | — | 暖色火光照明，烟雾中光线 | 4.5 | 2.7 | 近景 | 对白：0.0–2.2s「Yang Ruhuan consists of grilled lamb skewers, marinated in cumin.」
12. [回民街小吃] 用 5.4s ← c67 | 视频2 2:50–2:56 6.2s | s52 | 夜市摊位前，摊主正在整理红柳烤肉串，顾客在旁驻足观看。 | Lao Wang's Skewers（老 | 摊位灯光为主，环境较暗 | 2.7 | 2.9 | 中景 | 对白：0.1–5.8s「Lo Wang Skewers is particularly famous for its generous portions and extra crispy texture.」
13. [回民街小吃] 用 4.7s ← c72 | 视频2 4:25–4:30 4.7s | s80 | 摊位展示着色泽诱人的橘红色糯米饼，摊主正在后厨翻动锅中的食物。 | — | 自然光，摊位灯光辅助 | 4.2 | 3.9 | 近景 | 对白：0.1–4.0s「Traditionally filled with sweet red bean paste or crushed walnuts.」
14. [回民街小吃] 用 5.4s ← c91 | 视频2 9:40–9:46 6.0s | s169 | 西安贾三清真灌汤包子馆内，食客用筷子夹起红油浇淋、撒有葱花的白皮凉粉。 | 西安贾三清真灌汤包子馆 | 室内暖光 | 4.5 | 4.5 | 特写 | 对白：0.1–5.7s「Serving their signature chilled mung bean jelly noodles with an extra dose of numbing sichuan peppercorns.」

还没用上的候选镜头，按所属部分和检索相关度排列：
【白天的古城】
c2 | 视频1 0:32–0:40 7.9s | s11 | 镜头从西安城隍庙内的藏文浮雕墙移向庭院，展示了红墙、彩旗和游客。 | 西安城隍庙 | 自然光，明亮 | 3.7 | 3.0 | 中景 | 对白：0.0–7.9s「They give us free water we napkins Its beautiful You must come here guys And now at the Guangyan Temp, it is the only Lama temple in thesangxi Province…
c27 | 视频1 7:49–7:57 8.2s | s82 | 游客在博物馆内近距离观看并拍摄一件装饰精美的三面棺木展品。 | — | 室内自然光，光线柔和均匀 | 3.5 | 3.1 | 近景 | 对白：0.0–8.1s「This was a stone tomb door, It's a con that I'm trying to show you, it's a three sided coffin it's painted, if carving and everything.」
c3 | 视频1 1:00–1:07 7.1s | s13 | 游客在红色传统建筑庭院内参观，有人拍照、有人静坐祈福。 | — | 自然光，明亮柔和 | 3.3 | 2.3 | 中景 | 对白：0.0–7.1s「So beautiful When Kanhi, the Em Kanhi inspected the northwestern part of China because you know, back then the King, the emperor always have to pay vis…
c21 | 视频1 5:56–6:02 5.7s | s61 | 无人机俯瞰视角展示古城墙与密集的传统民居建筑群，绿树掩映间可见历史建筑细节。 | — | 阴天漫射光 | 3.6 | 2.1 | 远景 | 对白：0.0–5.7s「Booking up here at top of the city wall is one of the best ways to get a bird eye view of the whole city or kind of at least in the ancient times right…
c22 | 视频1 6:23–6:32 9.2s | s63 | 游客在西安古城墙步道上漫步、休息和拍照，远处可见传统建筑群。 | 西安古城墙 | 阴天自然光 | 3.4 | 2.1 | 全景 | 对白：0.0–6.1s「I probably mentioned that earlier that you can rent a bike, right, As it is about 13 to 14 km long.」 6.2–8.3s「It would take you about two hours.」 8.4–9…
c0 | 视频1 0:02–0:04 2.0s | s1 | 一名长发女子在寺庙转经筒前驻足，背景可见其他游客与红柱建筑。 | — | 自然光充足，画面明亮清晰 | 3.3 | 2.5 | 中景 | 对白：0.0–2.0s「I hope you enjoyed the last episode aboutterracotta Warriors.」
c10 | 视频1 2:53–3:01 7.6s | s31 | 游客在布达拉宫广场及红宫前参观游览，画面展示了寺庙建筑与经幡 | 布达拉宫广场、红宫 | 自然光，阴天或散射光 | 3.8 | 2.0 | 中景 | 对白：0.0–7.6s「That time, a very famous one actually was married to Tibet to one of the King or prince in Tibet, he wanted to unite the country of China and Tibet so …
c5 | 视频1 1:23–1:30 7.3s | s17 | 女子在寺庙前转经筒祈福，随后手持经文前往功德箱献金。 | — | 自然光充足 | 3.5 | 2.9 | 中景 | 对白：0.0–2.8s「In other ethnic groups in the Northwest, it is also the only primary worship place for the green Taara in China.」 3.4–7.3s「One and only so I scanned th…
c23 | 视频1 6:37–6:41 4.4s | s65 | 白衣女子在古城墙下沿石阶行走，背景是灰色砖墙与红灯笼。 | — | 自然光，阴天或散射光 | 3.4 | 2.7 | 中景
c18 | 视频1 5:02–5:12 9.9s | s58 | 游客在西安古城墙及城门建筑前驻足、休息与交谈 | 西安古城墙 | 自然光，阴天 | 3.3 | 2.2 | 中景 | 对白：0.0–2.3s「And then they are 13 km long.」 2.4–3.3s「So it is really long.」 3.3–6.0s「That's why you can rent a bike here as well.」 6.1–9.8s「Probably not at this ent…
c9 | 视频1 2:46–2:53 7.6s | s31 | 镜头缓慢上摇展示布达拉宫宏伟的建筑外观与彩旗 | 布达拉宫 | 自然光，光线充足 | 4.3 | 2.7 | 全景 | 对白：0.5–3.7s「Thousands something years ago, one of the princess.」 4.4–7.6s「That time, a very famous one actually was married to Tibet to one of the King or prince i…
c13 | 视频1 3:52–3:54 2.3s | s41 | 一名女子站在故宫红墙前，背景可见黄色琉璃瓦屋顶的建筑。 | 北京故宫 | 自然光，明亮柔和 | 4.0 | 3.3 | 全景 | 对白：0.0–2.3s「I really love this area that was right outside the temple called the corner of Life.」
【大唐不夜城】
c38 | 视频1 10:55–11:04 8.3s | s107 | 游客在夜间灯光璀璨的广场观看喷泉表演，人群熙攘热闹。 | — | 人造灯光，明亮 | 3.7 | 2.2 | 全景 | 对白：0.0–8.3s「So long that this is the area where the show is every night Like I said there is a water foun show every night at7 but times can change so please actua…
c36 | 视频1 10:13–10:16 3.1s | s102 | 夜幕下的大唐不夜城街头人头攒动，背景是巨大的喷泉表演与红色建筑灯光秀。 | 大唐不夜城 | 霓虹灯、舞台灯光、路灯 | 4.3 | 2.5 | 全景 | 对白：0.0–3.1s「I'm not wrong There is a big huge gigantic performance I missed it because I don't have other chance to come but I I would insert maybe a video of it h…
c37 | 视频1 10:40–10:49 9.0s | s105 | 夜晚繁华的商业街区，灯火通明，游客熙攘，有人拍照互动。 | — | 暖色调人造光，路灯与店铺 | 4.0 | 2.1 | 全景 | 对白：0.0–9.0s「Girls, ladies, like the older ladies, even the aunties, they will be dressed in these costumes taking pictures and then the guys also when you're aroun…
c59 | 视频2 0:43–0:48 4.5s | s16 | 夜晚街头老酸奶摊位前，摊主与顾客互动售卖饮品 | — | 暖黄灯光 | 2.8 | 2.6 | 中景 | 对白：0.1–4.2s「Lao Liuhuulaang is one of the oldest and most well loved spots for this dish.」
c20 | 视频1 5:50–5:56 5.7s | s61 | 从高处俯瞰中式传统建筑群与繁忙街道的车流人流景象 | — | 阴天自然光，亮度适中 | 3.5 | 2.4 | 远景 | 对白：0.0–5.7s「Booking up here at top of the city wall is one of the best ways to get a bird eye view of the whole city or kind of at least in the ancient times right…
c14 | 视频1 4:14–4:16 2.2s | s46 | 西安古城墙夜景下，一名女子对着镜头介绍景点，背景有游客经过。 | 西安城墙 | 暖色灯光 | 3.9 | 4.7 | 中景 | 对白：0.0–2.2s「Bils in the 1400, the most woke Cup fortification in the world, Gu.」
c8 | 视频1 2:26–2:30 4.2s | s25 | 游客手持广仁寺祈福签文，展示许愿内容并准备还愿。 | 广仁寺 | 自然光充足 | 3.6 | 3.6 | 特写 | 对白：0.0–0.2s「I'm going to write my wishes and put it on this wall.」 0.3–2.5s「sister, I'm going have to borrow a pen.」 2.6–4.0s「I'm going to write my wishes.」
c42 | 视频1 12:08–12:16 8.2s | s113 | 特写展示精致摆盘的餐厅美食与装饰，包括画笔、荔枝盆栽及甜点。 | — | 室内暖光 | 3.2 | 3.8 | 特写 | 对白：0.0–0.7s「Very sweet tastete just like liy juice is very good.」 0.8–1.4s「I like it.」 1.5–2.0s「by glass.」 2.1–7.3s「I should say that they do have sometimes in bat…
c94 | 视频2 10:35–10:39 3.5s | s186 | 两名食客在户外举着长条状食物展示，背景有蜘蛛网装饰。 | — | 自然光，明亮 | 3.6 | 3.7 | 中景 | 对白：0.0–3.5s「The noodles are hand pulledled to an impressive thickness, making them chewy and hardy.」
c7 | 视频1 2:03–2:09 6.3s | s22 | 女子在拉萨布达拉宫前的藏式神龛前驻足观看，背景可见经幡墙。 | 布达拉宫广场（推测） | 自然光，明亮柔和 | 3.9 | 3.6 | 中景 | 对白：0.5–3.6s「It says it will give you fortune luck.」 4.5–4.9s「た？」 5.3–6.3s「And health.」
c1 | 视频1 0:19–0:24 4.9s | s8 | 女子在景区门口等车前往寺庙，背景有围栏和旅游大巴 | — | 自然光充足 | 2.9 | 4.4 | 近景 | 对白：0.0–4.9s「It's too hot, so gotta have some watermelon waiting for the car to go to this temple is's supposed to be really beautiful.」
c32 | 视频1 9:07–9:14 7.1s | s94 | 游客在夜市摊位前排队购买臭豆腐和烤串等小吃 | — | 暖黄灯光，摊位照明充足 | 3.1 | 3.9 | 中景 | 对白：1.1–3.7s「Let me show you guys some of the food, maybe.」
【回民街小吃】
c65 | 视频2 2:23–2:26 3.1s | s44 | 特写镜头展示一盘热气腾腾的红豆糯米饭，一只手端着盘子呈现食物细节。 | — | 自然光，侧逆光突出食物质 | 3.9 | 3.4 | 近景 | 对白：0.0–3.1s「And they are a go to dessert for locals who want a light, but satisfying treat after a meal.」
c92 | 视频2 10:06–10:12 5.9s | s175 | 特写镜头展示筷子夹起一只热气腾腾、皮薄馅满的生煎包，汤汁微溢。 | — | 暖光，食物色泽突出 | 4.9 | 3.8 | 特写 | 对白：0.1–5.2s「The meat filling is savory and fragrant, often featuring a blend of pork, lamb, or beef.」
c74 | 视频2 4:52–4:58 5.4s | s84 | 特写镜头展示红油汤底中撒入葱花和芝麻的动作过程。 | — | 暖色调室内光，食物色泽鲜 | 4.6 | 3.2 | 特写 | 对白：0.1–5.1s「This tangy and spicy soup is packed with delicate dumplings filled with minced pork and leeks.」
c90 | 视频2 9:22–9:27 4.7s | s163 | 特写镜头展示筷子夹起拌好的凉粉，色泽诱人且可见油光与葱花点缀。 | — | 自然光，明亮柔和 | 4.9 | 4.7 | 特写 | 对白：0.1–4.7s「Cut into soft, slippery noodles and drenched in vinegar, garlic, chili oil, and sesame paste.」
c64 | 视频2 2:12–2:15 3.1s | s39 | 特写镜头展示将刚出锅的红豆糯米饭盛入盘中，米饭洁白油润，红豆饱满，热气腾腾。 | — | 自然光，明亮柔和 | 4.6 | 4.8 | 近景 | 对白：0.1–3.1s「And the honey glaze topping gives it a natural sweetness that isn't overpowering.」
c83 | 视频2 7:30–7:34 4.1s | s128 | 特写展示刚做好的肉夹馍成品及切开的瞬间 | — | 室内顶光，食物表面有油光 | 4.5 | 4.0 | 近景 | 对白：0.0–2.1s「A spicy twist on the classic Rujiammo.」 2.5–4.1s「This version is packed with tender cu spiced lamb.」
c61 | 视频2 1:16–1:20 4.3s | s23 | 特写镜头展示烤炉内正在烘烤的饼类食物，画面聚焦于食物的色泽与状态。 | — | 自然光/室内光 | 4.3 | 4.7 | 近景 | 对白：0.0–4.0s「Fanjirujiyamo is a must visit, serving hand shredded meat inside freshly baked fire grilled flat breadread.」
c89 | 视频2 9:07–9:12 4.8s | s158 | 一碗色泽红亮、热气腾腾的面条特写，筷子夹起面条展示诱人质感。 | — | 暖光，食物色泽突出 | 4.7 | 2.1 | 近景 | 对白：0.1–4.4s「Known for its deep, flavorful beef broth that is slow cooked for hours.」
c58 | 视频2 0:30–0:34 3.9s | s12 | 画面展示了胡辣汤与油条的特写，包含热气腾腾的汤和金黄酥脆的面食。 | — | 自然光，室内照明 | 4.3 | 3.3 | 近景 | 对白：0.0–3.9s「A classic Xan breakfast dish, Huutang is a thick spicy soup made with black pepper, beef, potatoes and vegetables.」
c53 | 视频1 15:02–15:09 6.3s | s127 | 酒店自助餐厅内展示着各式烘焙点心与烤盘菜肴的特写镜头。 | — | 暖色室内灯光，食物色泽鲜 | 3.2 | 4.4 | 特写 | 对白：0.0–3.8s「And then you would get to the pastry stations, pancake, waffle, again, bread pudding.」 4.6–5.5s「Cant。」
c76 | 视频2 5:23–5:28 4.3s | s94 | 特写镜头展示厨师用筷子搅拌砂锅中的汤面，随后换勺子舀汤的动作过程。 | — | 自然光，室内照明充足 | 3.6 | 3.5 | 近景 | 对白：0.0–0.8s「It's comforting warmth and bold seasoning.」 1.2–4.0s「Make it a favorite among both locals and visitors.」
c51 | 视频1 14:41–14:49 7.7s | s125 | 西安一家名为 Jianbing Crepe 的煎饼摊前，有人正在制作和展示特色鸡蛋煎饼。 | — | 暖色调室内灯光，食物区域 | 3.4 | 4.1 | 中景 | 对白：0.0–2.3s「And then over here is one of my favorite stations.」 2.8–7.6s「Jmping it's like a bean curd and egg fun, but you can choose.」

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
