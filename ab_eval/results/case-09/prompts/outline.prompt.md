# edit_outline（temperature 0.2）

## system

你是视频剪辑师，负责从原始素材里整理精彩瞬间。只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。

## prompt

用户的剪辑要求：
「把两段合成一条竖屏短视频，剪辑点卡在音乐节拍上」

下面是 2.2 分钟原始素材切成的 5 个场景，按拍摄时间排列，
每个场景有标题和概况。
每行：[场景编号] 起止时间 | 标题 | 概况

[1] 视频 1 0:00–0:30 | 夜店狂欢与演唱会现场 | 视频展示了夜晚夜店人群在蓝色灯光下狂欢，随后切换至演唱会现场观众挥舞手臂观看表演的场景。
[2] 视频 2 0:30–0:54 | 夜间演唱会激光秀 | 夜晚演唱会现场，绿色与蓝色激光束交织，大屏幕播放表演者画面，观众举手互动并举手机拍摄。
[3] 视频 2 0:54–1:18 | 夜店绿光激光秀 | 夜晚的 KTV 或夜店内，大屏幕播放影像，绿色激光束穿梭人群，观众举手机拍摄记录精彩瞬间。
[4] 视频 2 1:18–1:39 | 绿色激光夜店演出 | 夜晚的夜店或 KTV 包厢内，绿色激光束在人群中交织，观众挥舞手臂并举手机拍摄，气氛热烈。
[5] 视频 2 1:39–2:12 | 夜场绿紫激光秀 | 夜晚室内演唱会中，绿色激光与紫色屏幕交织，观众举手挥舞手机观看表演。

这批素材属于「人物与对话」，这类成片常用的结构是：开场最有情绪的一个瞬间 → 交代人物和场合 → 按事情的发展展开
→ 高潮（最热闹或最动人的时刻）→ 收尾（告别、合影或安静的一幕）。
设计时可以参考，但以用户的要求为准。

请按用户的要求，从这些场景里设计一个约 20 秒的成片，按顺序写：
0. title：成片的片名，会作为片头标题出现在画面上，像短片标题那样简短
   （不超过 12 个字），不写时长；
1. materials：通读上面的场景，挑出符合要求的场景，每条写场景编号（scene）和
   这个场景里能用的具体内容（content：地点、主体、画面，不超过 40 个字），
   不超过 8 条；
2. design：用 2–3 句话写成片的设计：开头、中间、结尾分别用 materials 里的
   哪些场景，为什么这样安排；写完对照用户的要求检查内容和先后顺序；
3. pace：成片的剪辑节奏，按用户的要求判断：整段（每段素材完整保留，约 10 秒，
   适合慢慢看的风景、过程）、适中（每段最多 6 秒）、快（每段最多 3 秒，
   适合节奏轻快的混剪）；
4. unmet：要求里提到、但场景里找不到的内容，都找得到就留空；
5. parts：照 design 分成 1–2 个部分（不能超过 2 个），按成片里的
   先后排列，先后由 design 决定，不必按拍摄时间：
   - name 是这一部分的标题，summary 用一句话说明这部分讲什么；
   - weight 是这一部分的分量（1–10 的整数），程序会按分量把 20 秒
     分给各部分，每部分至少 10 秒；
   - scenes 是这一部分要的 1–4 个镜头：query 用一个短语说明要的画面
     （如“远处的城堡”“教堂里的彩色玻璃窗”），所有部分的 query 互不重复；
     sources 写这个画面取自 materials 里的哪几个场景（1–3 个编号）。
程序会按每个 query 从全部场景里检索一个最符合的场景，一个场景只用一次。
只输出 JSON。

## schema

```json
{
 "type": "object",
 "properties": {
  "title": {
   "type": "string",
   "maxLength": 30
  },
  "materials": {
   "type": "array",
   "minItems": 1,
   "maxItems": 8,
   "items": {
    "type": "object",
    "properties": {
     "scene": {
      "type": "integer",
      "minimum": 1
     },
     "content": {
      "type": "string",
      "minLength": 1,
      "maxLength": 40
     }
    },
    "required": [
     "scene",
     "content"
    ]
   }
  },
  "design": {
   "type": "string",
   "minLength": 1,
   "maxLength": 150
  },
  "pace": {
   "type": "string",
   "enum": [
    "整段",
    "适中",
    "快"
   ]
  },
  "unmet": {
   "type": "array",
   "items": {
    "type": "string",
    "minLength": 1,
    "maxLength": 30
   },
   "maxItems": 5
  },
  "parts": {
   "type": "array",
   "minItems": 1,
   "maxItems": 2,
   "items": {
    "type": "object",
    "properties": {
     "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 20
     },
     "summary": {
      "type": "string",
      "minLength": 1,
      "maxLength": 80
     },
     "weight": {
      "type": "integer",
      "minimum": 1,
      "maximum": 10
     },
     "scenes": {
      "type": "array",
      "minItems": 1,
      "maxItems": 4,
      "items": {
       "type": "object",
       "properties": {
        "query": {
         "type": "string",
         "minLength": 1,
         "maxLength": 20
        },
        "sources": {
         "type": "array",
         "minItems": 1,
         "maxItems": 3,
         "items": {
          "type": "integer",
          "minimum": 1
         }
        }
       },
       "required": [
        "query",
        "sources"
       ]
      }
     }
    },
    "required": [
     "name",
     "summary",
     "weight",
     "scenes"
    ]
   }
  }
 },
 "required": [
  "title",
  "materials",
  "design",
  "pace",
  "unmet",
  "parts"
 ]
}
```
