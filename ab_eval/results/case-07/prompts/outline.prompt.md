# edit_outline（temperature 0.2）

## system

你是视频剪辑师，负责从原始素材里整理精彩瞬间。只根据给出的描述，不要编造素材里没有的东西。所有文字用简体中文。

## prompt

用户的剪辑要求：
「挑最漂亮的冰雕和灯光，剪成一条哈尔滨冰雪大世界夜景短片」

下面是 13.0 分钟原始素材切成的 5 个场景，按拍摄时间排列，
每个场景有标题和概况。
每行：[场景编号] 起止时间 | 标题 | 概况

[1] 0:00–0:33 | 彩色冰雕夜景 | 夜幕下，巨大的彩色冰雕建筑群在灯光映照下熠熠生辉，游客驻足观赏、拍照打卡，摄影师捕捉精彩瞬间。
[2] 0:33–1:04 | 夜色中的冰雕城堡 | 夜幕下，游客穿梭于巨大的彩色冰雕群与城堡间，蓝紫灯光映照出晶莹剔透的冰雪景观。
[3] 1:04–1:26 | 哈尔滨冰雪大世界夜景 | 夜幕下，哈尔滨冰雪大世界的宏伟冰雕与彩色灯光交相辉映，游客在雪地广场漫步观赏冰雕城堡群。
[4] 5:47–10:19 | 冰雪大世界夜景 | 夜幕下，五彩灯光照亮冰雕城堡与动物造型，游客在雪地广场穿梭观赏，呈现梦幻童话场景。
[5] 12:16–13:00 | 冰雪大世界夜景 | 夜幕下，五彩斑斓的冰雕建筑群在灯光映照下熠熠生辉，游客漫步于雪地与冰雕广场观赏灯光秀。

这批素材属于「城市漫步」，这类成片常用的结构是：定场远景（城市全貌或标志性地标）→ 街巷氛围（走进老街、石板路、拱门）
→ 地标高潮（最好的建筑或地标镜头）→ 人文细节（街头艺人、行人、店铺）
→ 收尾（黄昏、远景或走远的背影）。开场用最抓人的一个镜头。
设计时可以参考，但以用户的要求为准。

请按用户的要求，从这些场景里设计一个约 45 秒的成片，按顺序写：
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
5. parts：照 design 分成 1–4 个部分（不能超过 4 个），按成片里的
   先后排列，先后由 design 决定，不必按拍摄时间：
   - name 是这一部分的标题，summary 用一句话说明这部分讲什么；
   - weight 是这一部分的分量（1–10 的整数），程序会按分量把 45 秒
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
   "maxItems": 4,
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
