"""Test set v1 for development, before the release is uploaded (the release
copies and their case-XX.json win).

    python -m ab_eval.testset cases <folder> [<number>...]
    python -m ab_eval.testset trim <src> <dst> <seconds>
    python -m ab_eval.testset mux <video> <audio> <dst>

``cases`` writes case-XX.json, as the case list of test set v1 gives it,
for every case (or the numbers given) whose files are all in ``folder``
(their sha256 read from the files). ``trim`` keeps the first ``seconds`` by
stream copy (case 03: the first 60 minutes of the Edinburgh walk; case 07:
the first 13 minutes);
``mux`` copies the video stream and adds the first audio stream of another
file as AAC 256 kbit/s (case 01: Meridian's silent picture plus its English
stereo track).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av

AAC_BITRATE = 256_000
COMMONS = "https://commons.wikimedia.org/wiki/File:"
# (files, seconds, music, subtitles, instruction, license, origin)
CASES: dict[str, tuple[int, int, bool, bool, str, str, str]] = {
    "01": (
        1,
        60,
        False,
        True,
        "剪成一条悬疑感的电影预告片，保留关键对白，不剧透结局",
        "CC BY 4.0（Netflix）",
        "https://opencontent.netflix.com/（Meridian）",
    ),
    "02": (
        1,
        90,
        False,
        True,
        "按时间顺序讲完博主在上海的这一天，保留说话的关键片段",
        "CC BY 3.0",
        f"{COMMONS}24_Hours_in_Shanghai_-_City_Walk,_Speaking_Chinese,"
        "_Local_Food_%26_Lamborghini_-_上海.webm",
    ),
    "03": (
        1,
        60,
        True,
        False,
        "剪成一条爱丁堡城市短片，挑最有代表性的街景和建筑",
        "CC BY 3.0",
        f"{COMMONS}Walking_in_EDINBURGH_-_Scotland_(UK)_-_4K_60fps_(UHD).webm",
    ),
    "04": (
        2,
        60,
        False,
        True,
        "把两集合成一条深圳旅行故事，以冲浪为高潮",
        "CC BY 3.0",
        f"{COMMONS}I_Travel_Alone_to_Shenzhen,_China_Ep.4_pt_1.webm",
    ),
    "05": (
        2,
        60,
        True,
        False,
        "把两段合成一条西安一日短片：白天古迹、夜晚大唐不夜城、回民街小吃",
        "CC BY 3.0",
        f"{COMMONS}Xi%27an_Street_Food_Tour.webm",
    ),
    "06": (
        1,
        60,
        True,
        False,
        "按春夏秋冬的顺序剪成一条岩国城市宣传片",
        "CC BY 3.0",
        f"{COMMONS}IWAKUNI_City,_Yamaguchi,_Japan,_4K_Long_Ver..webm",
    ),
    "07": (
        1,
        45,
        True,
        False,
        "挑最漂亮的冰雕和灯光，剪成一条哈尔滨冰雪大世界夜景短片",
        "CC BY 3.0",
        f"{COMMONS}2025_Harbin_Ice_%26_Snow_World_-_Walking_Tour_4K.webm",
    ),
    "08": (
        1,
        60,
        False,
        False,
        "剪成一条不剧透结局的电影预告片",
        "CC BY 3.0",
        f"{COMMONS}Tears_of_Steel_in_4k_-_Official_Blender_Foundation_release.webm",
    ),
    "09": (
        2,
        20,
        False,
        False,
        "把两段合成一条竖屏短视频，剪辑点卡在音乐节拍上",
        "CC BY-SA 4.0（Hakken）",
        f"{COMMONS}Kevin_%26_Perry_Go_Large_trance_party,_Amnesia_Ibiza,_2023.webm",
    ),
    "10": (
        2,
        60,
        True,
        False,
        "把白天的埃菲尔铁塔航拍和夜晚的巴黎街景合成一条从白天到夜晚的巴黎短片",
        "CC BY 3.0",
        f"{COMMONS}Eiffel_Tower,_Paris_2022.webm",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def write_cases(folder: Path, numbers: list[str] | None = None) -> list[str]:
    """case-XX.json for every case (or ``numbers``) whose files are all in
    ``folder``; returns the ids written."""
    written = []
    for number, (count, seconds, music, subs, text, licence, origin) in CASES.items():
        if numbers and number not in numbers:
            continue
        files = [
            next(iter(sorted(folder.glob(f"case-{number}-{k}.*"))), None)
            for k in range(1, count + 1)
        ]
        if any(f is None for f in files):
            continue
        case = {
            "id": f"case-{number}",
            "provider": "A",
            "sources": [{"file": f.name, "sha256": _sha256(f)} for f in files],
            "request": {
                "target_seconds": seconds,
                "format": {"music": music, "subtitles": subs},
                "instruction": text,
            },
            "license": licence,
            "origin": origin,
        }
        (folder / f"case-{number}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(case["id"])
    return written


def trim(src: Path, dst: Path, seconds: float) -> None:
    """Copy every stream of ``src`` up to ``seconds`` into ``dst``."""
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        streams = [s for s in inp.streams if s.type in ("video", "audio")]
        mapped = {s.index: out.add_stream_from_template(s) for s in streams}
        for packet in inp.demux(streams):
            if packet.dts is None:
                continue
            if packet.dts * packet.time_base >= seconds:
                continue
            packet.stream = mapped[packet.stream.index]
            out.mux(packet)


def mux(video: Path, audio: Path, dst: Path) -> None:
    """``video``'s picture stream copied, ``audio``'s first audio stream
    encoded to stereo AAC."""
    with (
        av.open(str(video)) as vin,
        av.open(str(audio)) as ain,
        av.open(str(dst), "w") as out,
    ):
        vsrc = vin.streams.video[0]
        asrc = ain.streams.audio[0]
        vdst = out.add_stream_from_template(vsrc)
        adst = out.add_stream("aac", rate=48000, layout="stereo")
        adst.bit_rate = AAC_BITRATE
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000)
        for packet in vin.demux(vsrc):
            if packet.dts is None:
                continue
            packet.stream = vdst
            out.mux(packet)
        for frame in ain.decode(asrc):
            for piece in resampler.resample(frame):
                out.mux(adst.encode(piece))
        for piece in resampler.resample(None):
            out.mux(adst.encode(piece))
        out.mux(adst.encode(None))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m ab_eval.testset")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_cases = sub.add_parser("cases")
    p_cases.add_argument("folder", type=Path)
    p_cases.add_argument("numbers", nargs="*", help="e.g. 01 03 (default: all)")
    p_trim = sub.add_parser("trim")
    p_trim.add_argument("src", type=Path)
    p_trim.add_argument("dst", type=Path)
    p_trim.add_argument("seconds", type=float)
    p_mux = sub.add_parser("mux")
    p_mux.add_argument("video", type=Path)
    p_mux.add_argument("audio", type=Path)
    p_mux.add_argument("dst", type=Path)
    args = parser.parse_args()
    if args.cmd == "cases":
        print(" ".join(write_cases(args.folder, args.numbers)))
    elif args.cmd == "trim":
        trim(args.src, args.dst, args.seconds)
    else:
        mux(args.video, args.audio, args.dst)


if __name__ == "__main__":
    main()
